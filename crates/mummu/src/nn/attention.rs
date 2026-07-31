//! Cache-aware grouped-query attention (GQA), shared by every decoder in the
//! zoo. Covers both proven variants: plain GQA with projection bias (Qwen2)
//! and per-head q/k RMSNorm without bias (LFM2).

use burn::module::Module;
use burn::nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::tensor::{DType, Tensor, TensorData, activation, backend::Backend};

use super::MAX_CONTEXT_TOKENS;
use super::rope::apply_rope;

/// Per-layer KV cache entry: cached keys and values, each `[b, nkv, seq, hd]`.
/// `None` until the layer's first forward (the prompt prefill seeds it).
pub type LayerKv<B> = Option<(Tensor<B, 4>, Tensor<B, 4>)>;

/// Additive causal mask `[1, 1, t, past+t]`: query row `i` (absolute position
/// `past+i`) may attend to key columns `0..=past+i`; future columns get a
/// large negative. `-1e4` (not `-inf`) so the f16 GPU path (max ~65504) never
/// overflows — `exp(-1e4)` is still exactly 0 after softmax.
pub fn causal_mask<B: Backend>(t: usize, past: usize, device: &B::Device) -> Tensor<B, 4> {
    assert!(t >= 1, "causal_mask: need at least one query row, got t=0");
    assert!(
        past + t <= MAX_CONTEXT_TOKENS,
        "causal_mask: position {past}+{t} exceeds MAX_CONTEXT_TOKENS ({MAX_CONTEXT_TOKENS})"
    );
    let kcols = past + t;
    let mut m = vec![0f32; t * kcols];
    for i in 0..t {
        let qpos = past + i;
        for j in (qpos + 1)..kcols {
            m[i * kcols + j] = -1e4;
        }
    }
    // Dtype pinned to the backend TYPE: unspecified-dtype creation follows
    // the per-DEVICE policy another alias sharing the device may have locked.
    let dtype = crate::backend::float_dtype::<B>();
    Tensor::<B, 2>::from_data(TensorData::new(m, [t, kcols]), (device, dtype))
        .reshape([1, 1, t, kcols])
}

/// GQA expand: `[b, nkv, s, hd]` → `[b, nkv*group, s, hd]`, each KV head
/// repeated `group` times contiguously (HF `repeat_kv`).
pub fn repeat_kv<B: Backend>(x: Tensor<B, 4>, group: usize) -> Tensor<B, 4> {
    assert!(group >= 1, "repeat_kv: group must be >= 1");
    if group == 1 {
        return x;
    }
    let [b, nkv, s, hd] = x.dims();
    debug_assert!(nkv >= 1 && hd >= 1, "repeat_kv: degenerate kv shape");
    x.reshape([b, nkv, 1, s, hd])
        .repeat_dim(2, group)
        .reshape([b, nkv * group, s, hd])
}

/// Grouped-query attention with a per-layer KV cache. Field names mirror the
/// HF checkpoint layout (`q_proj`/`k_proj`/`v_proj`/`o_proj`); architectures
/// whose checkpoints differ (LFM2's `out_proj`, `q_layernorm`) remap keys at
/// load time instead of renaming fields.
#[derive(Module, Debug)]
pub struct GqaAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    /// Per-head RMSNorm on q, applied post-projection at `[b, t, nh, hd]`
    /// (LFM2-style). `None` for architectures without it (Qwen2).
    pub q_norm: Option<RmsNorm<B>>,
    /// Per-head RMSNorm on k — present iff `q_norm` is.
    pub k_norm: Option<RmsNorm<B>>,
}

/// Shape/behavior config for [`GqaAttention`].
#[derive(Debug, Clone)]
pub struct GqaAttentionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Projection bias on q/k/v (Qwen2: true; LFM2: false). `o_proj` never
    /// has bias in either.
    pub bias: bool,
    /// Per-head q/k RMSNorm (LFM2: eps of the model; Qwen2: `None`).
    pub qk_norm_eps: Option<f64>,
}

impl GqaAttentionConfig {
    /// Initialize the module (random weights; real weights come from import).
    pub fn init<B: Backend>(&self, device: &B::Device) -> GqaAttention<B> {
        assert!(
            self.num_kv_heads >= 1 && self.num_heads.is_multiple_of(self.num_kv_heads),
            "GQA: num_heads ({}) must be a positive multiple of num_kv_heads ({})",
            self.num_heads,
            self.num_kv_heads
        );
        assert!(
            self.head_dim >= 2 && self.head_dim.is_multiple_of(2),
            "GQA: head_dim must be even and >= 2 for RoPE, got {}",
            self.head_dim
        );
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;
        let norm = |eps: f64| {
            RmsNormConfig::new(self.head_dim)
                .with_epsilon(eps)
                .init(device)
        };
        GqaAttention {
            q_proj: LinearConfig::new(self.hidden_size, q_dim)
                .with_bias(self.bias)
                .init(device),
            k_proj: LinearConfig::new(self.hidden_size, kv_dim)
                .with_bias(self.bias)
                .init(device),
            v_proj: LinearConfig::new(self.hidden_size, kv_dim)
                .with_bias(self.bias)
                .init(device),
            o_proj: LinearConfig::new(q_dim, self.hidden_size)
                .with_bias(false)
                .init(device),
            q_norm: self.qk_norm_eps.map(norm),
            k_norm: self.qk_norm_eps.map(norm),
        }
    }
}

impl<B: Backend> GqaAttention<B> {
    /// Cache-aware forward: RoPE the new q/k at the offset positions, append
    /// the new k/v to this layer's cache, attend over the full cached range.
    ///
    /// `x` is `[b, t, hidden]` — the prompt at prefill (`kv == None`), a
    /// single new token per decode step after. Returns `[b, t, hidden]`.
    #[allow(clippy::too_many_arguments)] // mirrors the proven reference signature
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        cos: &Tensor<B, 4>,
        sin: &Tensor<B, 4>,
        mask: Option<&Tensor<B, 4>>,
        kv: &mut LayerKv<B>,
    ) -> Tensor<B, 3> {
        let [b, t, _h] = x.dims();
        let (nh, nkv, hd) = (num_heads, num_kv_heads, head_dim);
        assert!(
            nkv >= 1 && nh.is_multiple_of(nkv),
            "GQA forward: num_heads ({nh}) must be a positive multiple of num_kv_heads ({nkv})"
        );
        debug_assert!(
            self.q_norm.is_some() == self.k_norm.is_some(),
            "GQA forward: q_norm and k_norm must be both present or both absent"
        );

        // Per-head q/k RMSNorm (when present) applies post-projection, before
        // transpose + RoPE — the LFM2 ordering, validated against Ollama.
        let q = self.q_proj.forward(x.clone()).reshape([b, t, nh, hd]);
        let q = match &self.q_norm {
            Some(norm) => norm.forward(q),
            None => q,
        }
        .swap_dims(1, 2);
        let k_new = self.k_proj.forward(x.clone()).reshape([b, t, nkv, hd]);
        let k_new = match &self.k_norm {
            Some(norm) => norm.forward(k_new),
            None => k_new,
        }
        .swap_dims(1, 2);
        let v_new = self
            .v_proj
            .forward(x)
            .reshape([b, t, nkv, hd])
            .swap_dims(1, 2);

        let q = apply_rope(q, cos, sin);
        let k_new = apply_rope(k_new, cos, sin);

        // Append to (or seed) the cache, then attend over everything so far.
        let (k_all, v_all) = match kv.take() {
            Some((pk, pv)) => (
                Tensor::cat(vec![pk, k_new], 2),
                Tensor::cat(vec![pv, v_new], 2),
            ),
            None => (k_new, v_new),
        };
        *kv = Some((k_all.clone(), v_all.clone()));

        let group = nh / nkv;
        let k = repeat_kv(k_all, group);
        let v = repeat_kv(v_all, group);

        // f32 island: Qwen-class attention logits overflow f16 (max 65504)
        // in the q·kᵀ scores, collapsing softmax to NaN — llama.cpp pins this
        // same matmul to f32 precision for the same reason. Scores + mask +
        // softmax run in f32, the probabilities (all in [0, 1]) return to the
        // ambient dtype for the value matmul. Every cast is a no-op on f32
        // backends.
        let ambient = q.dtype();
        let scale = 1.0 / (hd as f32).sqrt();
        let mut scores = q
            .cast(DType::F32)
            .matmul(k.cast(DType::F32).swap_dims(2, 3))
            .mul_scalar(scale);
        if let Some(m) = mask {
            scores = scores.add(m.clone().cast(DType::F32));
        }
        let probs = activation::softmax(scores, 3).cast(ambient);
        let ctx = probs.matmul(v).swap_dims(1, 2).reshape([b, t, nh * hd]);
        self.o_proj.forward(ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Cpu;
    use crate::nn::rope::rope_tables;

    type Dev = burn::tensor::Device<Cpu>;

    const HIDDEN: usize = 16;
    const HEADS: usize = 4;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 4;
    const THETA: f32 = 1e4;

    fn attn(qk_norm: bool, device: &Dev) -> GqaAttention<Cpu> {
        GqaAttentionConfig {
            hidden_size: HIDDEN,
            num_heads: HEADS,
            num_kv_heads: KV_HEADS,
            head_dim: HEAD_DIM,
            bias: true,
            qk_norm_eps: qk_norm.then_some(1e-5),
        }
        .init(device)
    }

    /// Deterministic pseudo-random input `[1, t, HIDDEN]`.
    fn input(t: usize, seed: f32, device: &Dev) -> Tensor<Cpu, 3> {
        let data: Vec<f32> = (0..t * HIDDEN)
            .map(|i| ((i as f32 + seed) * 0.7).sin())
            .collect();
        Tensor::<Cpu, 2>::from_data(TensorData::new(data, [t, HIDDEN]), device)
            .reshape([1, t, HIDDEN])
    }

    /// Full forward over `t` positions in one call (prefill-style).
    fn full_forward(a: &GqaAttention<Cpu>, x: Tensor<Cpu, 3>, device: &Dev) -> Vec<f32> {
        let t = x.dims()[1];
        let (cos, sin) = rope_tables::<Cpu>(t, 0, HEAD_DIM, THETA, device);
        let mask = causal_mask::<Cpu>(t, 0, device);
        let mut kv: LayerKv<Cpu> = None;
        a.forward(
            x,
            HEADS,
            KV_HEADS,
            HEAD_DIM,
            &cos,
            &sin,
            Some(&mask),
            &mut kv,
        )
        .into_data()
        .to_vec::<f32>()
        .unwrap()
    }

    #[test]
    fn causal_mask_is_strictly_causal() {
        let device = Dev::default();
        let m = causal_mask::<Cpu>(3, 2, &device);
        assert_eq!(m.dims(), [1, 1, 3, 5]);
        let v = m.into_data().to_vec::<f32>().unwrap();
        for i in 0..3 {
            for j in 0..5 {
                let expect = if j > 2 + i { -1e4 } else { 0.0 };
                assert_eq!(v[i * 5 + j], expect, "row {i} col {j}");
            }
        }
        // A single decode step attends to everything cached: all zeros.
        let one = causal_mask::<Cpu>(1, 4, &device);
        assert!(
            one.into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|&x| x == 0.0)
        );
    }

    #[test]
    fn repeat_kv_repeats_each_head_contiguously() {
        let device = Dev::default();
        let x = Tensor::<Cpu, 1>::from_floats([1.0, 2.0, 3.0, 4.0], &device).reshape([1, 2, 1, 2]); // 2 kv heads, hd=2
        let y = repeat_kv(x, 2); // -> 4 heads
        assert_eq!(y.dims(), [1, 4, 1, 2]);
        assert_eq!(
            y.into_data().to_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]
        );
    }

    /// The load-bearing invariant: prefill + one-token-at-a-time decode
    /// through the KV cache must equal one full forward over the same tokens.
    #[test]
    fn kv_cache_decode_matches_full_forward() {
        let device = Dev::default();
        for qk_norm in [false, true] {
            let a = attn(qk_norm, &device);
            let x = input(6, 3.0, &device);

            // Reference: all 6 positions in one causal forward; keep the last row.
            let full = full_forward(&a, x.clone(), &device);
            let last_full = &full[5 * HIDDEN..];

            // Cached: prefill 5, then decode position 5 alone.
            let mut kv: LayerKv<Cpu> = None;
            let prefill = x.clone().narrow(1, 0, 5);
            let (cos, sin) = rope_tables::<Cpu>(5, 0, HEAD_DIM, THETA, &device);
            let mask = causal_mask::<Cpu>(5, 0, &device);
            let _ = a.forward(
                prefill,
                HEADS,
                KV_HEADS,
                HEAD_DIM,
                &cos,
                &sin,
                Some(&mask),
                &mut kv,
            );

            let step = x.narrow(1, 5, 1);
            let (cos1, sin1) = rope_tables::<Cpu>(1, 5, HEAD_DIM, THETA, &device);
            let out = a
                .forward(step, HEADS, KV_HEADS, HEAD_DIM, &cos1, &sin1, None, &mut kv)
                .into_data()
                .to_vec::<f32>()
                .unwrap();

            for (i, (c, f)) in out.iter().zip(last_full).enumerate() {
                assert!(
                    (c - f).abs() < 1e-4,
                    "qk_norm={qk_norm} elem {i}: cached {c} vs full {f}"
                );
            }
        }
    }

    /// Causality: changing a future token must not change an earlier output row.
    #[test]
    fn future_tokens_cannot_affect_past_outputs() {
        let device = Dev::default();
        let a = attn(false, &device);
        let x1 = input(4, 1.0, &device);
        // Same first 3 tokens, different 4th.
        let x2 = Tensor::cat(vec![x1.clone().narrow(1, 0, 3), input(1, 99.0, &device)], 1);
        let (o1, o2) = (full_forward(&a, x1, &device), full_forward(&a, x2, &device));
        // Rows 0..3 identical; row 3 differs.
        for i in 0..3 * HIDDEN {
            assert!((o1[i] - o2[i]).abs() < 1e-6, "past row changed at {i}");
        }
        let last_differs = o1[3 * HIDDEN..]
            .iter()
            .zip(&o2[3 * HIDDEN..])
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(last_differs, "the changed token's own row should differ");
    }

    #[test]
    #[should_panic(expected = "must be a positive multiple")]
    fn init_rejects_indivisible_head_grouping() {
        let device = Dev::default();
        let _ = GqaAttentionConfig {
            hidden_size: HIDDEN,
            num_heads: 3,
            num_kv_heads: 2,
            head_dim: HEAD_DIM,
            bias: false,
            qk_norm_eps: None,
        }
        .init::<Cpu>(&device);
    }
}
