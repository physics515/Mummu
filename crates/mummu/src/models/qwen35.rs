//! Qwen3.5 / Qwen3.8 ("qwen35") hybrid decoder: Gated DeltaNet linear
//! attention on three of every four layers, gated full attention (partial
//! RoPE) on the fourth, SwiGLU MLPs, RMSNorm everywhere, tied or untied head.
//!
//! Ported from llama.cpp's reference (`src/models/qwen35.cpp` +
//! `delta-net-base.cpp`, fetched 2026-08-21), the only implementation with
//! local same-weights parity available (`llama-server` runs these GGUFs).
//! Per layer:
//!
//! - **Full attention** (`(i+1) % full_attention_interval == 0`): the q
//!   projection emits query and a per-head **output gate** interleaved
//!   (`[q_h | gate_h]` per head); per-head q/k RMSNorm; RoPE over only the
//!   first `rope_dim` of the 256-wide heads (the metadata's MRoPE sections
//!   degenerate to standard RoPE for text-only inputs); softmax attention;
//!   then `out ⊙ sigmoid(gate)` before the output projection.
//! - **Gated DeltaNet** (the rest): one projection mixes q/k/v, a second
//!   emits the gate `z`; the mix runs through a depthwise causal conv
//!   (kernel `conv_kernel`, rolling state) + SiLU; q/k are L2-normalized
//!   per head (`x / max(‖x‖, ε)`) and tiled from `n_k_heads` to
//!   `n_v_heads`; the recurrence per head with state `S ∈ R^{d_k×d_v}`:
//!   `S ← S·exp(g);  v̂ = Sᵀk;  S += k(β(v − v̂))ᵀ;  o = Sᵀ(q/√d_k)` with
//!   `β = σ(x·Wβ)` and `g = softplus(x·Wα + dt_bias)·a` (`a` holds
//!   `-exp(A_log)`, negative); the output is gated-RMS-normed
//!   (`RMS(o)·silu(z)` per head) and projected back.
//!
//! The NextN/MTP block some checkpoints append (`nextn_predict_layers = 1`)
//! is a draft head for speculative decoding, unused by the main forward —
//! its tensors are explicitly skipped on import.

use std::path::Path;

use burn::module::{Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{
    Embedding, EmbeddingConfig, Linear, LinearConfig, PaddingConfig1d, RmsNorm, RmsNormConfig,
};
use burn::tensor::{Device, DType, Int, Tensor, TensorData, activation};

use crate::gguf::{GgufFile, GgufMap, GgufTensorInfo, GgufValue};
use crate::import::ImportError;
use crate::models::CausalLm;
use crate::models::qwen2::EosIds;
use crate::quant::QuantPolicy;
use crate::nn::{LayerKv, SwiGluMlp, SwiGluMlpConfig, causal_mask, repeat_kv, rope_tables};

/// Architecture hyperparameters, read from a GGUF header's `qwen35.*`
/// metadata (the family currently ships as GGUF; a safetensors `config.json`
/// path can join later).
#[derive(Debug, Clone)]
pub struct Qwen35Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    /// Trunk layers only — `block_count - nextn_predict_layers`.
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Attention head width (`key_length` == `value_length`; 256 across the
    /// family — decoupled from `hidden_size / num_heads`).
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    /// How many leading dims of each head RoPE rotates (`rope.dimension_count`).
    pub rope_dim: usize,
    /// Layer `i` is full attention iff `(i+1) % interval == 0`.
    pub full_attention_interval: usize,
    /// Depthwise conv kernel length in the DeltaNet mix path.
    pub conv_kernel: usize,
    /// DeltaNet value width (`ssm.inner_size` = `n_v_heads · d_state`).
    pub d_inner: usize,
    /// Per-head key/value width (`ssm.state_size`).
    pub d_state: usize,
    /// DeltaNet key/query heads (`ssm.group_count`).
    pub n_k_heads: usize,
    /// DeltaNet value heads (`ssm.time_step_rank` — llama.cpp's reuse).
    pub n_v_heads: usize,
    pub eos_token_id: EosIds,
}

impl Qwen35Config {
    #[must_use]
    pub fn is_attention(&self, layer: usize) -> bool {
        (layer + 1).is_multiple_of(self.full_attention_interval)
    }

    /// q/k projection width in the DeltaNet mix (`n_k_heads · d_state`).
    #[must_use]
    pub fn key_dim(&self) -> usize {
        self.n_k_heads * self.d_state
    }

    /// Channels through the DeltaNet conv: q + k + v concatenated.
    #[must_use]
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.d_inner
    }

    /// Hyperparameters from a GGUF header's `qwen35.*` metadata.
    pub fn from_gguf(f: &GgufFile) -> Result<Self, String> {
        let arch = f.architecture().unwrap_or("<missing>");
        if arch != "qwen35" {
            return Err(format!("GGUF architecture '{arch}' is not qwen35"));
        }
        let meta_usize = |key: &str| -> Result<usize, String> {
            f.get(key)
                .and_then(GgufValue::as_u64)
                .map(|v| usize::try_from(v).expect("metadata fits usize"))
                .ok_or_else(|| format!("GGUF metadata missing {key}"))
        };
        let meta_f32 = |key: &str| -> Result<f32, String> {
            f.get(key)
                .and_then(GgufValue::as_f32)
                .ok_or_else(|| format!("GGUF metadata missing {key}"))
        };
        let embd = f
            .tensor("token_embd.weight")
            .ok_or("GGUF has no token_embd.weight")?;
        // ggml dims are fastest-varying first: [hidden, vocab].
        let vocab_size = usize::try_from(*embd.dims.get(1).ok_or("token_embd is not 2-D")?)
            .expect("vocab fits usize");

        let block_count = meta_usize("qwen35.block_count")?;
        let nextn = f
            .get("qwen35.nextn_predict_layers")
            .and_then(GgufValue::as_u64)
            .map_or(0, |v| usize::try_from(v).expect("small"));
        if nextn >= block_count {
            return Err(format!(
                "nextn_predict_layers ({nextn}) must be below block_count ({block_count})"
            ));
        }
        let key_length = meta_usize("qwen35.attention.key_length")?;
        let value_length = meta_usize("qwen35.attention.value_length")?;
        if key_length != value_length {
            return Err(format!(
                "key_length ({key_length}) != value_length ({value_length}) is not implemented"
            ));
        }

        let eos = f
            .get("tokenizer.ggml.eos_token_id")
            .and_then(GgufValue::as_u64)
            .ok_or("GGUF metadata missing tokenizer.ggml.eos_token_id")?;

        let cfg = Self {
            vocab_size,
            hidden_size: meta_usize("qwen35.embedding_length")?,
            num_layers: block_count - nextn,
            num_attention_heads: meta_usize("qwen35.attention.head_count")?,
            num_key_value_heads: meta_usize("qwen35.attention.head_count_kv")?,
            head_dim: key_length,
            intermediate_size: meta_usize("qwen35.feed_forward_length")?,
            rms_norm_eps: f64::from(meta_f32("qwen35.attention.layer_norm_rms_epsilon")?),
            rope_theta: meta_f32("qwen35.rope.freq_base")?,
            rope_dim: meta_usize("qwen35.rope.dimension_count")?,
            full_attention_interval: f
                .get("qwen35.full_attention_interval")
                .and_then(GgufValue::as_u64)
                .map_or(4, |v| usize::try_from(v).expect("small")),
            conv_kernel: meta_usize("qwen35.ssm.conv_kernel")?,
            d_inner: meta_usize("qwen35.ssm.inner_size")?,
            d_state: meta_usize("qwen35.ssm.state_size")?,
            n_k_heads: meta_usize("qwen35.ssm.group_count")?,
            n_v_heads: meta_usize("qwen35.ssm.time_step_rank")?,
            eos_token_id: EosIds::One(u32::try_from(eos).map_err(|_| "EOS out of u32")?),
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<(), String> {
        if self.d_inner != self.n_v_heads * self.d_state {
            return Err(format!(
                "ssm.inner_size ({}) != n_v_heads ({}) · d_state ({}) — layout not implemented",
                self.d_inner, self.n_v_heads, self.d_state
            ));
        }
        if !self.n_v_heads.is_multiple_of(self.n_k_heads) {
            return Err(format!(
                "n_v_heads ({}) must be a multiple of n_k_heads ({})",
                self.n_v_heads, self.n_k_heads
            ));
        }
        if self.rope_dim > self.head_dim || !self.rope_dim.is_multiple_of(2) {
            return Err(format!(
                "rope_dim ({}) must be even and <= head_dim ({})",
                self.rope_dim, self.head_dim
            ));
        }
        if self.full_attention_interval == 0 || self.conv_kernel < 2 {
            return Err("degenerate full_attention_interval or conv_kernel".into());
        }
        Ok(())
    }
}

/// `Linear::forward` without touching the weight's shape: burn's Linear
/// unsqueezes the weight to the input rank, and a reshape on **packed**
/// quantized storage is broken in burn 0.21 (physical element count differs
/// from the logical one — measured with Q4 on flex AND wgpu, 2026-08-21).
/// Flatten the FLOAT input instead; the weight goes into the matmul as-is.
/// All qwen35 projections are bias-free.
fn qlinear(l: &Linear, x: Tensor<3>) -> Tensor<3> {
    debug_assert!(l.bias.is_none(), "qwen35 projections are bias-free");
    let [b, t, d_in] = x.dims();
    let w = l.weight.val(); // [in, out]
    let d_out = w.dims()[1];
    x.reshape([b * t, d_in]).matmul(w).reshape([b, t, d_out])
}

/// The 2-D twin of [`qlinear`] (the lm-head path).
fn qlinear2(l: &Linear, x: Tensor<2>) -> Tensor<2> {
    debug_assert!(l.bias.is_none(), "qwen35 projections are bias-free");
    x.matmul(l.weight.val())
}

/// Gated full attention (see the module docs). Field names are this port's
/// own (the family has no HF safetensors convention to mirror yet).
#[derive(Module, Debug)]
pub struct GatedAttention {
    /// Emits `[q_h | gate_h]` interleaved per head — `2·num_heads·head_dim` wide.
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    /// Per-head RMSNorm over `head_dim`.
    pub q_norm: RmsNorm,
    pub k_norm: RmsNorm,
}

impl GatedAttention {
    #[allow(clippy::too_many_arguments)] // mirrors the reference data flow
    fn forward(
        &self,
        x: Tensor<3>,
        cfg: &Qwen35Config,
        cos: &Tensor<4>,
        sin: &Tensor<4>,
        mask: Option<&Tensor<4>>,
        kv: &mut LayerKv,
    ) -> Tensor<3> {
        let [b, t, _] = x.dims();
        let (nh, nkv, hd) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );

        // Split the joint projection into q and gate: per head the layout is
        // [q (hd) | gate (hd)], so a [b, t, nh, 2, hd] view separates them.
        let qg = qlinear(&self.q_proj, x.clone()).reshape([b, t, nh, 2, hd]);
        let q = qg.clone().narrow(3, 0, 1).reshape([b, t, nh, hd]);
        let gate = qg.narrow(3, 1, 1).reshape([b, t, nh, hd]);

        let q = self.q_norm.forward(q).swap_dims(1, 2); // [b, nh, t, hd]
        let k_new = qlinear(&self.k_proj, x.clone()).reshape([b, t, nkv, hd]);
        let k_new = self.k_norm.forward(k_new).swap_dims(1, 2);
        let v_new = qlinear(&self.v_proj, x)
            .reshape([b, t, nkv, hd])
            .swap_dims(1, 2);

        // Partial RoPE: rotate the first rope_dim dims, pass the rest through.
        let rope = |x: Tensor<4>| -> Tensor<4> {
            let rot = x.clone().narrow(3, 0, cfg.rope_dim);
            let rest = x.narrow(3, cfg.rope_dim, hd - cfg.rope_dim);
            let rot = crate::nn::apply_rope(rot, cos, sin);
            Tensor::cat(vec![rot, rest], 3)
        };
        let q = rope(q);
        let k_new = rope(k_new);

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

        // f32 island for the scores — the same overflow guard as GqaAttention.
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
        let ctx = probs.matmul(v); // [b, nh, t, hd]

        // Per-head output gate: out ⊙ sigmoid(gate).
        let gated = ctx
            .swap_dims(1, 2) // [b, t, nh, hd]
            .mul(activation::sigmoid(gate))
            .reshape([b, t, nh * hd]);
        qlinear(&self.o_proj, gated)
    }
}

/// Gated DeltaNet linear attention (see the module docs).
#[derive(Module, Debug)]
pub struct GatedDeltaNet {
    /// Mixes q/k/v: `hidden → 2·key_dim + d_inner`.
    pub qkv_proj: Linear,
    /// The gate `z`: `hidden → d_inner`.
    pub z_proj: Linear,
    /// Per-value-head β logits: `hidden → n_v_heads`.
    pub beta_proj: Linear,
    /// Per-value-head decay logits: `hidden → n_v_heads`.
    pub alpha_proj: Linear,
    /// Decay bias added to the α logits before softplus.
    pub dt_bias: Param<Tensor<1>>,
    /// `-exp(A_log)` — negative per-head decay magnitudes.
    pub a: Param<Tensor<1>>,
    /// Depthwise causal conv over the q/k/v mix, kernel `conv_kernel`.
    pub conv1d: Conv1d,
    /// Gated output RMSNorm over `d_state` (per value head).
    pub norm: RmsNorm,
    pub out_proj: Linear,
}

/// DeltaNet decode state: the rolling conv window and the recurrent memory.
pub struct DeltaState {
    /// Last `conv_kernel - 1` mix columns, `[b, conv_dim, k-1]`.
    pub conv: Option<Tensor<3>>,
    /// Per-head associative memory `[b, n_v_heads, d_state, d_state]`.
    pub state: Option<Tensor<4>>,
}

impl GatedDeltaNet {
    fn forward(
        &self,
        x: Tensor<3>,
        cfg: &Qwen35Config,
        cache: &mut DeltaState,
    ) -> Tensor<3> {
        let [b, t, _] = x.dims();
        let (hk, hv, ds) = (cfg.n_k_heads, cfg.n_v_heads, cfg.d_state);
        let key_dim = cfg.key_dim();
        let conv_dim = cfg.conv_dim();
        let kk = cfg.conv_kernel;
        let device = x.device();

        let mixed = qlinear(&self.qkv_proj, x.clone()); // [b, t, conv_dim]
        let z = qlinear(&self.z_proj, x.clone()); // [b, t, d_inner]
        let beta = activation::sigmoid(qlinear(&self.beta_proj, x.clone())); // [b, t, hv]
        // g = softplus(α + dt_bias) · a, with a = -exp(A_log) < 0.
        let alpha =
            qlinear(&self.alpha_proj, x).add(self.dt_bias.val().reshape([1, 1, hv]));
        let g = activation::softplus(alpha, 1.0).mul(self.a.val().reshape([1, 1, hv]));

        // Depthwise causal conv over the sequence, rolling the decode state
        // exactly like nn::ShortConv (algebraic equivalence proven there).
        let mix_cm = mixed.swap_dims(1, 2); // channel-major [b, conv_dim, t]
        let conv_out = if t > 1 {
            self.conv1d.forward(mix_cm.clone()).narrow(2, 0, t)
        } else {
            let window = match &cache.conv {
                Some(prev) => Tensor::cat(vec![prev.clone(), mix_cm.clone()], 2),
                None => {
                    let pad = Tensor::<3>::zeros([b, conv_dim, kk - 1], &device);
                    Tensor::cat(vec![pad, mix_cm.clone()], 2)
                }
            };
            let w = self.conv1d.weight.val().reshape([1, conv_dim, kk]);
            window.mul(w).sum_dim(2)
        };
        cache.conv = Some({
            let combined = match cache.conv.take() {
                Some(prev) => Tensor::cat(vec![prev, mix_cm], 2),
                None => mix_cm,
            };
            let len = combined.dims()[2];
            if len >= kk - 1 {
                combined.narrow(2, len - (kk - 1), kk - 1)
            } else {
                let pad = Tensor::<3>::zeros([b, conv_dim, (kk - 1) - len], &device);
                Tensor::cat(vec![pad, combined], 2)
            }
        });
        let conv_out = activation::silu(conv_out.swap_dims(1, 2)); // [b, t, conv_dim]

        // Split into q/k/v and L2-normalize q/k per head: x / max(‖x‖, ε).
        let eps = cfg.rms_norm_eps as f32;
        let l2 = |x: Tensor<4>| -> Tensor<4> {
            let norm = x
                .clone()
                .powi_scalar(2)
                .sum_dim(3)
                .sqrt()
                .clamp_min(eps);
            x.div(norm)
        };
        let q = l2(conv_out.clone().narrow(2, 0, key_dim).reshape([b, t, hk, ds]));
        let k = l2(conv_out
            .clone()
            .narrow(2, key_dim, key_dim)
            .reshape([b, t, hk, ds]));
        let v = conv_out
            .narrow(2, 2 * key_dim, cfg.d_inner)
            .reshape([b, t, hv, ds]);

        // Tile k-heads across the value heads (llama.cpp's ggml_repeat:
        // head h_v reads k-head h_v % n_k_heads).
        let tile = hv / hk;
        let expand = |x: Tensor<4>| -> Tensor<4> {
            if tile == 1 {
                x
            } else {
                // [b, t, hk, ds] → [b, t, tile·hk, ds] tiling whole blocks.
                x.repeat_dim(2, tile)
            }
        };
        let q = expand(q).swap_dims(1, 2); // [b, hv, t, ds]
        let k = expand(k).swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        // The recurrence, one token at a time. State S[b, h, i, j]:
        // i indexes the key dim, j the value dim.
        let scale = 1.0 / (ds as f32).sqrt();
        let mut s = cache.state.take().unwrap_or_else(|| {
            Tensor::<4>::zeros([b, hv, ds, ds], &device)
        });
        let mut outs: Vec<Tensor<4>> = Vec::with_capacity(t);
        for tau in 0..t {
            let q_t = q.clone().narrow(2, tau, 1).mul_scalar(scale); // [b, hv, 1, ds]
            let k_t = k.clone().narrow(2, tau, 1); // [b, hv, 1, ds]
            let v_t = v.clone().narrow(2, tau, 1); // [b, hv, 1, ds]
            let g_t = g
                .clone()
                .narrow(1, tau, 1)
                .reshape([b, hv, 1, 1]); // per-head decay logit
            let b_t = beta
                .clone()
                .narrow(1, tau, 1)
                .reshape([b, hv, 1, 1]);

            s = s.mul(g_t.exp());
            // v̂[j] = Σ_i S[i, j]·k[i]  — k over the key axis.
            let v_hat = s.clone().mul(k_t.clone().swap_dims(2, 3)).sum_dim(2); // [b, hv, 1, ds]
            let d = v_t.sub(v_hat).mul(b_t); // [b, hv, 1, ds]
            // S += k ⊗ d  (outer product over [key, value]).
            s = s.add(k_t.swap_dims(2, 3).matmul(d.clone()));
            // o[j] = Σ_i S[i, j]·q[i].
            let o = s.clone().mul(q_t.swap_dims(2, 3)).sum_dim(2); // [b, hv, 1, ds]
            outs.push(o);
        }
        cache.state = Some(s);
        let o = Tensor::cat(outs, 2); // [b, hv, t, ds]

        // Gated RMSNorm per value head, then flatten and project out.
        let o = self.norm.forward(o.swap_dims(1, 2)); // [b, t, hv, ds]
        let z = z.reshape([b, t, hv, ds]);
        let gated = o.mul(activation::silu(z)).reshape([b, t, cfg.d_inner]);
        qlinear(&self.out_proj, gated)
    }
}

/// One trunk layer: exactly one of `self_attn` / `linear_attn`.
#[derive(Module, Debug)]
pub struct Qwen35Layer {
    pub input_norm: RmsNorm,
    pub post_attn_norm: RmsNorm,
    pub self_attn: Option<GatedAttention>,
    pub linear_attn: Option<GatedDeltaNet>,
    pub mlp: SwiGluMlp,
}

/// The qwen35 decoder stack.
#[derive(Module, Debug)]
pub struct Qwen35 {
    pub embed_tokens: Embedding,
    pub layers: Vec<Qwen35Layer>,
    pub norm: RmsNorm,
    /// Untied head when the checkpoint carries `output.weight`; tied to the
    /// embedding otherwise.
    pub lm_head: Option<Linear>,
}

/// Per-layer decode cache.
pub enum Qwen35Kv {
    Attn(LayerKv),
    Delta(DeltaState),
}

/// A weight-loaded qwen35 plus its config.
pub struct LoadedQwen35 {
    pub model: Qwen35,
    pub config: Qwen35Config,
    /// `None` for GGUF loads (self-contained; no sibling file).
    pub tokenizer_config: Option<crate::tok_config::TokenizerConfig>,
    /// P9 stage 3(c): remote FFN clusters of a partitioned pack — each
    /// layer's `mlp` then holds only the *local* clusters and the pool adds
    /// the rest (exact when every cluster runs). `None` = plain dense.
    pub ffn_pool: Option<std::sync::Arc<crate::nn::ExpertPool>>,
    /// Opt-in skipping: clusters whose gate energy is below `tau` × the
    /// row's total energy are not computed (lossy — only from a measured
    /// skip table). `0.0` = exact.
    pub ffn_skip_tau: f32,
    /// P9 stage 4: the working-set schedule this model runs under, one entry
    /// per trunk layer. `None` = every cluster is permanently resident (the
    /// tier design), so there is nothing to stage.
    pub ffn_plan: Option<std::sync::Arc<crate::workingset::Plan>>,
}

fn build(cfg: &Qwen35Config, device: &Device, untied_head: bool) -> Qwen35 {
    let norm = |dim: usize, dev: &Device| {
        RmsNormConfig::new(dim)
            .with_epsilon(cfg.rms_norm_eps)
            .init(dev)
    };
    let linear = |inp: usize, out: usize, dev: &Device| {
        LinearConfig::new(inp, out).with_bias(false).init(dev)
    };
    let mlp_cfg = SwiGluMlpConfig {
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
    };
    let layers = (0..cfg.num_layers)
        .map(|i| {
            let attn = cfg.is_attention(i);
            Qwen35Layer {
                input_norm: norm(cfg.hidden_size, device),
                post_attn_norm: norm(cfg.hidden_size, device),
                self_attn: attn.then(|| GatedAttention {
                    q_proj: linear(
                        cfg.hidden_size,
                        2 * cfg.num_attention_heads * cfg.head_dim,
                        device,
                    ),
                    k_proj: linear(
                        cfg.hidden_size,
                        cfg.num_key_value_heads * cfg.head_dim,
                        device,
                    ),
                    v_proj: linear(
                        cfg.hidden_size,
                        cfg.num_key_value_heads * cfg.head_dim,
                        device,
                    ),
                    o_proj: linear(
                        cfg.num_attention_heads * cfg.head_dim,
                        cfg.hidden_size,
                        device,
                    ),
                    q_norm: norm(cfg.head_dim, device),
                    k_norm: norm(cfg.head_dim, device),
                }),
                linear_attn: (!attn).then(|| GatedDeltaNet {
                    qkv_proj: linear(cfg.hidden_size, cfg.conv_dim(), device),
                    z_proj: linear(cfg.hidden_size, cfg.d_inner, device),
                    beta_proj: linear(cfg.hidden_size, cfg.n_v_heads, device),
                    alpha_proj: linear(cfg.hidden_size, cfg.n_v_heads, device),
                    dt_bias: Param::from_tensor(Tensor::zeros([cfg.n_v_heads], device)),
                    a: Param::from_tensor(Tensor::zeros([cfg.n_v_heads], device)),
                    conv1d: Conv1dConfig::new(cfg.conv_dim(), cfg.conv_dim(), cfg.conv_kernel)
                        .with_groups(cfg.conv_dim())
                        .with_padding(PaddingConfig1d::Explicit(
                            cfg.conv_kernel - 1,
                            cfg.conv_kernel - 1,
                        ))
                        .with_bias(false)
                        .init(device),
                    norm: norm(cfg.d_state, device),
                    out_proj: linear(cfg.d_inner, cfg.hidden_size, device),
                }),
                mlp: mlp_cfg.init(device),
            }
        })
        .collect();
    Qwen35 {
        embed_tokens: EmbeddingConfig::new(cfg.vocab_size, cfg.hidden_size).init(device),
        layers,
        norm: norm(cfg.hidden_size, device),
        lm_head: untied_head.then(|| linear(cfg.hidden_size, cfg.vocab_size, device)),
    }
}

/// GGUF (llama.cpp `qwen35` arch) names → this port's parameter paths.
/// `trunk_layers` gates the NextN/MTP block: any `blk.i` at or beyond it is
/// the draft head, deliberately skipped (unused by the main forward).
fn gguf_tensor_map(info: &GgufTensorInfo, trunk_layers: usize) -> Option<GgufMap> {
    match info.name.as_str() {
        "token_embd.weight" => {
            return Some(GgufMap::Rename("model.embed_tokens.weight".into()));
        }
        "output_norm.weight" => return Some(GgufMap::Rename("model.norm.weight".into())),
        "output.weight" => return Some(GgufMap::Rename("lm_head.weight".into())),
        _ => {}
    }
    let rest = info.name.strip_prefix("blk.")?;
    let (layer, field) = rest.split_once('.')?;
    let layer: usize = layer.parse().ok()?;
    if layer >= trunk_layers {
        return Some(GgufMap::Skip); // the NextN/MTP draft block
    }
    if field == "ssm_conv1d.weight" {
        // ggml stores the depthwise kernel squeezed [k, channels]; the
        // Conv1d module wants [channels, 1, k] — same bytes.
        let (&k, &channels) = (info.dims.first()?, info.dims.get(1)?);
        return Some(GgufMap::Reshape(
            format!("model.layers.{layer}.linear_attn.conv1d.weight"),
            vec![channels, 1, k],
        ));
    }
    let mapped = qwen35_field(field)?;
    Some(GgufMap::Rename(format!("model.layers.{layer}.{mapped}")))
}

/// GGUF per-layer field → this port's parameter path (minus the layer
/// prefix). Shared by the GGUF map and the pack loader.
fn qwen35_field(field: &str) -> Option<&'static str> {
    Some(match field {
        "attn_norm.weight" => "input_norm.weight",
        "post_attention_norm.weight" => "post_attn_norm.weight",
        // Full-attention layers.
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "attn_q_norm.weight" => "self_attn.q_norm.weight",
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        // Gated DeltaNet layers.
        "attn_qkv.weight" => "linear_attn.qkv_proj.weight",
        "attn_gate.weight" => "linear_attn.z_proj.weight",
        "ssm_beta.weight" => "linear_attn.beta_proj.weight",
        "ssm_alpha.weight" => "linear_attn.alpha_proj.weight",
        "ssm_dt.bias" => "linear_attn.dt_bias",
        "ssm_a" => "linear_attn.a",
        "ssm_norm.weight" => "linear_attn.norm.weight",
        "ssm_out.weight" => "linear_attn.out_proj.weight",
        // FFN.
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        _ => return None,
    })
}

/// Load a qwen35 model straight from a GGUF file — the classic f32 path,
/// which is [`load_from_gguf_quantized`] with quantization off. One import
/// path serves every precision (P9's "single path" rule).
pub fn load_from_gguf(
    path: &Path,
    device: &Device,
) -> Result<LoadedQwen35, ImportError> {
    load_from_gguf_quantized(path, device, QuantPolicy::Off)
}

/// **Streaming** GGUF import with optional keep-quantized weights: one
/// tensor at a time is dequantized to f32 (whatever the source stored —
/// BF16, K-quants, IQ quants), moved to the device, **re-quantized** per
/// `policy` when eligible, and assigned. Peak memory is the finished model
/// plus a single f32 tensor — never the whole model at f32, which is what
/// makes the 27B tier loadable at all (its f32 form is ~109 GB).
pub fn load_from_gguf_quantized(
    path: &Path,
    device: &Device,
    policy: QuantPolicy,
) -> Result<LoadedQwen35, ImportError> {
    let parse = |reason: String| ImportError::Parse {
        file: path.to_path_buf(),
        reason,
    };
    let f = GgufFile::open(path).map_err(|e| parse(e.to_string()))?;
    let config = Qwen35Config::from_gguf(&f).map_err(parse)?;
    let untied = f.tensor("output.weight").is_some();
    let trunk = config.num_layers;
    let mut model = build(&config, device, untied);

    let mut assigned = 0usize;
    for info in &f.tensors {
        let mapped = gguf_tensor_map(info, trunk)
            .ok_or_else(|| parse(format!("unmapped tensor name '{}'", info.name)))?;
        let (name, shape) = match mapped {
            GgufMap::Skip => continue,
            GgufMap::Rename(name) => (
                name,
                info.dims
                    .iter()
                    .rev()
                    .map(|&d| d as usize)
                    .collect::<Vec<_>>(),
            ),
            GgufMap::Reshape(name, shape) => {
                (name, shape.iter().map(|&d| d as usize).collect())
            }
        };
        let values = f
            .read_tensor_f32(&info.name)
            .map_err(|e| parse(e.to_string()))?;
        assign_param(
            &mut model,
            &name,
            ParamSrc::F32 { values, shape },
            policy,
            device,
        )
        .map_err(parse)?;
        assigned += 1;
    }

    // Both directions of completeness, loudly: every mapped tensor landed
    // (assign_param errors otherwise) and the count matches what the
    // architecture requires.
    let expected = expected_tensor_count(&config, untied);
    if assigned != expected {
        return Err(parse(format!(
            "GGUF supplied {assigned} trunk tensors, the architecture needs {expected}"
        )));
    }
    Ok(LoadedQwen35 {
        model,
        config,
        tokenizer_config: None,
        ffn_pool: None,
        ffn_skip_tau: 0.0,
        ffn_plan: None,
    })
}

/// How many trunk tensors a checkpoint must supply (the completeness gate's
/// other half). Per attention layer 11 (2 norms + q/k/v/o + q/k norm +
/// 3 FFN), per DeltaNet layer 14 (2 norms + qkv/z + β/α/dt/a + conv +
/// ssm-norm + out + 3 FFN), plus embedding, final norm, and the untied head
/// when present.
fn expected_tensor_count(cfg: &Qwen35Config, untied: bool) -> usize {
    let per_layer: usize = (0..cfg.num_layers)
        .map(|i| if cfg.is_attention(i) { 11 } else { 14 })
        .sum();
    per_layer + 2 + usize::from(untied)
}

/// Build a device tensor from row-major f32 `values` of `shape`, cast to the
/// backend float dtype.
fn device_tensor<const D: usize>(
    values: Vec<f32>,
    shape: [usize; D],
    device: &Device,
) -> Tensor<D> {
    let dtype = crate::backend::float_dtype();
    Tensor::from_data(TensorData::new(values, shape), (device, dtype))
}

/// A 2-D **linear weight**: GGUF row-major is `[out, in]`, burn's `Linear`
/// wants `[in, out]` — transpose on device, then quantize when the policy
/// takes it.
fn linear_weight(
    values: Vec<f32>,
    shape: &[usize],
    policy: QuantPolicy,
    device: &Device,
) -> Result<Tensor<2>, String> {
    let &[out, inp] = shape else {
        return Err(format!("linear weight must be 2-D, got {shape:?}"));
    };
    let w = device_tensor::<2>(values, [out, inp], device).swap_dims(0, 1);
    Ok(if policy.eligible(&[inp, out]) {
        crate::quant::quantize_weight(policy, w)
    } else {
        w
    })
}

/// Where a parameter's data comes from: raw f32 (the GGUF streaming path —
/// transposed/quantized here) or a ready 2-D tensor (the pack path — already
/// `[in, out]` at its chosen precision).
pub enum ParamSrc {
    F32 { values: Vec<f32>, shape: Vec<usize> },
    Ready2(Tensor<2>),
}

/// A linear weight from either source.
fn take_linear(
    ready: Option<Tensor<2>>,
    values: Vec<f32>,
    shape: &[usize],
    policy: QuantPolicy,
    device: &Device,
) -> Result<Tensor<2>, String> {
    match ready {
        Some(t) => Ok(t),
        None => linear_weight(values, shape, policy, device),
    }
}

/// Route one mapped tensor into its module field. An unknown path is a loud
/// error — silence here would mean silently dropped weights.
fn assign_param(
    model: &mut Qwen35,
    name: &str,
    src: ParamSrc,
    policy: QuantPolicy,
    device: &Device,
) -> Result<(), String> {
    use burn::module::Param;

    let (values, shape_vec, ready): (Vec<f32>, Vec<usize>, Option<Tensor<2>>) = match src {
        ParamSrc::F32 { values, shape } => (values, shape, None),
        ParamSrc::Ready2(t) => (Vec::new(), t.dims().to_vec(), Some(t)),
    };
    let shape = shape_vec.as_slice();

    let expect_1d = |values: Vec<f32>, shape: &[usize]| -> Result<Tensor<1>, String> {
        let &[n] = shape else {
            return Err(format!("expected 1-D, got {shape:?}"));
        };
        Ok(device_tensor::<1>(values, [n], device))
    };

    match name {
        "model.embed_tokens.weight" => {
            // Embeddings stay float — token gather has no quantized kernel.
            let t = match ready {
                Some(t) => t,
                None => {
                    let &[v, h] = shape else {
                        return Err(format!("embedding must be 2-D, got {shape:?}"));
                    };
                    device_tensor::<2>(values, [v, h], device)
                }
            };
            model.embed_tokens.weight = Param::from_tensor(t);
            return Ok(());
        }
        "model.norm.weight" => {
            model.norm.gamma = Param::from_tensor(expect_1d(values, shape)?);
            return Ok(());
        }
        "lm_head.weight" => {
            let head = model
                .lm_head
                .as_mut()
                .ok_or("checkpoint has output.weight but the model built a tied head")?;
            head.weight = Param::from_tensor(take_linear(ready, values, shape, policy, device)?);
            return Ok(());
        }
        _ => {}
    }

    let rest = name
        .strip_prefix("model.layers.")
        .ok_or_else(|| format!("unknown parameter path '{name}'"))?;
    let (layer, field) = rest
        .split_once('.')
        .ok_or_else(|| format!("bad layer path '{name}'"))?;
    let layer: usize = layer.parse().map_err(|_| format!("bad layer in '{name}'"))?;
    let l = model
        .layers
        .get_mut(layer)
        .ok_or_else(|| format!("layer {layer} out of range"))?;

    let missing_attn = || format!("'{name}' targets an attention block on a DeltaNet layer");
    let missing_delta = || format!("'{name}' targets a DeltaNet block on an attention layer");

    match field {
        "input_norm.weight" => l.input_norm.gamma = Param::from_tensor(expect_1d(values, shape)?),
        "post_attn_norm.weight" => {
            l.post_attn_norm.gamma = Param::from_tensor(expect_1d(values, shape)?);
        }
        "mlp.gate_proj.weight" => {
            l.mlp.gate_proj.weight =
                Param::from_tensor(take_linear(ready, values, shape, policy, device)?);
        }
        "mlp.up_proj.weight" => {
            l.mlp.up_proj.weight = Param::from_tensor(take_linear(ready, values, shape, policy, device)?);
        }
        "mlp.down_proj.weight" => {
            l.mlp.down_proj.weight =
                Param::from_tensor(take_linear(ready, values, shape, policy, device)?);
        }
        _ => {
            if let Some(attn_field) = field.strip_prefix("self_attn.") {
                let attn = l.self_attn.as_mut().ok_or_else(missing_attn)?;
                match attn_field {
                    "q_proj.weight" => {
                        attn.q_proj.weight =
                            Param::from_tensor(take_linear(ready, values, shape, policy, device)?);
                    }
                    "k_proj.weight" => {
                        attn.k_proj.weight =
                            Param::from_tensor(take_linear(ready, values, shape, policy, device)?);
                    }
                    "v_proj.weight" => {
                        attn.v_proj.weight =
                            Param::from_tensor(take_linear(ready, values, shape, policy, device)?);
                    }
                    "o_proj.weight" => {
                        attn.o_proj.weight =
                            Param::from_tensor(take_linear(ready, values, shape, policy, device)?);
                    }
                    "q_norm.weight" => {
                        attn.q_norm.gamma = Param::from_tensor(expect_1d(values, shape)?);
                    }
                    "k_norm.weight" => {
                        attn.k_norm.gamma = Param::from_tensor(expect_1d(values, shape)?);
                    }
                    other => return Err(format!("unknown attention field '{other}'")),
                }
                return Ok(());
            }
            if let Some(delta_field) = field.strip_prefix("linear_attn.") {
                let delta = l.linear_attn.as_mut().ok_or_else(missing_delta)?;
                match delta_field {
                    "qkv_proj.weight" => {
                        delta.qkv_proj.weight =
                            Param::from_tensor(take_linear(ready, values, shape, policy, device)?);
                    }
                    "z_proj.weight" => {
                        delta.z_proj.weight =
                            Param::from_tensor(take_linear(ready, values, shape, policy, device)?);
                    }
                    "beta_proj.weight" => {
                        delta.beta_proj.weight =
                            Param::from_tensor(take_linear(ready, values, shape, policy, device)?);
                    }
                    "alpha_proj.weight" => {
                        delta.alpha_proj.weight =
                            Param::from_tensor(take_linear(ready, values, shape, policy, device)?);
                    }
                    "out_proj.weight" => {
                        delta.out_proj.weight =
                            Param::from_tensor(take_linear(ready, values, shape, policy, device)?);
                    }
                    "dt_bias" => delta.dt_bias = Param::from_tensor(expect_1d(values, shape)?),
                    "a" => delta.a = Param::from_tensor(expect_1d(values, shape)?),
                    "norm.weight" => {
                        delta.norm.gamma = Param::from_tensor(expect_1d(values, shape)?);
                    }
                    "conv1d.weight" => {
                        let &[ch, one, k] = shape else {
                            return Err(format!("conv kernel must be 3-D, got {shape:?}"));
                        };
                        if one != 1 {
                            return Err(format!("conv kernel middle dim must be 1, got {one}"));
                        }
                        delta.conv1d.weight =
                            Param::from_tensor(device_tensor::<3>(values, [ch, 1, k], device));
                    }
                    other => return Err(format!("unknown DeltaNet field '{other}'")),
                }
                return Ok(());
            }
            return Err(format!("unknown layer field '{field}'"));
        }
    }
    Ok(())
}

/// How each GGUF tensor enters a `.mummu` pack (see `crate::pack`).
pub fn pack_actions(
    info: &GgufTensorInfo,
    trunk_layers: usize,
) -> Option<crate::pack::ImportAction> {
    use crate::pack::ImportAction as A;
    match info.name.as_str() {
        "token_embd.weight" => return Some(A::Embedding),
        "output_norm.weight" => return Some(A::Vector),
        "output.weight" => return Some(A::Linear),
        _ => {}
    }
    let rest = info.name.strip_prefix("blk.")?;
    let (layer, field) = rest.split_once('.')?;
    let layer: usize = layer.parse().ok()?;
    if layer >= trunk_layers {
        return Some(A::Skip);
    }
    Some(match field {
        "ssm_conv1d.weight" => A::Conv,
        "attn_norm.weight" | "post_attention_norm.weight" | "attn_q_norm.weight"
        | "attn_k_norm.weight" | "ssm_dt.bias" | "ssm_a" | "ssm_norm.weight" => A::Vector,
        _ => {
            qwen35_field(field)?; // known linear fields only
            A::Linear
        }
    })
}

/// The FFN entry names of every trunk layer — the partitioner's input.
#[must_use]
pub fn ffn_names(trunk_layers: usize) -> Vec<crate::partition::FfnNames> {
    (0..trunk_layers)
        .map(|l| crate::partition::FfnNames {
            gate: format!("blk.{l}.ffn_gate.weight"),
            up: format!("blk.{l}.ffn_up.weight"),
            down: format!("blk.{l}.ffn_down.weight"),
        })
        .collect()
}

/// Pack tensor name → parameter path (the pack keeps GGUF names).
fn pack_param_path(name: &str, trunk_layers: usize) -> Option<String> {
    match name {
        "token_embd.weight" => return Some("model.embed_tokens.weight".into()),
        "output_norm.weight" => return Some("model.norm.weight".into()),
        "output.weight" => return Some("lm_head.weight".into()),
        _ => {}
    }
    let rest = name.strip_prefix("blk.")?;
    let (layer, field) = rest.split_once('.')?;
    let layer: usize = layer.parse().ok()?;
    if layer >= trunk_layers {
        return None;
    }
    if field == "ssm_conv1d.weight" {
        return Some(format!("model.layers.{layer}.linear_attn.conv1d.weight"));
    }
    Some(format!("model.layers.{layer}.{}", qwen35_field(field)?))
}

/// Load from a `.mummu` pack, choosing each tensor's precision through
/// `choose` (the planner's tiering hook — per tensor, so a policy can mix
/// levels). Quantized levels arrive pre-packed (no re-quantization).
pub fn load_from_pack(
    dir: &Path,
    device: &Device,
    choose: &dyn Fn(&crate::pack::TensorEntry) -> crate::pack::Precision,
) -> Result<LoadedQwen35, ImportError> {
    load_from_pack_inner(dir, device, choose, None, None)
}

/// Load a **partitioned** pack with only `local(layer)` FFN clusters in each
/// layer's `mlp` (at the level `choose` picks for that entry); the caller
/// attaches the remote clusters as an `ExpertPool` via [`LoadedQwen35::with_ffn_pool`].
/// Every layer must keep at least one local cluster.
pub fn load_from_pack_partitioned(
    dir: &Path,
    device: &Device,
    choose: &dyn Fn(&crate::pack::TensorEntry) -> crate::pack::Precision,
    local: &dyn Fn(usize) -> Vec<usize>,
) -> Result<LoadedQwen35, ImportError> {
    load_from_pack_inner(dir, device, choose, Some(local), None)
}

/// [`load_from_pack_partitioned`], with the **embedding table pinned to
/// `embed_device`** instead of following the rest of the model.
///
/// An embedding is a gather, not a matmul, so it is the one large tensor a
/// pack never quantizes — on the 27B it is 5.09 GB, a quarter of the model,
/// and it is read once per token while a layer's weights are read 65 times.
/// Holding it on the host frees that VRAM for weights that actually compute,
/// at the cost of one `[1, hidden]` transfer per token.
pub fn load_from_pack_partitioned_split(
    dir: &Path,
    device: &Device,
    embed_device: &Device,
    choose: &dyn Fn(&crate::pack::TensorEntry) -> crate::pack::Precision,
    local: &dyn Fn(usize) -> Vec<usize>,
) -> Result<LoadedQwen35, ImportError> {
    load_from_pack_inner(dir, device, choose, Some(local), Some(embed_device))
}

/// One layer's FFN restricted to `clusters` of a partitioned pack, at
/// `precision`, in Linear layout — the local slab or a remote executor's
/// weights. Columns of gate/up and rows of down are sliced straight from
/// the stored bytes (no re-quantization).
pub fn load_ffn_clusters(
    pack: &crate::pack::Pack,
    layer: usize,
    clusters: &[usize],
    precision: crate::pack::Precision,
    device: &Device,
) -> Result<crate::nn::ExpertWeights, String> {
    use burn::module::Param;
    let part = pack
        .manifest
        .ffn_partition
        .as_ref()
        .ok_or("pack has no FFN partition")?;
    let spans = part.layers.get(layer).ok_or("layer out of range")?;
    let names = &part.names[layer];
    let ranges: Vec<(usize, usize)> = clusters
        .iter()
        .map(|&c| spans.get(c).map(|s| (s.start, s.len)).ok_or_else(|| format!("cluster {c} out of range")))
        .collect::<Result<_, _>>()?;
    if ranges.is_empty() {
        return Err(format!("layer {layer}: empty cluster set"));
    }
    let entry = |name: &str| pack.entry(name).ok_or_else(|| format!("missing {name}"));
    let pick = |e: &crate::pack::TensorEntry| {
        if e.precisions.contains_key(&precision) {
            precision
        } else {
            *e.precisions.keys().max().expect("stored level")
        }
    };
    let g = entry(&names[0])?;
    let u = entry(&names[1])?;
    let d = entry(&names[2])?;
    Ok(crate::nn::ExpertWeights {
        gate: Param::from_tensor(pack.tensor_cols(g, pick(g), &ranges, device)?),
        up: Param::from_tensor(pack.tensor_cols(u, pick(u), &ranges, device)?),
        down: Param::from_tensor(pack.tensor_rows(d, pick(d), &ranges, device)?),
    })
}

impl LoadedQwen35 {
    /// Attach the remote FFN clusters (one pool row per layer, ragged).
    #[must_use]
    pub fn with_ffn_pool(mut self, pool: std::sync::Arc<crate::nn::ExpertPool>) -> Self {
        assert_eq!(pool.num_layers(), self.config.num_layers, "FFN pool must have one row per layer");
        self.ffn_pool = Some(pool);
        self
    }

    /// Opt-in cluster skipping at energy threshold `tau` (see `ffn_skip_tau`).
    #[must_use]
    pub fn with_ffn_skip(mut self, tau: f32) -> Self {
        self.ffn_skip_tau = tau.max(0.0);
        self
    }

    /// Run the FFN clusters as a **working set** under `plan`: each layer
    /// stages what the next needs while it computes, and evicts behind
    /// itself (P9 stage 4). Without a plan every cluster stays permanently
    /// resident, which is the tier design.
    #[must_use]
    pub fn with_ffn_plan(mut self, plan: std::sync::Arc<crate::workingset::Plan>) -> Self {
        self.ffn_plan = Some(plan);
        self
    }
}

fn load_from_pack_inner(
    dir: &Path,
    device: &Device,
    choose: &dyn Fn(&crate::pack::TensorEntry) -> crate::pack::Precision,
    local: Option<&dyn Fn(usize) -> Vec<usize>>,
    embed_device: Option<&Device>,
) -> Result<LoadedQwen35, ImportError> {
    use crate::pack::{Pack, Role};
    let parse = |reason: String| ImportError::Parse {
        file: dir.to_path_buf(),
        reason,
    };
    let pack = Pack::open(dir).map_err(parse)?;
    let header = pack.header().map_err(parse)?;
    let config = Qwen35Config::from_gguf(&header).map_err(parse)?;
    let untied = pack.entry("output.weight").is_some();
    let trunk = config.num_layers;
    let mut model = build(&config, device, untied);

    // Partitioned FFN entries → (layer, proj index) for the local-cluster path.
    let ffn_index: std::collections::HashMap<&str, (usize, usize)> = match (&local, &pack.manifest.ffn_partition) {
        (Some(_), Some(part)) => part
            .names
            .iter()
            .enumerate()
            .flat_map(|(l, n)| n.iter().enumerate().map(move |(i, name)| (name.as_str(), (l, i))))
            .collect(),
        (Some(_), None) => {
            return Err(parse("partitioned load requested but the pack has no FFN partition".into()));
        }
        _ => std::collections::HashMap::new(),
    };
    let mut assigned = 0usize;
    for entry in &pack.manifest.tensors {
        let Some(path) = pack_param_path(&entry.name, trunk) else {
            continue; // NextN block members, if a pack kept any
        };
        if let (Some(local), Some(&(layer, proj))) = (local, ffn_index.get(entry.name.as_str())) {
            let clusters = local(layer);
            let part = pack.manifest.ffn_partition.as_ref().expect("checked above");
            let spans = &part.layers[layer];
            let ranges: Vec<(usize, usize)> = clusters
                .iter()
                .map(|&c| {
                    spans
                        .get(c)
                        .map(|s| (s.start, s.len))
                        .ok_or_else(|| parse(format!("layer {layer}: cluster {c} out of range")))
                })
                .collect::<Result<_, _>>()?;
            if ranges.is_empty() {
                return Err(parse(format!("layer {layer}: no local FFN cluster (every layer needs one)")));
            }
            let precision = {
                let p = choose(entry);
                if entry.precisions.contains_key(&p) { p } else { *entry.precisions.keys().max().expect("stored level") }
            };
            let t = if proj == 2 {
                pack.tensor_rows(entry, precision, &ranges, device)
            } else {
                pack.tensor_cols(entry, precision, &ranges, device)
            }
            .map_err(parse)?;
            assign_param(&mut model, &path, ParamSrc::Ready2(t), QuantPolicy::Off, device).map_err(parse)?;
            assigned += 1;
            continue;
        }
        let precision = {
            let p = choose(entry);
            if entry.precisions.contains_key(&p) {
                p
            } else {
                // Fall back to the best float level the pack stored.
                *entry
                    .precisions
                    .keys()
                    .max()
                    .ok_or_else(|| parse(format!("'{}' has no stored precision", entry.name)))?
            }
        };
        let src = match entry.role {
            // The embedding may live somewhere else entirely — see
            // `load_from_pack_partitioned_split`.
            Role::Embedding => ParamSrc::Ready2(
                pack.tensor::<2>(entry, precision, embed_device.unwrap_or(device))
                    .map_err(parse)?,
            ),
            Role::Linear | Role::Expert { .. } => ParamSrc::Ready2(
                pack.tensor::<2>(entry, precision, device).map_err(parse)?,
            ),
            Role::Vector | Role::Conv => ParamSrc::F32 {
                values: pack.read_f32(entry).map_err(parse)?,
                shape: entry.shape.clone(),
            },
        };
        assign_param(&mut model, &path, src, QuantPolicy::Off, device).map_err(parse)?;
        assigned += 1;
    }
    let expected = expected_tensor_count(&config, untied);
    if assigned != expected {
        return Err(parse(format!(
            "pack supplied {assigned} trunk tensors, the architecture needs {expected}"
        )));
    }
    Ok(LoadedQwen35 {
        model,
        config,
        tokenizer_config: None,
        ffn_pool: None,
        ffn_skip_tau: 0.0,
        ffn_plan: None,
    })
}

impl CausalLm for LoadedQwen35 {
    type Cache = Vec<Qwen35Kv>;

    fn is_eos(&self, id: u32) -> bool {
        self.config.eos_token_id.contains(id)
    }

    fn new_cache(&self) -> Self::Cache {
        (0..self.config.num_layers)
            .map(|i| {
                if self.config.is_attention(i) {
                    Qwen35Kv::Attn(None)
                } else {
                    Qwen35Kv::Delta(DeltaState {
                        conv: None,
                        state: None,
                    })
                }
            })
            .collect()
    }

    fn forward(
        &self,
        new_ids: &[u32],
        past: usize,
        cache: &mut Self::Cache,
        device: &Device,
    ) -> Tensor<2> {
        let t = new_ids.len();
        assert!(t >= 1, "qwen35 forward: need at least one token");
        assert!(
            cache.len() == self.config.num_layers,
            "qwen35 forward: cache has {} layers, model has {}",
            cache.len(),
            self.config.num_layers
        );
        let cfg = &self.config;

        // The embedding may live on a different device from the rest of the
        // model (it is a gather, so it is often left on the host to keep VRAM
        // for weights that compute — see `load_from_pack_partitioned_split`).
        // A gather needs its indices on the SAME device as the table, so the
        // indices are built there and only the small `[1, t, hidden]` result
        // crosses over.
        let embed_device = self.model.embed_tokens.weight.val().device();
        let ids32: Vec<i32> = new_ids.iter().map(|&i| i as i32).collect();
        let input = Tensor::<1, Int>::from_data(
            TensorData::new(ids32, [t]),
            (&embed_device, crate::backend::int_dtype()),
        )
        .reshape([1, t]);
        let mut x = self.model.embed_tokens.forward(input).to_device(device);

        let (cos, sin) = rope_tables(t, past, cfg.rope_dim, cfg.rope_theta, device);
        let mask = (t > 1).then(|| causal_mask(t, past, device));

        for (li, (layer, kv)) in self.model.layers.iter().zip(cache.iter_mut()).enumerate() {
            let h = layer.input_norm.forward(x.clone());
            let h = match (&layer.self_attn, &layer.linear_attn, kv) {
                (Some(attn), None, Qwen35Kv::Attn(kv_state)) => {
                    attn.forward(h, cfg, &cos, &sin, mask.as_ref(), kv_state)
                }
                (None, Some(delta), Qwen35Kv::Delta(state)) => delta.forward(h, cfg, state),
                _ => unreachable!("qwen35 forward: layer/cache kind mismatch"),
            };
            x = x.add(h);
            let h2 = layer.post_attn_norm.forward(x.clone());
            // SwiGLU spelled through qlinear (the mlp's own forward would
            // reshape a packed quantized weight — see qlinear).
            let gate = activation::silu(qlinear(&layer.mlp.gate_proj, h2.clone()));
            let up = qlinear(&layer.mlp.up_proj, h2.clone());
            let mut ffn = qlinear(&layer.mlp.down_proj, gate.clone().mul(up));
            if let Some(pool) = &self.ffn_pool {
                // Working set: issue THIS layer's staging decisions before
                // its FFN runs, so the transfers for the next layer overlap
                // this layer's compute instead of stalling in front of it.
                // Nothing here blocks — a cluster that has not landed by the
                // time its layer runs simply computes on the host.
                if let Some(plan) = &self.ffn_plan
                    && let Some(sched) = plan.layers.get(li)
                {
                    pool.apply_schedule(li, sched, device);
                }
                // Remote clusters of a partitioned FFN (exact sum; skip only
                // when a measured tau was chosen).
                let [b, tt, hd] = ffn.dims();
                let local_energy: Vec<f32> = if self.ffn_skip_tau > 0.0 {
                    gate.powf_scalar(2.0)
                        .sum_dim(2)
                        .into_data()
                        .convert::<f32>()
                        .to_vec::<f32>()
                        .expect("local gate energy")
                } else {
                    Vec::new()
                };
                let skip = (self.ffn_skip_tau > 0.0).then_some((self.ffn_skip_tau, local_energy.as_slice()));
                if let Some(remote) = pool.run_dense(li, h2.reshape([b * tt, hd]), skip) {
                    ffn = ffn.add(remote.reshape([b, tt, hd]));
                }
            }
            x = x.add(ffn);
        }
        let x = self.model.norm.forward(x);

        // Last position only → logits [1, vocab].
        let last = x.narrow(1, t - 1, 1).reshape([1, cfg.hidden_size]);
        match &self.model.lm_head {
            Some(head) => qlinear2(head, last),
            None => {
                // Tied head: logits = h · Eᵀ. The embedding may be on another
                // device (host-resident gather table), and unlike the gather
                // this IS a matmul — so run it where the big tensor lives and
                // move only the `[1, vocab]` result, rather than dragging a
                // multi-GB table across the bus every token.
                let e = self.model.embed_tokens.weight.val(); // [vocab, hidden]
                let out_device = last.device();
                let e_device = e.device();
                last.to_device(&e_device)
                    .matmul(e.swap_dims(0, 1))
                    .to_device(&out_device)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toy config exercising both layer kinds: layer 1 is full attention
    /// (`(1+1) % 2 == 0`), layers 0 and 2 are DeltaNet.
    fn toy_config() -> Qwen35Config {
        Qwen35Config {
            vocab_size: 64,
            hidden_size: 16,
            num_layers: 3,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 8,
            intermediate_size: 24,
            rms_norm_eps: 1e-6,
            rope_theta: 1e4,
            rope_dim: 4,
            full_attention_interval: 2,
            conv_kernel: 3,
            d_inner: 12, // 3 v-heads × d_state 4
            d_state: 4,
            n_k_heads: 1,
            n_v_heads: 3,
            eos_token_id: EosIds::One(0),
        }
    }

    fn toy_model() -> LoadedQwen35 {
        let cfg = toy_config();
        cfg.validate().expect("toy config validates");
        let device = crate::backend::cpu_device();
        LoadedQwen35 {
            model: build(&cfg, &device, false),
            config: cfg,
            tokenizer_config: None,
            ffn_pool: None,
            ffn_skip_tau: 0.0,
            ffn_plan: None,
        }
    }

    /// The load-bearing invariant for BOTH caches (attention KV and the
    /// DeltaNet conv window + recurrent state): prefill + one-token decode
    /// steps must produce exactly the logits of a single full prefill.
    #[test]
    fn cached_decode_matches_full_prefill() {
        let m = toy_model();
        let device = crate::backend::cpu_device();
        let ids: Vec<u32> = vec![3, 17, 42, 9, 60, 11];

        let mut full_cache = m.new_cache();
        let full = m
            .forward(&ids, 0, &mut full_cache, &device)
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        let mut cache = m.new_cache();
        let _ = m.forward(&ids[..3], 0, &mut cache, &device);
        let mut last = Vec::new();
        for (i, &id) in ids.iter().enumerate().skip(3) {
            last = m
                .forward(&[id], i, &mut cache, &device)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
        }
        assert_eq!(full.len(), last.len());
        for (i, (f, s)) in full.iter().zip(&last).enumerate() {
            assert!(
                (f - s).abs() < 1e-4,
                "logit {i}: full {f} vs stepped {s}"
            );
        }
    }

    /// DeltaNet state actually carries information: the same final token
    /// after different prefixes must produce different logits.
    #[test]
    fn recurrent_state_carries_the_prefix() {
        let m = toy_model();
        let device = crate::backend::cpu_device();
        let mut c1 = m.new_cache();
        let mut c2 = m.new_cache();
        let _ = m.forward(&[1, 2, 3], 0, &mut c1, &device);
        let _ = m.forward(&[9, 8, 7], 0, &mut c2, &device);
        let a = m
            .forward(&[5], 3, &mut c1, &device)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let b = m
            .forward(&[5], 3, &mut c2, &device)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let max_diff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff > 1e-6, "different prefixes must change the logits");
    }

    #[test]
    fn config_layer_kinds_follow_the_interval() {
        let cfg = toy_config();
        assert!(!cfg.is_attention(0));
        assert!(cfg.is_attention(1));
        assert!(!cfg.is_attention(2));
        // The 27B pattern: every 4th layer.
        let mut real = cfg;
        real.full_attention_interval = 4;
        let attn: Vec<usize> = (0..8).filter(|&i| real.is_attention(i)).collect();
        assert_eq!(attn, vec![3, 7]);
    }
}
