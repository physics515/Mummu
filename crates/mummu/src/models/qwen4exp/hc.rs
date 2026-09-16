//! Hyper-connections: the grouped-norm low-rank stream mixer and the
//! `2·sigmoid(inject)` combine.
//!
//! qwen4exp has no `attn_norm`/`ffn_norm`/`output_norm`. Instead the residual
//! is `H` parallel streams (`[b, t, H·E]`, **stream-major**: stream `s` is
//! elements `s·E .. (s+1)·E`), and every block reads a gated mean of them and
//! writes its output back into each stream with a per-stream weight. Ported
//! from llama.cpp `src/models/qwen4exp.cpp` (`build_hc_mix` /
//! `build_hc_combine`, the unfused branch) and cross-checked against
//! transformers `Qwen4ExpTextGatedResidual` + `Qwen4ExpTextDecoderLayer`:
//!
//! ```text
//! mix(res):      xn     = grouped_rms(res) · norm           # rms per stream over E, eps inside the sqrt
//!                lo     = silu(W_down · xn · (1/H))          # scale BEFORE silu
//!                gated  = xn · sigmoid(W_up · lo)
//!                mixed  = (1/H) · Σ_s gated[s]               # [b, t, E]
//!                inject = W_inject · xn                      # [b, t, H], absent on the output mixer
//! combine(...):  res[s] += block_out · 2·sigmoid(inject[s] / H)
//! ```
//!
//! Both sources agree on every step above (see the port report); the norm
//! gammas arrive already folded to `1 + w` by the GGUF converter, so they are
//! multiplied as stored. No biases anywhere.

use burn::module::{Module, Param};
use burn::nn::Linear;
use burn::tensor::{DType, Tensor, activation};

/// `Linear::forward` without touching the weight's shape — the stand-in for
/// `qwen35::qlinear` until that is `pub(crate)`. burn's `Linear` unsqueezes
/// the weight to the input rank, and a reshape on **packed** quantized
/// storage is broken (the physical element count differs from the logical
/// one), so the FLOAT input is flattened to `[b·t, in]` instead and the
/// weight goes into the matmul exactly as stored (`[in, out]`). Decode-shape
/// quantized weights take the packed GEMV, as in qwen35.
pub(super) fn linear3(l: &Linear, x: Tensor<3>) -> Tensor<3> {
    debug_assert!(l.bias.is_none(), "qwen4exp projections are bias-free");
    let [b, t, d_in] = x.dims();
    let w = l.weight.val(); // [in, out]
    let [w_in, d_out] = w.dims();
    assert_eq!(
        w_in, d_in,
        "projection expects {w_in} input features, got {d_in}"
    );
    let x2 = crate::nn::refarith::linear_input2(x.reshape([b * t, d_in]), d_in, d_out);
    let y = match crate::nn::try_q4s_gemv(&x2, &w) {
        Some(y) => y,
        None => x2.matmul(w),
    };
    y.reshape([b, t, d_out])
}

/// A bias-free `Linear` around a ready `[in, out]` weight (possibly packed).
/// Building the struct directly skips `LinearConfig::init`'s random fill,
/// which the loader would immediately overwrite.
pub(super) fn linear_from(weight: Tensor<2>) -> Linear {
    Linear {
        weight: Param::from_tensor(weight),
        bias: None,
    }
}

/// Grouped RMSNorm: `x [b, t, G·E]` normalized over each of the `G`
/// stream-major groups of `E`, then scaled by the full-width `gamma [G·E]`.
///
/// `v / sqrt(mean(v²) + eps)` — ggml's `rms_norm` and transformers'
/// `x · rsqrt(mean(x²) + eps)` both keep eps INSIDE the root. The mean is taken
/// in f32 (as burn's `RmsNorm` does) so an f16 activation cannot overflow the
/// square.
pub(super) fn grouped_rms(x: Tensor<3>, gamma: Tensor<1>, groups: usize, eps: f64) -> Tensor<3> {
    let [b, t, width] = x.dims();
    assert!(
        groups > 0 && width.is_multiple_of(groups),
        "width {width} is not {groups} whole groups"
    );
    let hidden = width / groups;
    let dtype = x.dtype();
    let g = x.reshape([b, t, groups, hidden]);
    let rms = g
        .clone()
        .cast(DType::F32)
        .powi_scalar(2)
        .mean_dim(3)
        .add_scalar(eps)
        .sqrt()
        .cast(dtype); // [b, t, G, 1]
    g.div(rms)
        .reshape([b, t, width])
        .mul(gamma.reshape([1, 1, width]))
}

/// `res[s] += block_out · 2·sigmoid(inject[s] / H)` for every stream `s`.
///
/// `2·sigmoid` centres the scatter weights on 1, so a zero injection is a
/// plain residual add into every stream. Weight-free, hence also a free
/// function; [`HyperConnection::combine`] forwards here.
#[must_use]
pub fn hc_combine(
    res: Tensor<3>,
    block_out: Tensor<3>,
    inject: Tensor<3>,
    streams: usize,
) -> Tensor<3> {
    let [b, t, width] = res.dims();
    let [bo_b, bo_t, hidden] = block_out.dims();
    assert_eq!(
        (bo_b, bo_t),
        (b, t),
        "block output batch/time must match the residual"
    );
    assert_eq!(
        width,
        streams * hidden,
        "residual width {width} is not {streams} streams of {hidden}"
    );
    assert_eq!(
        inject.dims(),
        [b, t, streams],
        "inject is one weight per stream"
    );
    let w = activation::sigmoid(inject.div_scalar(streams as f32))
        .mul_scalar(2.0)
        .reshape([b, t, streams, 1]);
    let add = block_out.reshape([b, t, 1, hidden]).mul(w); // [b, t, H, E]
    res.reshape([b, t, streams, hidden])
        .add(add)
        .reshape([b, t, width])
}

/// One hyper-connection mixer: `hc_attn_*`, `hc_ffn_*`, or (without an
/// inject projection) the final `output_hc_*` that stands in for the output
/// norm.
#[derive(Module, Debug)]
pub struct HyperConnection {
    /// Grouped-norm gamma `[H·E]`, stream-major, already `1 + w`.
    pub norm: Param<Tensor<1>>,
    /// `H·E → R` (GGUF `hc_*_down`, ne `[H·E, R]`).
    pub down: Linear,
    /// `R → H·E` (GGUF `hc_*_up`, ne `[R, H·E]`).
    pub up: Linear,
    /// `H·E → H` scatter logits (GGUF `hc_*_inject`); `None` on the output
    /// mixer, which only mixes.
    pub inject: Option<Linear>,
    /// `E`: width of one stream and of the mixed block input.
    pub hidden: usize,
    /// `H`: number of residual streams.
    pub streams: usize,
    /// RMSNorm epsilon (`rms_norm_eps`).
    pub eps: f64,
}

impl HyperConnection {
    /// Assemble from ready weights. Linear weights are burn `[in, out]`
    /// (the GGUF row-major `[out, in]` transposed, as `linear_weight` does
    /// for qwen35) and may be packed.
    ///
    /// # Panics
    ///
    /// When a shape disagrees with `hidden`/`streams` — a wrong shape here
    /// means a mis-mapped tensor, which would otherwise mix garbage.
    #[must_use]
    pub fn from_weights(
        norm: Tensor<1>,
        down: Tensor<2>,
        up: Tensor<2>,
        inject: Option<Tensor<2>>,
        hidden: usize,
        streams: usize,
        eps: f64,
    ) -> Self {
        let width = hidden * streams;
        assert_eq!(norm.dims(), [width], "hc norm is [H*E]");
        let [d_in, rank] = down.dims();
        assert_eq!(d_in, width, "hc down reads the whole residual");
        assert_eq!(up.dims(), [rank, width], "hc up maps R back to H*E");
        if let Some(w) = &inject {
            assert_eq!(w.dims(), [width, streams], "hc inject is H*E -> H");
        }
        Self {
            norm: Param::from_tensor(norm),
            down: linear_from(down),
            up: linear_from(up),
            inject: inject.map(linear_from),
            hidden,
            streams,
            eps,
        }
    }

    /// The low rank `R`.
    #[must_use]
    pub fn low_rank(&self) -> usize {
        self.down.weight.val().dims()[1]
    }

    /// Mix the residual streams into one block input.
    ///
    /// `res [b, t, H·E]` → (`mixed [b, t, E]`, `inject [b, t, H]` when this
    /// mixer has an inject projection). The inject logits are returned raw;
    /// [`Self::combine`] applies the `2·sigmoid(·/H)`.
    #[must_use]
    pub fn mix(&self, res: Tensor<3>) -> (Tensor<3>, Option<Tensor<3>>) {
        let [b, t, width] = res.dims();
        let (h, e) = (self.streams, self.hidden);
        assert_eq!(
            width,
            h * e,
            "residual width {width} is not {h} streams of {e}"
        );
        let xn = grouped_rms(res, self.norm.val(), h, self.eps); // [b, t, HE]
        let inv_h = 1.0 / h as f32;
        // The 1/H scale sits BEFORE the silu in both references.
        let lo = activation::silu(linear3(&self.down, xn.clone()).mul_scalar(inv_h));
        let gate = linear3(&self.up, lo); // [b, t, HE]
        let inject = self.inject.as_ref().map(|l| linear3(l, xn.clone()));
        let mixed = xn
            .mul(activation::sigmoid(gate))
            .reshape([b, t, h, e])
            .sum_dim(2) // [b, t, 1, E]
            .reshape([b, t, e])
            .mul_scalar(inv_h);
        (mixed, inject)
    }

    /// Write a block's output back into every stream:
    /// `res [b, t, H·E] + block_out [b, t, E] · 2·sigmoid(inject [b, t, H] / H)`.
    #[must_use]
    pub fn combine(&self, res: Tensor<3>, block_out: Tensor<3>, inject: Tensor<3>) -> Tensor<3> {
        hc_combine(res, block_out, inject, self.streams)
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Device, TensorData};

    use super::*;

    /// Deterministic xorshift values in `[-scale, scale)`: host-side, so the
    /// scalar reference and the tensors see the exact same numbers.
    fn rand_vec(seed: &mut u64, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed ^= *seed << 13;
                *seed ^= *seed >> 7;
                *seed ^= *seed << 17;
                ((*seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    }

    fn t1(v: &[f32], device: &Device) -> Tensor<1> {
        Tensor::from_data(TensorData::new(v.to_vec(), [v.len()]), device)
    }

    fn t3(v: &[f32], shape: [usize; 3], device: &Device) -> Tensor<3> {
        Tensor::from_data(TensorData::new(v.to_vec(), shape), device)
    }

    /// GGUF row-major `[out, in]` values → burn `[in, out]` weight.
    fn weight(v: &[f32], out: usize, inp: usize, device: &Device) -> Tensor<2> {
        Tensor::<2>::from_data(TensorData::new(v.to_vec(), [out, inp]), device).swap_dims(0, 1)
    }

    fn host(t: Tensor<3>) -> Vec<f32> {
        t.into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .expect("f32")
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// `W · x` with `W` row-major `[out, in]` — the GGUF meaning of a linear.
    fn matvec(w: &[f32], x: &[f32], out: usize) -> Vec<f32> {
        let inp = x.len();
        (0..out)
            .map(|o| (0..inp).map(|i| w[o * inp + i] * x[i]).sum())
            .collect()
    }

    /// Scalar reference for one token, transcribed from the spec (section 2).
    #[allow(clippy::too_many_arguments)]
    fn ref_mix(
        res: &[f32],
        norm: &[f32],
        down: &[f32],
        up: &[f32],
        inject: Option<&[f32]>,
        e: usize,
        h: usize,
        r: usize,
        eps: f32,
    ) -> (Vec<f32>, Option<Vec<f32>>) {
        let mut xn = vec![0.0; h * e];
        for s in 0..h {
            let v = &res[s * e..(s + 1) * e];
            let ms = v.iter().map(|x| x * x).sum::<f32>() / e as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            for i in 0..e {
                xn[s * e + i] = v[i] * inv * norm[s * e + i];
            }
        }
        let lo: Vec<f32> = matvec(down, &xn, r)
            .into_iter()
            .map(|x| {
                let x = x / h as f32;
                x * sigmoid(x)
            })
            .collect();
        let gate = matvec(up, &lo, h * e);
        let mut mixed = vec![0.0; e];
        for s in 0..h {
            for i in 0..e {
                mixed[i] += xn[s * e + i] * sigmoid(gate[s * e + i]);
            }
        }
        for m in &mut mixed {
            *m /= h as f32;
        }
        (mixed, inject.map(|w| matvec(w, &xn, h)))
    }

    fn ref_combine(res: &[f32], block: &[f32], inject: &[f32], e: usize, h: usize) -> Vec<f32> {
        let mut out = res.to_vec();
        for s in 0..h {
            let w = 2.0 * sigmoid(inject[s] / h as f32);
            for i in 0..e {
                out[s * e + i] += block[i] * w;
            }
        }
        out
    }

    fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= tol * (1.0 + w.abs()),
                "{what}[{i}]: got {g}, want {w}"
            );
        }
    }

    /// The tensor mixer and combine equal the per-token scalar transcription
    /// of the spec on random weights — stream-major layout, eps inside the
    /// root, the 1/H before the silu, and the GGUF `[out, in]` → burn
    /// `[in, out]` transpose all have to line up for this to hold.
    #[test]
    fn mix_and_combine_match_the_scalar_reference() {
        let device = crate::backend::cpu_device();
        let (b, t, e, h, r) = (2, 3, 6, 3, 4);
        let eps = 1e-6_f32;
        let mut seed = 0x9E37_79B9_7F4A_7C15;
        let norm: Vec<f32> = rand_vec(&mut seed, h * e, 0.5)
            .into_iter()
            .map(|x| 1.0 + x)
            .collect();
        let down = rand_vec(&mut seed, r * h * e, 0.8);
        let up = rand_vec(&mut seed, h * e * r, 0.8);
        let inj = rand_vec(&mut seed, h * h * e, 0.8);
        let res = rand_vec(&mut seed, b * t * h * e, 2.0);
        let block = rand_vec(&mut seed, b * t * e, 1.5);

        let hc = HyperConnection::from_weights(
            t1(&norm, &device),
            weight(&down, r, h * e, &device),
            weight(&up, h * e, r, &device),
            Some(weight(&inj, h, h * e, &device)),
            e,
            h,
            f64::from(eps),
        );
        assert_eq!(hc.low_rank(), r);
        let res_t = t3(&res, [b, t, h * e], &device);
        let (mixed, inject) = hc.mix(res_t.clone());
        let inject = inject.expect("attn/ffn mixers carry an inject projection");
        let combined = hc.combine(res_t, t3(&block, [b, t, e], &device), inject.clone());
        let (mixed, inject, combined) = (host(mixed), host(inject), host(combined));

        for tok in 0..b * t {
            let r_tok = &res[tok * h * e..(tok + 1) * h * e];
            let (m_ref, i_ref) = ref_mix(r_tok, &norm, &down, &up, Some(&inj), e, h, r, eps);
            let i_ref = i_ref.expect("inject requested");
            assert_close(&mixed[tok * e..(tok + 1) * e], &m_ref, 1e-5, "mixed");
            assert_close(&inject[tok * h..(tok + 1) * h], &i_ref, 1e-5, "inject");
            let c_ref = ref_combine(r_tok, &block[tok * e..(tok + 1) * e], &i_ref, e, h);
            assert_close(
                &combined[tok * h * e..(tok + 1) * h * e],
                &c_ref,
                1e-5,
                "combined",
            );
        }
    }

    /// The output mixer has no inject projection and must say so rather than
    /// return a zero tensor a caller could mistake for real logits.
    #[test]
    fn the_output_mixer_returns_no_inject() {
        let device = crate::backend::cpu_device();
        let (e, h, r) = (4, 2, 3);
        let mut seed = 7;
        let hc = HyperConnection::from_weights(
            t1(&vec![1.0; h * e], &device),
            weight(&rand_vec(&mut seed, r * h * e, 1.0), r, h * e, &device),
            weight(&rand_vec(&mut seed, h * e * r, 1.0), h * e, r, &device),
            None,
            e,
            h,
            1e-6,
        );
        let res = t3(&rand_vec(&mut seed, h * e, 1.0), [1, 1, h * e], &device);
        let (mixed, inject) = hc.mix(res);
        assert_eq!(mixed.dims(), [1, 1, e]);
        assert!(inject.is_none(), "output mixer has no inject");
    }

    /// `2·sigmoid(0) = 1`: a zero injection is a plain residual add of the
    /// block output into EVERY stream — the property the converter's
    /// centring relies on.
    #[test]
    fn zero_inject_adds_the_block_output_to_every_stream() {
        let device = crate::backend::cpu_device();
        let (b, t, e, h) = (1, 2, 5, 4);
        let mut seed = 99;
        let res = rand_vec(&mut seed, b * t * h * e, 3.0);
        let block = rand_vec(&mut seed, b * t * e, 3.0);
        let out = host(hc_combine(
            t3(&res, [b, t, h * e], &device),
            t3(&block, [b, t, e], &device),
            Tensor::zeros([b, t, h], &device),
            h,
        ));
        for tok in 0..t {
            for s in 0..h {
                for i in 0..e {
                    let idx = (tok * h + s) * e + i;
                    let want = res[idx] + block[tok * e + i];
                    assert!(
                        (out[idx] - want).abs() <= 1e-6 * (1.0 + want.abs()),
                        "token {tok} stream {s} elem {i}: {} != {want}",
                        out[idx]
                    );
                }
            }
        }
    }

    /// Packed (Q8 block-32) down/up weights go through `linear3` without a
    /// weight reshape and land within quantization error of the float mix.
    /// A reshape of packed storage would panic or scramble here. The inject
    /// projection stays float, as in the shipped file (`hc_*_inject` is F32,
    /// and its 4-wide rows are not whole 32-blocks anyway).
    #[test]
    fn mix_runs_on_packed_weights() {
        use crate::quant::{QuantPolicy, quantize_weight};
        let device = crate::backend::cpu_device();
        let (e, h, r) = (16, 2, 32); // widths divisible by the 32-block
        let mut seed = 1234;
        let norm = vec![1.0; h * e];
        let down = rand_vec(&mut seed, r * h * e, 0.3);
        let up = rand_vec(&mut seed, h * e * r, 0.3);
        let inj = rand_vec(&mut seed, h * h * e, 0.3);
        let build = |q: bool| {
            let w = |v: &[f32], o, i| {
                let t = weight(v, o, i, &device);
                if q {
                    quantize_weight(QuantPolicy::Q8, t)
                } else {
                    t
                }
            };
            HyperConnection::from_weights(
                t1(&norm, &device),
                w(&down, r, h * e),
                w(&up, h * e, r),
                Some(weight(&inj, h, h * e, &device)),
                e,
                h,
                1e-6,
            )
        };
        let res = rand_vec(&mut seed, 3 * h * e, 1.0);
        let run = |hc: &HyperConnection| {
            let (m, i) = hc.mix(t3(&res, [1, 3, h * e], &device));
            (host(m), host(i.expect("inject")))
        };
        let (mf, i_f) = run(&build(false));
        let (mq, iq) = run(&build(true));
        assert_close(&mq, &mf, 2e-2, "packed mixed");
        assert_close(&iq, &i_f, 2e-2, "packed inject");
    }
}
