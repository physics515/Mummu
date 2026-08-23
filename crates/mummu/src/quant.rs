//! P9 — the keep-quantized runtime policy (stage 1).
//!
//! One model path: the same module structs and the same forward code serve
//! float and quantized weights — a `Param` holding a quantized tensor
//! executes through the backend's `q_matmul` (burn-cubecl runs the mixed
//! float×quantized matmul natively; burn-flex falls back to per-op
//! dequantize, slower but with the same memory-resident win). Import
//! **re-quantizes**: whatever the source stored (BF16, Q4_K, IQ4_XS, …) is
//! dequantized per tensor and re-quantized into this one scheme.
//!
//! What stays float, deliberately:
//! - **Embeddings** — token gather (`q_select`) has no kernel on the GPU
//!   backends, and a wrong silent fallback is worse than the ~2 bytes/param
//!   this table costs.
//! - Norm gammas, biases, conv kernels, per-head vectors — tiny, and their
//!   precision anchors the numerics.
//!
//! Scheme facts measured on this machine (2026-08-21, `quant-probe`):
//! Q8S (tensor or block-32) is correct on flex CPU, wgpu AND CUDA (~0.04%
//! matmul error). Q4S is correct on flex CPU and CUDA (~0.8%) but **wgpu's
//! Q4 kernel returns garbage** (~98% error) — [`QuantPolicy::Q4`] must be
//! refused on wgpu until that kernel is fixed upstream.

use burn::tensor::quantization::{
    Calibration, QuantLevel, QuantParam, QuantScheme, QuantValue, compute_q_params, compute_range,
};
use burn::tensor::Tensor;

/// Which quantization the keep-quantized path applies on import.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantPolicy {
    /// No quantization — the classic f32 path.
    Off,
    /// 8-bit symmetric, block-32 scales: the proven default (≈4x smaller
    /// than f32, ~0.04% matmul error on every backend).
    Q8,
    /// 4-bit symmetric, block-32 scales (≈8x smaller, ~0.8% matmul error).
    /// Verified correct on flex CPU and CUDA; **wgpu's Q4 kernel returns
    /// garbage** (quant-probe 2026-08-21: ~98% error) — consumers must
    /// refuse Q4-on-wgpu loudly rather than run it.
    Q4,
}

impl QuantPolicy {
    /// Parse the `MUMMU_QUANT` convention: `q8` / `int8` → [`Self::Q8`],
    /// `off`/empty/unset → [`Self::Off`]. Unknown values are a loud error.
    pub fn from_env() -> Result<Self, String> {
        match std::env::var("MUMMU_QUANT") {
            Err(_) => Ok(Self::Off),
            Ok(v) if v.is_empty() || v.eq_ignore_ascii_case("off") => Ok(Self::Off),
            Ok(v) if v.eq_ignore_ascii_case("q8") || v.eq_ignore_ascii_case("int8") => {
                Ok(Self::Q8)
            }
            Ok(v) if v.eq_ignore_ascii_case("q4") || v.eq_ignore_ascii_case("int4") => {
                Ok(Self::Q4)
            }
            Ok(other) => Err(format!(
                "unknown MUMMU_QUANT {other:?} (expected q8, q4, or off)"
            )),
        }
    }

    /// The burn scheme this policy denotes; `None` for [`Self::Off`].
    #[must_use]
    pub fn scheme(self) -> Option<QuantScheme> {
        match self {
            Self::Off => None,
            Self::Q8 => Some(
                QuantScheme::default()
                    .with_value(QuantValue::Q8S)
                    .with_level(QuantLevel::block([32]))
                    .with_param(QuantParam::F32),
            ),
            Self::Q4 => Some(
                QuantScheme::default()
                    .with_value(QuantValue::Q4S)
                    .with_level(QuantLevel::block([32]))
                    .with_param(QuantParam::F32),
            ),
        }
    }

    /// Is a 2-D weight of `dims` worth quantizing? Small projections lose
    /// more accuracy than the bytes they return; 256×256 (65k elements) is
    /// the floor — everything matmul-heavy in a real LLM clears it. The
    /// last dim must also divide by the block width: burn's block scales
    /// run along rows and a non-divisible row (qwen35-27B's 48-wide β/α
    /// projections) crashes the block-range reshape.
    #[must_use]
    pub fn eligible(self, dims: &[usize]) -> bool {
        const BLOCK: usize = 32; // keep in sync with `scheme()`
        self != Self::Off
            && dims.len() == 2
            && dims.iter().product::<usize>() >= (1 << 16)
            && dims[1].is_multiple_of(BLOCK)
    }
}

/// Quantize one weight tensor per `policy` (min-max calibration — weights
/// are static, so calibration is exact). The caller has already decided
/// eligibility; `Off` is a caller bug.
pub fn quantize_weight<const D: usize>(
    policy: QuantPolicy,
    tensor: Tensor<D>,
) -> Tensor<D> {
    let scheme = policy
        .scheme()
        .expect("quantize_weight called with QuantPolicy::Off");
    let range = compute_range(&scheme, &tensor, &Calibration::MinMax);
    let qparams = compute_q_params(&scheme, range);
    tensor.quantize(&scheme, qparams)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    #[test]
    fn policy_env_parsing() {
        // from_env reads the process env — test the string logic via the
        // parse arms directly instead of mutating global state.
        assert_eq!(QuantPolicy::Off.scheme(), None);
        assert!(QuantPolicy::Q8.scheme().is_some());
        assert!(QuantPolicy::Q8.eligible(&[512, 512]));
        assert!(!QuantPolicy::Q8.eligible(&[16, 16]));
        assert!(!QuantPolicy::Q8.eligible(&[1024]));
        assert!(!QuantPolicy::Off.eligible(&[4096, 4096]));
        // Non-block-divisible rows (the 27B's [5120, 48] β/α) stay float.
        assert!(!QuantPolicy::Q4.eligible(&[5120, 48]));
        assert!(QuantPolicy::Q4.eligible(&[5120, 10240]));
    }

    #[test]
    fn q8_roundtrip_error_is_small() {
        let device = crate::backend::cpu_device();
        let w = Tensor::<2>::random([64, 64], Distribution::Default, &device);
        let host = w.clone().into_data().to_vec::<f32>().unwrap();
        let max_abs = host.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let q = quantize_weight(QuantPolicy::Q8, w);
        let back = q.dequantize().into_data().to_vec::<f32>().unwrap();
        let max_err = host
            .iter()
            .zip(&back)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // Block-32 int8: worst-case error is scale/2 = max|block|/254.
        assert!(
            max_err <= max_abs / 100.0,
            "Q8 round-trip error too large: {max_err} vs max |w| {max_abs}"
        );
    }
}
