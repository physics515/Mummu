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
//! The **ladder itself** — which rungs exist, how wide each is, which way
//! demotion runs, and how far the source precision lets it climb — lives in
//! the `mummu-mix` crate, together with the planner that chooses among them.
//! None of that needs a tensor library. What is left here is the one part
//! that does: turning a rung into a burn `QuantScheme`.
//!
//! Scheme facts measured on this machine along the path production takes
//! (packed bytes from a pack onto the device, then multiplied — NOT a
//! synthetic tensor quantized on-device, which measures the probe rather
//! than the runtime): Q8S block-32 is correct on flex CPU, wgpu and CUDA
//! (0.58% matmul error against the pack's own f32); Q4S likewise (9.97%).
//! The long-standing "wgpu's Q4 kernel returns garbage" note is **withdrawn**
//! — 2026-08-23, `examples/pack-precision-probe.rs`.

use burn::tensor::quantization::{
    Calibration, QuantLevel, QuantParam, QuantScheme, QuantValue, compute_q_params, compute_range,
};
use burn::tensor::Tensor;

pub use mummu_mix::QuantPolicy;

/// The burn `QuantScheme` a rung denotes.
///
/// A trait rather than an inherent method because [`QuantPolicy`] belongs to
/// `mummu-mix`, which has no burn dependency and should not gain one — this
/// is the seam between the ladder (pure data) and the tensor library.
pub trait SchemeExt {
    /// `None` for the float rungs ([`QuantPolicy::Off`], [`QuantPolicy::F16`]),
    /// which are not quantizations at all.
    fn scheme(self) -> Option<QuantScheme>;
}

impl SchemeExt for QuantPolicy {
    fn scheme(self) -> Option<QuantScheme> {
        // Block width must stay in step with `QuantPolicy::eligible`, which
        // rejects rows that do not divide it.
        let value = match self {
            QuantPolicy::Off | QuantPolicy::F16 => return None,
            QuantPolicy::Q8 => QuantValue::Q8S,
            QuantPolicy::Q4 => QuantValue::Q4S,
            QuantPolicy::Q2 => QuantValue::Q2S,
        };
        Some(
            QuantScheme::default()
                .with_value(value)
                .with_level(QuantLevel::block([32]))
                .with_param(QuantParam::F32),
        )
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

    /// `LADDER`, `bits`, `demote` and `promote` must describe the SAME order.
    /// They are four separate matches over one enum and drifted once already.
    #[test]
    fn the_ladder_is_consistent_in_every_direction() {
        let ladder = QuantPolicy::LADDER;
        for pair in ladder.windows(2) {
            let (hi, lo) = (pair[0], pair[1]);
            assert!(hi.bits() > lo.bits(), "{hi:?} should be wider than {lo:?}");
            assert_eq!(hi.demote(), Some(lo), "{hi:?} demotes to {lo:?}");
            assert_eq!(lo.promote(), Some(hi), "{lo:?} promotes to {hi:?}");
        }
        assert_eq!(ladder[0].promote(), None, "nothing above the top rung");
        assert_eq!(
            ladder[ladder.len() - 1].demote(),
            None,
            "nothing below the bottom rung"
        );
    }

    /// A quantized checkpoint never earns f32, and nothing ever earns more
    /// than f32 — the cap is the source.
    #[test]
    fn the_ceiling_follows_the_source_precision() {
        // Q4_K_S: 15.36 GB for ~27 G params.
        assert_eq!(QuantPolicy::ceiling_for_source(4.55), QuantPolicy::F16);
        assert_eq!(QuantPolicy::ceiling_for_source(8.0), QuantPolicy::F16);
        assert_eq!(QuantPolicy::ceiling_for_source(16.0), QuantPolicy::F16);
        // A genuinely f32 source is the only thing that earns f32.
        assert_eq!(QuantPolicy::ceiling_for_source(32.0), QuantPolicy::Off);
        // ...and the ladder has no rung above it to climb to.
        assert_eq!(QuantPolicy::Off.promote(), None);
    }

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
