//! Empirical probe of burn 0.21's keep-quantized execution — the P9
//! substrate: which `QuantScheme`s actually run a mixed float×quantized
//! matmul on each backend, and at what error vs the f32 reference.
//!
//! `cargo run --release -p mummu --example quant-probe`

#![warn(clippy::pedantic, clippy::nursery, clippy::all)]

use burn::tensor::quantization::{Calibration, QuantScheme, QuantValue, ScaleDtype};
use burn::tensor::{Distribution, Tensor};

/// A ladder rung: burn 0.22.0-pre.3 replaced `QuantLevel` with the scheme's
/// own `per_tensor` / `per_block` setters, so a rung is now the setter to
/// apply.
type Rung = fn(QuantScheme) -> QuantScheme;

fn probe(label: &str, device: &burn::tensor::Device) {
    let [m, k, n] = [8usize, 2048, 2048];
    let act = Tensor::<2>::random([m, k], Distribution::Default, device);
    let weight = Tensor::<2>::random([k, n], Distribution::Default, device);
    let reference = act.clone().matmul(weight.clone());
    let ref_host = reference
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .unwrap();
    let ref_scale = ref_host.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

    let schemes: Vec<(&str, QuantValue, Rung)> = vec![
        ("Q8S/tensor", QuantValue::Q8S, |base| {
            base.per_tensor(ScaleDtype::F32)
        }),
        ("Q8S/block32", QuantValue::Q8S, |base| {
            base.per_block([32], ScaleDtype::F32)
        }),
        ("Q4S/tensor", QuantValue::Q4S, |base| {
            base.per_tensor(ScaleDtype::F32)
        }),
        ("Q4S/block32", QuantValue::Q4S, |base| {
            base.per_block([32], ScaleDtype::F32)
        }),
    ];
    for (name, value, level) in schemes {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            use burn::tensor::quantization::{compute_q_params, compute_range};
            let scheme = level(QuantScheme::default().with_value(value));
            let range = compute_range(&scheme, &weight, &Calibration::MinMax);
            let qparams = compute_q_params(&scheme, range);
            let wq = weight.clone().quantize(&scheme, qparams);
            let out = act.clone().matmul(wq);
            out.into_data()
                .convert::<f32>()
                .try_to_vec::<f32>()
                .unwrap()
        }));
        match outcome {
            Ok(got) => {
                let max_abs = got
                    .iter()
                    .zip(&ref_host)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                println!(
                    "[{label}] {name}: OK, max |Δ| = {max_abs:.4} ({:.3}% of max ref)",
                    100.0 * max_abs / ref_scale
                );
            }
            Err(payload) => {
                let msg = payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                    .unwrap_or_else(|| "<non-string panic>".into());
                let msg = msg.lines().next().unwrap_or("");
                println!("[{label}] {name}: FAILED — {msg}");
            }
        }
    }
}

fn main() {
    let cpu = mummu::backend::cpu_device();
    probe("cpu/flex", &cpu);
    if mummu::backend::use_gpu() {
        let gpu = mummu::backend::gpu_device();
        probe("gpu/wgpu", &gpu);
    }
    #[cfg(feature = "cuda")]
    {
        let cuda = mummu::backend::cuda_device();
        probe("gpu/cuda", &cuda);
    }
}
