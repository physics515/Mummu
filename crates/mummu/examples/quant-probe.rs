//! Empirical probe of burn 0.21's keep-quantized execution — the P9
//! substrate: which `QuantScheme`s actually run a mixed float×quantized
//! matmul on each backend, and at what error vs the f32 reference.
//!
//! `cargo run --release -p mummu --example quant-probe`

use burn::tensor::quantization::{Calibration, QuantLevel, QuantParam, QuantValue};
use burn::tensor::{Distribution, Tensor, backend::Backend};

fn probe<B: Backend>(label: &str, device: &B::Device) {
    let [m, k, n] = [8usize, 2048, 2048];
    let x = Tensor::<B, 2>::random([m, k], Distribution::Default, device);
    let w = Tensor::<B, 2>::random([k, n], Distribution::Default, device);
    let reference = x.clone().matmul(w.clone());
    let ref_host = reference
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .unwrap();
    let ref_scale = ref_host.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

    let schemes: Vec<(&str, QuantValue, QuantLevel)> = vec![
        ("Q8S/tensor", QuantValue::Q8S, QuantLevel::Tensor),
        ("Q8S/block32", QuantValue::Q8S, QuantLevel::block([32])),
        ("Q4S/tensor", QuantValue::Q4S, QuantLevel::Tensor),
        ("Q4S/block32", QuantValue::Q4S, QuantLevel::block([32])),
    ];
    for (name, value, level) in schemes {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            use burn::tensor::quantization::{compute_q_params, compute_range};
            let scheme =
                burn::tensor::quantization::QuantScheme::default()
                    .with_value(value)
                    .with_level(level)
                    .with_param(QuantParam::F32);
            let range = compute_range(&scheme, &w, &Calibration::MinMax);
            let qparams = compute_q_params(&scheme, range);
            let wq = w.clone().quantize(&scheme, qparams);
            let out = x.clone().matmul(wq);
            out.into_data().convert::<f32>().to_vec::<f32>().unwrap()
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
            Err(e) => {
                let msg = e
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string()))
                    .unwrap_or_else(|| "<non-string panic>".into());
                let msg = msg.lines().next().unwrap_or("");
                println!("[{label}] {name}: FAILED — {msg}");
            }
        }
    }
}

fn main() {
    let cpu = burn::tensor::Device::<mummu::backend::Cpu>::default();
    probe::<mummu::backend::Cpu>("cpu/flex", &cpu);
    if mummu::backend::use_gpu() {
        let gpu = burn::tensor::Device::<mummu::backend::Gpu>::default();
        probe::<mummu::backend::Gpu>("gpu/wgpu", &gpu);
    }
    #[cfg(feature = "cuda")]
    {
        let cuda = burn::tensor::Device::<mummu::backend::Cuda>::default();
        probe::<mummu::backend::Cuda>("gpu/cuda", &cuda);
    }
}
