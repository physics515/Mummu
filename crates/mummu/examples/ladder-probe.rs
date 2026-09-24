//! What each rung of the precision ladder actually costs and buys, measured
//! on a real pack tensor at this model's dimensions.
//!
//! Two ways a rung can be reached, and both matter to the rebalancer:
//!   * **from the pack** — read `q8.bin`/`q4.bin` straight onto the device.
//!     This is how a tensor is first placed, and how it is promoted back up.
//!   * **by demotion** — take the tensor already resident and requantize it
//!     lower without touching disk. This is the pressure response, and it is
//!     the only way to reach Q2 (no pack stores it).
//!
//! Reports relative error against the pack's own f32 bytes and the resident
//! byte count, which together are what a placement decision trades off.

#![warn(clippy::pedantic, clippy::nursery, clippy::all)]

use burn::tensor::{Device, Tensor, TensorData};
use mummu::backend;
use mummu::pack::{Pack, Precision, Role};
use mummu::quant::{QuantPolicy, quantize_weight};
use mummu_num::{f32_from_usize, f64_from_usize};
use std::path::PathBuf;

/// Largest 2-D weight considered, in parameters: the f32 reference lives on
/// the host and a multi-GB one turns this probe into a memory test.
const MAX_PARAMS: usize = 128 << 20;

/// Multiply `xs` by `w` on the GPU and return the worst relative error
/// against the f32 reference `want` (scaled by its largest magnitude), or
/// `None` when the matmul panicked or could not be read back.
fn gpu_rel_error(gpu: &Device, xs: &[f32], w: Tensor<2>, want: &[f32], scale: f32) -> Option<f32> {
    let xs = xs.to_vec();
    let gpu = gpu.clone();
    let k = xs.len();
    let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let x = Tensor::<2>::from_data(
            TensorData::new(xs, [1, k]),
            (&gpu, backend::float_dtype(&gpu)),
        );
        x.matmul(w)
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .ok()
    }))
    .ok()
    .flatten()?;
    Some(
        got.iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs() / scale)
            .fold(0.0f32, f32::max),
    )
}

fn main() {
    let dir = std::env::var("PACK_DIR").unwrap_or_else(|_| {
        r"D:\Docker Containers\mummu\models\qwen3.8-27b-ud-q4ks\pack".to_string()
    });
    let pack = Pack::open(&PathBuf::from(&dir)).expect("open pack");
    let gpu = backend::gpu_device();
    let cpu = backend::cpu_device();

    let entry = pack
        .manifest
        .tensors
        .iter()
        .filter(|t| matches!(t.role, Role::Linear | Role::Expert { .. }) && t.shape.len() == 2)
        .filter(|t| t.shape.iter().product::<usize>() <= MAX_PARAMS)
        .max_by_key(|t| t.shape.iter().product::<usize>())
        .expect("a 2-D weight");
    let [k, n] = [entry.shape[0], entry.shape[1]];
    let params = k * n;
    println!(
        "{} [{k}, {n}] = {:.1} M params\n",
        entry.name,
        f64_from_usize(params) / 1e6
    );

    let xs: Vec<f32> = (0..k)
        .map(|i| (f32_from_usize(i % 13) - 6.0) / 6.0)
        .collect();
    let want = {
        let f32s = pack.read_f32(entry).expect("read f32");
        let x = Tensor::<2>::from_data(TensorData::new(xs.clone(), [1, k]), &cpu);
        let w = Tensor::<2>::from_data(TensorData::new(f32s, [k, n]), &cpu);
        x.matmul(w)
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .unwrap()
    };
    let scale = want.iter().map(|v| v.abs()).fold(1e-6, f32::max);

    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    let report = |how: &str, w: Option<Tensor<2>>, bits: usize| {
        let verdict = w
            .and_then(|w| gpu_rel_error(&gpu, &xs, w, &want, scale))
            .map_or_else(|| "PANIC".to_string(), |worst| format!("rel {worst:.4}"));
        // Scales ride along at f32 per 32-value block; count them, or the
        // "16x smaller" claim for Q2 is a third off.
        // f16 carries no block scales; the quantized rungs do.
        let bytes = params * bits / 8 + if bits >= 16 { 0 } else { (params / 32) * 4 };
        println!(
            "  {how:<28} {verdict:<14} resident {:>6.1} MiB",
            f64_from_usize(bytes) / f64::from(1 << 20)
        );
    };

    for precision in [Precision::F16, Precision::Q8, Precision::Q4] {
        let bits = match precision {
            Precision::F16 => 16,
            Precision::Q8 => 8,
            _ => 4,
        };
        report(
            &format!("{precision:?} from pack"),
            pack.tensor::<2>(entry, precision, &gpu).ok(),
            bits,
        );
    }

    // Demotion: requantize what is already resident, no disk.
    for (from, to, bits) in [
        (Precision::Q8, QuantPolicy::Q4, 4),
        (Precision::Q8, QuantPolicy::Q2, 2),
        (Precision::Q4, QuantPolicy::Q2, 2),
    ] {
        let gpu2 = gpu.clone();
        let w = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let resident = pack.tensor::<2>(entry, from, &gpu2).ok()?;
            Some(quantize_weight(to, resident.dequantize()))
        }))
        .ok()
        .flatten();
        report(&format!("{from:?} -> demote to {to:?}"), w, bits);
    }
    std::panic::set_hook(hook);
}
