//! Does this Burn build do quantized matmul on the GPU, and is it correct?
//!
//! This is the single question that decides mummu's decode throughput. When
//! a Q4 weight can be multiplied **as Q4** on the device, a 16 GB model fits
//! a 16 GB card and every layer runs there (llama.cpp fits 59/66 layers of
//! the 27B in 12.4 GiB that way). When it cannot, the weight has to be
//! dequantized to f32 first — 8x the bytes — so most of the model spills to
//! the host and decode crawls.
//!
//! Three things are checked per backend, in order, because each only matters
//! if the previous one held:
//!
//! 1. **Does it run at all** (0.21's CUDA path panicked in kernel expansion:
//!    "Cast element count must match if input is not scalar")?
//! 2. **Is it correct** — same answer as dequantize-then-matmul, within
//!    quantization error?
//! 3. **Is it faster than dequantizing**, and does it avoid the f32 blowup?
//!
//! Run:
//! ```text
//! cargo run --release -p mummu --example qmatmul-probe               # wgpu + cpu
//! cargo run --release -p mummu --features cuda --example qmatmul-probe   # + cuda
//! ```

#![warn(clippy::pedantic, clippy::nursery, clippy::all)]

use std::time::Instant;

use burn::tensor::{Device, Distribution, Tensor};
use mummu::quant::{QuantPolicy, quantize_weight};

/// Timed runs per quantized arm. Timing discipline: warm once (autotune +
/// kernel compile land there, and a cold call reads ~20x its steady cost),
/// then average this many runs. A single-shot number measures launch
/// latency, not throughput.
const RUNS: u32 = 20;

/// One quantized arm: warm once and read the result back, then time `RUNS`
/// matmuls of `act` against `qw` — dequantized on-device first when
/// `dequantize` is set. Returns the warm result and seconds per run.
fn timed_arm(act: &Tensor<2>, qw: &Tensor<2>, dequantize: bool) -> (Vec<f32>, f64) {
    let rhs = || {
        if dequantize {
            qw.clone().dequantize()
        } else {
            qw.clone()
        }
    };
    let warm = act.clone().matmul(rhs()).into_data();
    let out = warm.convert::<f32>().try_to_vec::<f32>().expect("readback");
    let started = Instant::now();
    for _ in 0..RUNS {
        let _ = act.clone().matmul(rhs()).into_data();
    }
    (out, started.elapsed().as_secs_f64() / f64::from(RUNS))
}

fn probe(label: &str, device: &Device) {
    println!("\n=== {label} ===");
    // Decode shape: one token against a 27B-sized projection.
    let (m, k, n) = (1usize, 5120usize, 6144usize);
    let act = Tensor::<2>::random([m, k], Distribution::Default, device);
    let weight = Tensor::<2>::random([k, n], Distribution::Default, device);

    // Reference: plain f32 matmul.
    let Ok(want) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        act.clone()
            .matmul(weight.clone())
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .expect("f32 readback")
    })) else {
        println!("  f32 matmul PANICKED — backend unusable here");
        return;
    };

    // The floor: same shape, plain f32, warm + averaged.
    {
        let _ = act.clone().matmul(weight.clone()).into_data();
        let started = Instant::now();
        for _ in 0..20 {
            let _ = act.clone().matmul(weight.clone()).into_data();
        }
        println!(
            "  f32 matmul (floor): {:.2} ms",
            started.elapsed().as_secs_f64() * 1e3 / 20.0
        );
    }

    for policy in [QuantPolicy::Q8, QuantPolicy::Q4] {
        let Ok(qw) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            quantize_weight(policy, weight.clone())
        })) else {
            println!("  {policy:?}: quantize PANICKED");
            continue;
        };

        // (1) does a QUANTIZED matmul run at all?
        let direct =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| timed_arm(&act, &qw, false)));

        // The workaround we ship today: dequantize on-device, then matmul.
        let deq =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| timed_arm(&act, &qw, true)));

        match (direct, deq) {
            (Ok((dv, dt)), Ok((qv, qt))) => {
                // (2) correctness, against the f32 reference.
                let err = |v: &Vec<f32>| {
                    v.iter()
                        .zip(&want)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max)
                        / want.iter().map(|w| w.abs()).fold(1e-6, f32::max)
                };
                let (ed, eq) = (err(&dv), err(&qv));
                // The two paths must also agree with EACH OTHER — a direct
                // quantized matmul that silently computes something else is
                // worse than one that panics.
                let disagree = dv
                    .iter()
                    .zip(&qv)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                println!(
                    "  {policy:?}: direct q_matmul {:>7.2} ms (rel err {ed:.4})  |  dequantize+matmul {:>7.2} ms (rel err {eq:.4})  |  paths differ by {disagree:.4}",
                    dt * 1e3,
                    qt * 1e3
                );
                if ed > 0.25 {
                    println!("    ^ direct path looks WRONG (rel err {ed:.3}) — do not use it");
                } else if dt < qt {
                    println!("    ^ direct path is correct AND {:.1}x faster", qt / dt);
                }
            }
            (Err(_), Ok((qv, qt))) => {
                let eq = qv
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                println!(
                    "  {policy:?}: direct q_matmul PANICKED; dequantize+matmul {:>7.2} ms (abs err {eq:.4}) — the workaround is required here",
                    qt * 1e3
                );
            }
            (_, Err(_)) => println!("  {policy:?}: dequantize path PANICKED too"),
        }
    }
}

fn main() {
    // Panics inside the probe are the ANSWER, not a crash — keep them quiet.
    std::panic::set_hook(Box::new(|_| {}));

    probe("CPU (flex)", &mummu::backend::cpu_device());
    if mummu::backend::use_gpu() {
        probe("GPU (wgpu)", &mummu::backend::gpu_device());
    } else {
        println!("\n=== GPU (wgpu) === skipped: no adapter");
    }
    #[cfg(feature = "cuda")]
    probe("GPU (cuda)", &mummu::backend::cuda_device());
    #[cfg(not(feature = "cuda"))]
    println!("\n=== GPU (cuda) === skipped: build with --features cuda");

    let _ = std::panic::take_hook();
}
