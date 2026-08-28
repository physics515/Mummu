//! Parity + speed of the packed m=1 Q4S GEMV on the real GPU, at the 27B's
//! production shapes (gate/up: [5120, W], down: [W, 5120], W = 22 clusters
//! x 544). Baseline is what clusters run today: m=1 `x.matmul(wq)` through
//! burn's q_matmul (the dequantize-first path measured at 0.91 ms/cluster).
use burn::tensor::{Distribution, Tensor};
use mummu::backend;
use mummu::nn::try_q4s_gemv;
use mummu::quant::{QuantPolicy, quantize_weight};
use std::time::Instant;

fn bench(label: &str, iters: usize, f: &mut dyn FnMut() -> Tensor<2>) -> f64 {
    for _ in 0..3 {
        let _ = f().into_data(); // warm (autotune, pools)
    }
    let t0 = Instant::now();
    let mut last = None;
    for _ in 0..iters {
        last = Some(f());
    }
    let _ = last.unwrap().into_data(); // sync
    let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
    println!("  {label}: {ms:.3} ms/call");
    ms
}

fn main() {
    let gpu = backend::gpu_device();
    let w_width = 22 * 544usize;
    for (k, n, label) in [
        (5120usize, w_width, "gate/up"),
        (w_width, 5120usize, "down"),
    ] {
        println!("[{label}] W [{k}, {n}] Q4S, x [1, {k}] f32");
        let w = Tensor::<2>::random([k, n], Distribution::Uniform(-0.5, 0.5), &gpu);
        let wq = quantize_weight(
            if std::env::var("PROBE_Q8").is_ok() {
                QuantPolicy::Q8
            } else {
                QuantPolicy::Q4
            },
            w,
        );
        let x = Tensor::<2>::random([1, k], Distribution::Uniform(-1.0, 1.0), &gpu);

        let got = try_q4s_gemv(&x, &wq).expect("packed path must engage");
        let want = x.clone().matmul(wq.clone().dequantize());
        let diff = got
            .clone()
            .sub(want.clone())
            .abs()
            .max()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()[0];
        let scale = want.abs().max().into_data().try_to_vec::<f32>().unwrap()[0].max(1e-6);
        println!("  parity vs dequant reference: rel {:.2e}", diff / scale);

        let packed = bench("packed q4s_gemv", 50, &mut || {
            try_q4s_gemv(&x, &wq).unwrap()
        });
        let baseline = bench("today's q_matmul", 20, &mut || x.clone().matmul(wq.clone()));
        println!("  speedup: {:.1}x", baseline / packed);
    }
}
