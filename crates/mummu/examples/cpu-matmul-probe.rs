//! Where does CPU decode time go? Times the two matmul shapes a decode
//! actually runs — matrix-VECTOR (m=1, one token) vs a small batch (m=8) —
//! against the flex backend, and reports achieved GB/s.
//!
//! Decode is memory-bandwidth bound: at m=1 every weight byte is read once
//! for a single output row, so the arithmetic intensity is ~1 flop/byte and
//! no amount of threading helps. Batching raises reuse linearly.
use std::time::Instant;

use burn::tensor::{Distribution, Tensor};

fn main() {
    let device = mummu::backend::cpu_device();
    let hidden = 5120usize;
    for (label, m, n, k) in [
        ("decode  q_proj  m=1", 1usize, 6144usize, hidden),
        ("decode  ffn     m=1", 1, 17408, hidden),
        ("batch-8 ffn     m=8", 8, 17408, hidden),
        ("prefill ffn     m=64", 64, 17408, hidden),
    ] {
        let a = Tensor::<2>::random([m, k], Distribution::Default, &device);
        let b = Tensor::<2>::random([k, n], Distribution::Default, &device);
        // warm
        let _ = a.clone().matmul(b.clone()).into_data();
        let iters = if m == 1 { 20 } else { 5 };
        let t = Instant::now();
        for _ in 0..iters {
            let _ = a.clone().matmul(b.clone()).into_data();
        }
        let per = t.elapsed().as_secs_f64() / f64::from(iters);
        let bytes = (k * n * 4) as f64; // the weight matrix dominates traffic
        println!(
            "{label}: {:>8.2} ms/call  weights {:.1} MB  -> {:.1} GB/s  ({:.2} GFLOP/s)",
            per * 1e3,
            bytes / 1e6,
            bytes / per / 1e9,
            (2.0 * (m * n * k) as f64) / per / 1e9,
        );
    }
}
