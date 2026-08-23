//! Per-layer overhead probe: how long does ONE decode step of the qwen35
//! trunk cost, broken into the pieces? The bandwidth probe says weight
//! traffic accounts for a fraction of a second per token, so the rest is
//! overhead — this locates it.
use std::time::Instant;

use burn::tensor::{Distribution, Tensor};

fn main() {
    let device = mummu::backend::cpu_device();
    let hidden = 5120usize;
    let n = 30;

    // A decode-shaped matmul, timed alone.
    let x = Tensor::<2>::random([1, hidden], Distribution::Default, &device);
    let w = Tensor::<2>::random([hidden, hidden], Distribution::Default, &device);
    let t = Instant::now();
    for _ in 0..n {
        let _ = x.clone().matmul(w.clone()).into_data();
    }
    println!("matmul [1,5120]x[5120,5120] + readback: {:.2} ms", t.elapsed().as_secs_f64()*1e3/f64::from(n));

    // Same matmul WITHOUT the readback (lazy/queued work only).
    let t = Instant::now();
    for _ in 0..n {
        let _ = x.clone().matmul(w.clone());
    }
    println!("matmul alone (no readback):            {:.2} ms", t.elapsed().as_secs_f64()*1e3/f64::from(n));

    // A tiny elementwise op + readback: pure per-op overhead.
    let t = Instant::now();
    for _ in 0..n {
        let _ = x.clone().mul_scalar(1.0001).into_data();
    }
    println!("tiny op + readback (per-op overhead):  {:.2} ms", t.elapsed().as_secs_f64()*1e3/f64::from(n));

    // Chain of 20 small ops then ONE readback — how much does op count cost?
    let t = Instant::now();
    for _ in 0..n {
        let mut y = x.clone();
        for _ in 0..20 {
            y = y.mul_scalar(1.0001);
        }
        let _ = y.into_data();
    }
    println!("20 chained ops + 1 readback:           {:.2} ms", t.elapsed().as_secs_f64()*1e3/f64::from(n));
}
