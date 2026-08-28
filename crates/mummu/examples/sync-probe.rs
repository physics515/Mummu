//! The last unmeasured suspect: cross-device synchronization under WSL2
//! GPU-PV. The 27B alternates CPU trunk <-> GPU FFN clusters 65 times per
//! token; if each boundary costs a full device round trip, that is the
//! missing ~28 s.
use std::time::Instant;

use burn::tensor::{Distribution, Tensor};

fn main() {
    let cpu = mummu::backend::cpu_device();
    // wgpu here (the host build has no `cuda` feature); the point being
    // measured — per-boundary cross-device sync latency — is the same shape.
    let gpu = mummu::backend::gpu_device();
    let n = 20;
    let hidden = 5120usize;

    // A decode-sized activation crossing devices, both ways.
    let x_cpu = Tensor::<2>::random([1, hidden], Distribution::Default, &cpu);
    let _ = x_cpu.clone().to_device(&gpu).into_data(); // warm the GPU context

    let t = Instant::now();
    for _ in 0..n {
        let _ = x_cpu.clone().to_device(&gpu).into_data();
    }
    println!(
        "CPU->GPU->readback [1,5120]: {:.3} ms",
        t.elapsed().as_secs_f64() * 1e3 / f64::from(n)
    );

    let x_gpu = Tensor::<2>::random([1, hidden], Distribution::Default, &gpu);
    let t = Instant::now();
    for _ in 0..n {
        let _ = x_gpu.clone().to_device(&cpu).into_data();
    }
    println!(
        "GPU->CPU->readback [1,5120]: {:.3} ms",
        t.elapsed().as_secs_f64() * 1e3 / f64::from(n)
    );

    // A GPU matmul at cluster size, with a forced sync each time (what a
    // per-layer boundary costs).
    let w = Tensor::<2>::random([hidden, 2176], Distribution::Default, &gpu);
    let t = Instant::now();
    for _ in 0..n {
        let _ = x_gpu.clone().matmul(w.clone()).into_data();
    }
    let per = t.elapsed().as_secs_f64() / f64::from(n);
    println!(
        "GPU matmul [1,5120]x[5120,2176] + sync: {:.3} ms",
        per * 1e3
    );
    println!(
        "  x65 layers = {:.2} s/token of GPU round trips",
        per * 65.0
    );
}
