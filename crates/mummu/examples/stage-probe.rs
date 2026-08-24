//! Can VRAM be used as a WORKING SET instead of permanent residency?
//!
//! The hot-swap design: weights live in host RAM, and each layer's clusters
//! are staged into VRAM, computed, and evicted. That only wins if
//! host->VRAM staging is fast enough that GPU compute + staging beats CPU
//! compute. This measures the staging leg (PCIe) at real cluster sizes.
use std::time::Instant;

use burn::tensor::{Distribution, Tensor};

fn main() {
    let cpu = mummu::backend::cpu_device();
    let gpu = mummu::backend::gpu_device();
    let hidden = 5120usize;

    // One FFN cluster of the 27B: 17408/32 = 544 neurons, three projections.
    let cluster_cols = 544usize;
    for (label, rows, cols) in [
        ("one cluster  gate [5120,544]", hidden, cluster_cols),
        ("layer slab   gate [5120,17408]", hidden, 17408),
    ] {
        let w_cpu = Tensor::<2>::random([rows, cols], Distribution::Default, &cpu);
        let bytes = (rows * cols * 4) as f64;
        // warm
        let _ = w_cpu.clone().to_device(&gpu).into_data();
        let n = 10;
        let t = Instant::now();
        for _ in 0..n {
            let _ = w_cpu.clone().to_device(&gpu).into_data();
        }
        let per = t.elapsed().as_secs_f64() / f64::from(n);
        println!(
            "{label}: {:>7.2} ms  {:.1} MB -> {:.1} GB/s staging",
            per * 1e3,
            bytes / 1e6,
            bytes / per / 1e9
        );
    }

    // The comparison that decides it: for ONE layer's FFN, is
    // (stage weights + GPU matmul) faster than (CPU matmul)?
    let x_cpu = Tensor::<2>::random([1, hidden], Distribution::Default, &cpu);
    let w_cpu = Tensor::<2>::random([hidden, 17408], Distribution::Default, &cpu);
    let n = 5;

    let t = Instant::now();
    for _ in 0..n {
        let _ = x_cpu.clone().matmul(w_cpu.clone()).into_data();
    }
    let cpu_ms = t.elapsed().as_secs_f64() * 1e3 / f64::from(n);

    let t = Instant::now();
    for _ in 0..n {
        let w = w_cpu.clone().to_device(&gpu);
        let x = x_cpu.clone().to_device(&gpu);
        let _ = x.matmul(w).into_data();
    }
    let stage_ms = t.elapsed().as_secs_f64() * 1e3 / f64::from(n);

    // And: already-resident GPU compute (no staging), the upper bound.
    let w_gpu = w_cpu.clone().to_device(&gpu);
    let x_gpu = x_cpu.clone().to_device(&gpu);
    let _ = x_gpu.clone().matmul(w_gpu.clone()).into_data();
    let t = Instant::now();
    for _ in 0..n {
        let _ = x_gpu.clone().matmul(w_gpu.clone()).into_data();
    }
    let res_ms = t.elapsed().as_secs_f64() * 1e3 / f64::from(n);

    println!("\nOne layer FFN [1,5120]x[5120,17408]:");
    println!("  CPU compute (weights already in RAM): {cpu_ms:>8.2} ms");
    println!("  GPU stage + compute (hot-swap)      : {stage_ms:>8.2} ms");
    println!("  GPU compute, already resident       : {res_ms:>8.2} ms");
}
