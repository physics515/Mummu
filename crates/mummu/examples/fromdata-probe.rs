//! The 27 ms mystery, minimal: on this stack, `Tensor::from_data` onto the
//! flex device plus a first op measured ~27 ms per call inside the server —
//! per layer, per token, 1.4 s/token — while the GPU readback feeding it
//! measured 4.9 µs. This reproduces (or fails to) outside the server, with
//! and without a wgpu device alive, splitting creation vs first-op vs read.
use burn::tensor::{Tensor, TensorData};
use mummu::backend;
use std::time::Instant;

fn phase(label: &str) {
    let cpu = backend::cpu_device();
    for i in 0..6 {
        let d = TensorData::new(vec![0.5f32; 5120], [1usize, 5120]);
        let t0 = Instant::now();
        let t = Tensor::<2>::from_data(d, &cpu);
        let create = t0.elapsed();
        let t1 = Instant::now();
        let t = t.add_scalar(0.0);
        let op = t1.elapsed();
        let t2 = Instant::now();
        let _ = t.into_data();
        println!(
            "  [{label}] iter {i}: from_data {create:?} | first op {op:?} | read {:?}",
            t2.elapsed()
        );
    }
    let a = Tensor::<2>::zeros([1, 5120], &cpu);
    let t3 = Instant::now();
    let _ = a.add_scalar(0.0).into_data();
    println!("  [{label}] zeros-created op+read: {:?}", t3.elapsed());
}

fn main() {
    println!("flex only:");
    phase("no-gpu");
    // Bring the wgpu device to life the way the server does, then repeat —
    // if the cost only appears with an accelerator context alive, it is an
    // interaction, not a flex bug.
    let gpu = backend::gpu_device();
    let g = Tensor::<2>::zeros([64, 64], &gpu);
    let _ = g.add_scalar(1.0).into_data();
    println!("with a live wgpu device:");
    phase("gpu-alive");
    // And with wgpu work IN FLIGHT, as during decode.
    let big = Tensor::<2>::zeros([4096, 4096], &gpu);
    let _busy = big.clone().matmul(big);
    println!("with wgpu work in flight:");
    phase("gpu-busy");
}

// Phase 4 lives in a second binary-visible fn to keep main readable? No —
// append inline via a shim: see fromdata-contend.rs.
