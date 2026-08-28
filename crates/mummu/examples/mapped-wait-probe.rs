//! Decisive split: does `into_data` on a wgpu tensor BLOCK on the GPU, or
//! does it return deferred-mapped bytes whose first CPU touch pays the
//! fence? Submit ~hundreds of ms of GPU work, then time into_data, the
//! flex from_data, and the first byte-touch separately.
use burn::tensor::{Tensor, TensorData};
use mummu::backend;
use std::time::Instant;

fn main() {
    let gpu = backend::gpu_device();
    let cpu = backend::cpu_device();
    let a = Tensor::<2>::random(
        [2048, 2048],
        burn::tensor::Distribution::Uniform(-0.1, 0.1),
        &gpu,
    );
    // Warm the kernels/autotune first.
    let _ = a.clone().matmul(a.clone()).into_data();
    for round in 0..3 {
        // A long GPU chain ending in a small result, like a remote FFN.
        let mut t = a.clone();
        for _ in 0..24 {
            t = t.matmul(a.clone());
            let m = t.clone().max().reshape([1, 1]);
            t = t.sub(m.expand([2048, 2048]));
        }
        let small = t.slice([0..1, 0..5120.min(2048)]);
        let t0 = Instant::now();
        let data = small.into_data();
        let d_read = t0.elapsed();
        let t1 = Instant::now();
        let ten = Tensor::<2>::from_data(data, &cpu);
        let d_build = t1.elapsed();
        let t2 = Instant::now();
        let ten = ten.add_scalar(0.0);
        let d_touch = t2.elapsed();
        let t3 = Instant::now();
        let v = ten.into_data().try_to_vec::<f32>().unwrap();
        println!(
            "round {round}: into_data {d_read:?} | from_data {d_build:?} | first touch {d_touch:?} | final read {:?} (v0={:.3})",
            t3.elapsed(),
            v[0]
        );
    }
}
