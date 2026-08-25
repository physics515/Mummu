//! The last differentiator, isolated: the server spawns a FRESH thread per
//! layer per token, and `drain.build` is that thread's first-ever flex op —
//! after a process has thousands of resident flex pool slabs. If first-op-
//! on-a-fresh-thread scales with resident state, the 27 ms is per-thread
//! flex init, and the fix is a long-lived worker per device.
use burn::tensor::{Tensor, TensorData};
use mummu::backend;
use std::time::Instant;

fn round(label: &str, cpu: &burn::tensor::Device) {
    let mut first = Vec::new();
    let mut second = Vec::new();
    for _ in 0..8 {
        let cpu = cpu.clone();
        let h = std::thread::spawn(move || {
            let d = TensorData::new(vec![0.5f32; 5120], [1usize, 5120]);
            let t0 = Instant::now();
            let t = Tensor::<2>::from_data(d, &cpu);
            let t = t.add_scalar(0.0);
            let _ = t.into_data();
            let a = t0.elapsed();
            let d = TensorData::new(vec![0.5f32; 5120], [1usize, 5120]);
            let t1 = Instant::now();
            let t = Tensor::<2>::from_data(d, &cpu);
            let t = t.add_scalar(0.0);
            let _ = t.into_data();
            (a, t1.elapsed())
        });
        let (a, b) = h.join().unwrap();
        first.push(a.as_micros());
        second.push(b.as_micros());
    }
    println!("  [{label}] fresh-thread first op µs: {first:?}");
    println!("  [{label}]              second op µs: {second:?}");
}

fn main() {
    let gb: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let cpu = backend::cpu_device();
    round("baseline", &cpu);
    // Simulate the weight-slab population: many medium tensors held alive,
    // the shape of a loaded 27B on flex.
    let mut hold = Vec::new();
    for _ in 0..(gb * 128) {
        // 8 MB each
        hold.push(Tensor::<1>::zeros([2 * 1024 * 1024], &cpu));
    }
    println!("resident: {} tensors x 8MB = {gb} GB", hold.len());
    round("big-heap", &cpu);
    drop(hold);
}
