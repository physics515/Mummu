//! The missing server ingredient, isolated: the trunk saturates the flex
//! thread pool from the main thread while the dGPU worker tries to run one
//! tiny flex op. If the tiny op inflates from nanoseconds to tens of
//! milliseconds here, the 27 ms "first-op cost" was never an op cost at all
//! — it is pool starvation, and the fix is to keep worker threads out of
//! the flex pool entirely.
use burn::tensor::{Tensor, TensorData};
use mummu::backend;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let cpu = backend::cpu_device();
    let stop = Arc::new(AtomicBool::new(false));
    let worker = {
        let stop = stop.clone();
        let cpu = cpu.clone();
        std::thread::spawn(move || {
            let mut times = Vec::new();
            while !stop.load(Ordering::Relaxed) {
                let d = TensorData::new(vec![0.5f32; 5120], [1usize, 5120]);
                let t0 = Instant::now();
                let t = Tensor::<2>::from_data(d, &cpu);
                let t = t.add_scalar(0.0);
                let _ = t.into_data();
                times.push(t0.elapsed());
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            times
        })
    };
    // The trunk stand-in: big matmuls on the main thread, back to back.
    let a = Tensor::<2>::random(
        [2048, 2048],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &cpu,
    );
    let t0 = Instant::now();
    let mut n = 0u32;
    while t0.elapsed().as_secs() < 5 {
        let _ = a.clone().matmul(a.clone()).into_data();
        n += 1;
    }
    stop.store(true, Ordering::Relaxed);
    let times = worker.join().unwrap();
    let mut us: Vec<u128> = times.iter().map(|d| d.as_micros()).collect();
    us.sort_unstable();
    let pct = |p: usize| us.get(us.len() * p / 100).copied().unwrap_or(0);
    println!(
        "main ran {n} matmuls; worker tiny-op over {} samples: p50 {}µs p90 {}µs p99 {}µs max {}µs",
        us.len(),
        pct(50),
        pct(90),
        pct(99),
        us.last().copied().unwrap_or(0)
    );
}
