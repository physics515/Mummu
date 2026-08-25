//! Which of this machine's devices are worth giving work to, and how much?
//!
//! Placement across a discrete GPU, an integrated GPU and the CPU only pays
//! if each device is actually faster than moving the work elsewhere. Two
//! things are easy to assume and wrong:
//!
//! * that an integrated GPU beats the CPU — it shares system RAM with the
//!   CPU, and decode is memory-bound, so it may be no faster;
//! * that iGPU and CPU throughput **add** — they contend for the same
//!   memory controller, so running both can be slower than running one.
//!
//! This measures both: each device alone, then the CPU and iGPU together,
//! on the real weight shapes from a real pack.
use burn::tensor::{Device, DeviceKind, Tensor, TensorData};
use mummu::backend;
use mummu::pack::{Pack, Precision, Role};
use std::path::PathBuf;
use std::time::Instant;

/// Warm, then time `rounds` matmuls; returns milliseconds per matmul.
/// `None` if the device cannot run it at all.
fn time_matmul(device: &Device, w: &Tensor<2>, k: usize, m: usize, rounds: usize) -> Option<f64> {
    let w = w.clone();
    let device = device.clone();
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let xs: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) / 6.0).collect();
        let x = Tensor::<2>::from_data(
            TensorData::new(xs, [m, k]),
            (&device, backend::float_dtype(&device)),
        );
        // Warm: first call pays kernel compilation and autotune.
        let _ = x
            .clone()
            .matmul(w.clone())
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .ok()?;
        let started = Instant::now();
        for _ in 0..rounds {
            // `into_data` forces a readback, so the timing cannot be fooled
            // by an async queue that has not actually run yet.
            let _ = x
                .clone()
                .matmul(w.clone())
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .ok()?;
        }
        Some(started.elapsed().as_secs_f64() * 1000.0 / rounds as f64)
    }))
    .ok()
    .flatten()
}

fn main() {
    let dir = std::env::var("PACK_DIR").unwrap_or_else(|_| {
        r"D:\Docker Containers\mummu\models\qwen3.8-27b-ud-q4ks\pack".to_string()
    });
    let pack = Pack::open(&PathBuf::from(&dir)).expect("open pack");

    const MAX_PARAMS: usize = 128 << 20;
    let entry = pack
        .manifest
        .tensors
        .iter()
        .filter(|t| matches!(t.role, Role::Linear | Role::Expert { .. }) && t.shape.len() == 2)
        .filter(|t| t.shape.iter().product::<usize>() <= MAX_PARAMS)
        .max_by_key(|t| t.shape.iter().product::<usize>())
        .expect("a 2-D weight");
    let [k, n] = [entry.shape[0], entry.shape[1]];
    let params = k * n;
    // Q4 values + f32 block scales: what a decode step actually streams.
    let weight_bytes = params / 2 + params / 32 * 4;
    println!(
        "{} [{k}, {n}] = {:.1} M params, {:.0} MiB at Q4\n",
        entry.name,
        params as f64 / 1e6,
        weight_bytes as f64 / (1 << 20) as f64
    );

    let devices: Vec<(&str, Device)> = vec![
        ("dGPU (wgpu)", Device::wgpu(DeviceKind::DiscreteGpu(0))),
        ("iGPU (wgpu)", Device::wgpu(DeviceKind::IntegratedGpu(0))),
        ("CPU (flex)", backend::cpu_device()),
    ];

    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    // m=1 is decode, the shape that decides tok/s. m=16 stands in for a
    // small prefill, where there is enough arithmetic to hide latency.
    for m in [1usize, 16] {
        println!("m={m}:");
        for (label, device) in &devices {
            let Some(w) = pack.tensor::<2>(entry, Precision::Q4, device).ok() else {
                println!("  {label:<14} cannot hold the weight");
                continue;
            };
            match time_matmul(device, &w, k, m, 5) {
                None => println!("  {label:<14} FAILED"),
                Some(ms) => {
                    // Decode is bandwidth-bound, so bytes/second is the
                    // number that predicts tok/s — not FLOPs.
                    let gbs = weight_bytes as f64 / (ms / 1000.0) / 1e9;
                    println!("  {label:<14} {ms:8.2} ms   {gbs:6.1} GB/s of weights");
                }
            }
        }
    }

    // 0.2 GB/s on a DDR5 machine is far too slow to be "the CPU is slow";
    // it points at the *path*. Compare the rungs, and a dequantized weight,
    // on the CPU alone. Host RAM is not the scarce resource here (96 GiB), so
    // if float is faster then CPU-resident tensors should simply be stored
    // dequantized.
    println!(
        "
CPU only, by storage (m=1):"
    );
    {
        let cpu = backend::cpu_device();
        for (label, w) in [
            (
                "Q4 native",
                pack.tensor::<2>(entry, Precision::Q4, &cpu).ok(),
            ),
            (
                "Q8 native",
                pack.tensor::<2>(entry, Precision::Q8, &cpu).ok(),
            ),
            (
                "Q4 -> dequantized f32",
                pack.tensor::<2>(entry, Precision::Q4, &cpu)
                    .ok()
                    .map(Tensor::dequantize),
            ),
            // f32 on the host costs 4 B/param; f16 halves that. Worth it only
            // if burn-flex is not slower at it.
            (
                "Q4 -> dequantized f16",
                pack.tensor::<2>(entry, Precision::Q4, &cpu)
                    .ok()
                    .map(|w| w.dequantize().cast(burn::tensor::DType::F16)),
            ),
            (
                "F16 from pack",
                pack.tensor::<2>(entry, Precision::F16, &cpu).ok(),
            ),
        ] {
            match w.and_then(|w| time_matmul(&cpu, &w, k, 1, 3)) {
                None => println!("  {label:<20} FAILED"),
                Some(ms) => println!(
                    "  {label:<20} {ms:8.2} ms   {:6.1} GB/s of weights",
                    weight_bytes as f64 / (ms / 1000.0) / 1e9
                ),
            }
        }
    }

    // Do the CPU and the integrated GPU add, or contend? Run the same work on
    // both at once and compare against each alone. Anything close to the
    // slower of the two means they are sharing one memory controller and the
    // second device is buying nothing.
    println!("\nCPU + iGPU concurrently (m=1):");
    let igpu = Device::wgpu(DeviceKind::IntegratedGpu(0));
    let cpu = backend::cpu_device();
    let (wi, wc) = (
        pack.tensor::<2>(entry, Precision::Q4, &igpu).ok(),
        pack.tensor::<2>(entry, Precision::Q4, &cpu).ok(),
    );
    match (wi, wc) {
        (Some(wi), Some(wc)) => {
            // Warm both first so compilation is not counted as contention.
            let _ = time_matmul(&igpu, &wi, k, 1, 1);
            let _ = time_matmul(&cpu, &wc, k, 1, 1);
            let started = Instant::now();
            let handle = {
                let (igpu, wi) = (igpu.clone(), wi.clone());
                std::thread::spawn(move || time_matmul(&igpu, &wi, k, 1, 5))
            };
            let cpu_ms = time_matmul(&cpu, &wc, k, 1, 5);
            let igpu_ms = handle.join().ok().flatten();
            println!(
                "  iGPU {:?} ms, CPU {:?} ms, wall {:.0} ms for 5 rounds each",
                igpu_ms.map(|v| (v * 10.0).round() / 10.0),
                cpu_ms.map(|v| (v * 10.0).round() / 10.0),
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
        _ => println!("  one of the two could not hold the weight"),
    }
    std::panic::set_hook(hook);
}
