//! What does a split trunk cost in round trips?
//!
//! With the trunk on the host and the FFN on GPUs, every layer ships its
//! activations across and back: 64 layers x 2 crossings per token, for 20 KB
//! each. The DATA is nothing; the question is the latency and synchronization
//! per crossing, which does not care how small the payload is.
//!
//! This is the largest unexplained term in the 27B's decode time — ~2.2 s of
//! 3.91 s is not accounted for by the trunk's arithmetic or the FFN makespan
//! — so it is worth measuring before designing around it.
use burn::tensor::{Device, DeviceKind, Tensor, TensorData};
use mummu::backend;
use std::time::Instant;

const HIDDEN: usize = 5120;
const LAYERS: usize = 64;

fn timed(rounds: usize, mut f: impl FnMut()) -> f64 {
    f();
    let started = Instant::now();
    for _ in 0..rounds {
        f();
    }
    started.elapsed().as_secs_f64() * 1000.0 / rounds as f64
}

fn main() {
    let dgpu = Device::wgpu(DeviceKind::DiscreteGpu(0));
    let igpu = Device::wgpu(DeviceKind::IntegratedGpu(0));
    let cpu = backend::cpu_device();
    let dtype = backend::float_dtype(&cpu);

    let host = Tensor::<2>::from_data(
        TensorData::new(vec![0.5f32; HIDDEN], [1, HIDDEN]),
        (&cpu, dtype),
    );
    println!(
        "one decode activation: [1, {HIDDEN}] f32 = {} KiB\n",
        HIDDEN * 4 / 1024
    );

    // A single crossing, host -> device.
    let up = timed(50, || {
        let d = host.clone().to_device(&dgpu);
        // Touch it so the move cannot be elided or left queued.
        let _ = d.into_data().convert::<f32>().to_vec::<f32>().ok();
    });
    println!("  host -> dGPU -> host          {up:6.2} ms");

    // The shape a split trunk actually produces: across and back, per layer.
    let round = timed(20, || {
        let mut x = host.clone();
        for _ in 0..LAYERS {
            let on_gpu = x.clone().to_device(&dgpu);
            // Something trivial on the device, standing in for the FFN group.
            let y = on_gpu.clone().add(on_gpu);
            x = y.to_device(&cpu);
        }
        let _ = x.into_data().convert::<f32>().to_vec::<f32>().ok();
    });
    println!("  {LAYERS} layers, trunk on host   {round:6.2} ms/token   <- paid every token");

    // And with the iGPU in the mix, which cubecl cannot reach peer-to-peer,
    // so `backend::move_to` stages it through host memory: two crossings
    // instead of one, each way.
    let three = timed(10, || {
        let mut x = host.clone();
        for l in 0..LAYERS {
            let dev = if l % 8 == 0 { &igpu } else { &dgpu };
            let on = backend::move_to(x.clone(), dev);
            let y = on.clone().add(on);
            x = backend::move_to(y, &cpu);
        }
        let _ = x.into_data().convert::<f32>().to_vec::<f32>().ok();
    });
    println!("  same, 1 layer in 8 on the iGPU {three:6.2} ms/token");

    // The other candidate for the missing time: the trunk is not just its
    // matmuls. Every layer runs norms, rope, gating and elementwise work on
    // [1, 5120] — trivial arithmetic, but per-op overhead does not care.
    // A qwen3.5 layer runs on the order of 20 such ops; 64 layers is ~1280.
    {
        let small = Tensor::<2>::from_data(
            TensorData::new(vec![0.5f32; HIDDEN], [1, HIDDEN]),
            (&cpu, dtype),
        );
        let ops = 20 * LAYERS;
        let ms = timed(5, || {
            let mut x = small.clone();
            for _ in 0..ops {
                x = x.clone().mul_scalar(1.0001).add_scalar(0.0);
            }
            let _ = x.into_data().convert::<f32>().to_vec::<f32>().ok();
        });
        println!(
            "
  {ops} small elementwise ops on flex  {ms:7.2} ms/token"
        );
        let g = Tensor::<2>::from_data(
            TensorData::new(vec![0.5f32; HIDDEN], [1, HIDDEN]),
            (&dgpu, dtype),
        );
        let gms = timed(5, || {
            let mut x = g.clone();
            for _ in 0..ops {
                x = x.clone().mul_scalar(1.0001).add_scalar(0.0);
            }
            let _ = x.into_data().convert::<f32>().to_vec::<f32>().ok();
        });
        println!("  same on the dGPU                {gms:7.2} ms/token");
    }

    // The alternative: trunk and FFN on the SAME device, no crossing at all.
    let resident = Tensor::<2>::from_data(
        TensorData::new(vec![0.5f32; HIDDEN], [1, HIDDEN]),
        (&dgpu, dtype),
    );
    let none = timed(20, || {
        let mut x = resident.clone();
        for _ in 0..LAYERS {
            x = x.clone().add(x);
        }
        let _ = x.into_data().convert::<f32>().to_vec::<f32>().ok();
    });
    println!("  {LAYERS} layers, all on the dGPU {none:6.2} ms/token   <- what fitting buys");
}
