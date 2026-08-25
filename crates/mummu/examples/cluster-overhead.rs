//! What does splitting an FFN into clusters cost, in dispatch?
//!
//! Partitioning exists so placement can be fine-grained: 2048 clusters can be
//! spread across devices, one 64-layer FFN cannot. But the same split means a
//! token issues 2048 matmuls where a fused implementation issues ~64, and a
//! GPU matmul has a fixed per-launch cost that does not care how small the
//! work is.
//!
//! This measures the trade directly on the discrete GPU: one full-width FFN
//! matmul against the same arithmetic done as N cluster-width matmuls summed.
//! If the split is nearly free, placement granularity is worth keeping as it
//! is. If it is not, then no amount of scheduling reaches a fused runtime,
//! and the partition width itself is the thing to change.
use burn::tensor::{Device, DeviceKind, Tensor, TensorData};
use mummu::backend;
use std::time::Instant;

/// Warm, then time `rounds` iterations; milliseconds per iteration.
fn timed(rounds: usize, mut f: impl FnMut()) -> f64 {
    f(); // warm: kernel compilation and autotune
    let started = Instant::now();
    for _ in 0..rounds {
        f();
    }
    started.elapsed().as_secs_f64() * 1000.0 / rounds as f64
}

fn main() {
    let device = Device::wgpu(DeviceKind::DiscreteGpu(0));
    let dtype = backend::float_dtype(&device);

    // This model's real geometry: hidden 5120, FFN 17408, clusters of 544.
    const HIDDEN: usize = 5120;
    const INTER: usize = 17408;
    const CLUSTER: usize = 544;
    let clusters = INTER / CLUSTER;

    let x = Tensor::<2>::from_data(
        TensorData::new(vec![0.02f32; HIDDEN], [1, HIDDEN]),
        (&device, dtype),
    );

    // One full-width weight, and the same columns as separate cluster weights.
    let whole = Tensor::<2>::from_data(
        TensorData::new(vec![0.01f32; HIDDEN * INTER], [HIDDEN, INTER]),
        (&device, dtype),
    );
    let parts: Vec<Tensor<2>> = (0..clusters)
        .map(|_| {
            Tensor::<2>::from_data(
                TensorData::new(vec![0.01f32; HIDDEN * CLUSTER], [HIDDEN, CLUSTER]),
                (&device, dtype),
            )
        })
        .collect();

    println!("hidden {HIDDEN}, inter {INTER}, {clusters} clusters of {CLUSTER}\n");

    let one = timed(20, || {
        let y = x.clone().matmul(whole.clone());
        // Force completion: an async queue would otherwise time nothing.
        let _ = y.into_data().convert::<f32>().to_vec::<f32>().ok();
    });
    println!("  one full-width matmul          {one:7.2} ms");

    // The partitioned form:one matmul per cluster, results summed. This is
    // what `ExpertPool::run_dense` does for one layer's remote clusters.
    let split = timed(20, || {
        let mut acc: Option<Tensor<2>> = None;
        for w in &parts {
            let y = x.clone().matmul(w.clone());
            acc = Some(match acc {
                Some(a) => Tensor::cat(vec![a, y], 1),
                None => y,
            });
        }
        if let Some(a) = acc {
            let _ = a.into_data().convert::<f32>().to_vec::<f32>().ok();
        }
    });
    println!(
        "  {clusters} cluster matmuls           {split:7.2} ms   ({:.1}x)",
        split / one
    );

    // How much of that 2.4x is recoverable by using FEWER, WIDER clusters?
    // Cluster width is a pack-time choice, and it trades placement
    // granularity against dispatch count — the one knob here that is genuine
    // tuning rather than a kernel change.
    println!(
        "
  cluster width sweep (same total arithmetic):"
    );
    for width in [544usize, 1088, 2176, 4352, 8704] {
        if INTER % width != 0 {
            continue;
        }
        let n = INTER / width;
        let ws: Vec<Tensor<2>> = (0..n)
            .map(|_| {
                Tensor::<2>::from_data(
                    TensorData::new(vec![0.01f32; HIDDEN * width], [HIDDEN, width]),
                    (&device, dtype),
                )
            })
            .collect();
        let ms = timed(10, || {
            let mut acc: Option<Tensor<2>> = None;
            for w in &ws {
                let y = x.clone().matmul(w.clone());
                acc = Some(match acc {
                    Some(a) => Tensor::cat(vec![a, y], 1),
                    None => y,
                });
            }
            if let Some(a) = acc {
                let _ = a.into_data().convert::<f32>().to_vec::<f32>().ok();
            }
        });
        println!(
            "    {n:2} x {width:5}   {ms:7.2} ms   ({:.2}x fused)",
            ms / one
        );
    }

    // And the same full-width matmul with a QUANTIZED weight — what the
    // model actually holds on the card. Decode is bandwidth-bound, so 4-bit
    // weights moving an eighth of the bytes ought to be much FASTER than
    // f32. Whether burn's kernel delivers that is the question that decides
    // whether the gap to a fused runtime is about placement or about kernels.
    for policy in [mummu::quant::QuantPolicy::Q8, mummu::quant::QuantPolicy::Q4] {
        let qw = mummu::quant::quantize_weight(policy, whole.clone());
        let ms = timed(20, || {
            let y = x.clone().matmul(qw.clone());
            let _ = y.into_data().convert::<f32>().to_vec::<f32>().ok();
        });
        println!(
            "  one full-width at {policy:?}          {ms:7.2} ms   ({:.2}x f32)",
            ms / one
        );
    }

    // f16: half the real traffic of f32, and — unlike a quantized rung — no
    // dequantize step for the kernel to undo the saving with. If the Q4 path
    // materializes f32 regardless, f16 should beat BOTH, and it is reachable
    // by the existing ladder without any new kernel.
    {
        let half = whole.clone().cast(burn::tensor::DType::F16);
        let xh = x.clone().cast(burn::tensor::DType::F16);
        let ms = timed(20, || {
            let y = xh.clone().matmul(half.clone());
            let _ = y.into_data().convert::<f32>().to_vec::<f32>().ok();
        });
        println!(
            "  one full-width at F16            {ms:7.2} ms   ({:.2}x f32)",
            ms / one
        );
    }

    // Per token this model does 64 layers x 3 projections of the above.
    let per_token_whole = one * 64.0 * 3.0;
    let per_token_split = split * 64.0 * 3.0;
    println!("\n  extrapolated to 64 layers x 3 projections:");
    println!("    fused    {per_token_whole:8.0} ms/token");
    println!("    clustered{per_token_split:8.0} ms/token");
    println!("    ollama        62 ms/token (measured, native Q4_K)");
}
