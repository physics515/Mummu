//! Is a pack tensor loaded at Q4 (or Q8) onto wgpu correct, and *repeatably*
//! correct, at this model's real dimensions?
//!
//! This is the question the earlier probes did not answer. They quantized a
//! synthetic weight ON the device — something production never does — and
//! then blamed the matmul when the result was garbage. Production loads
//! already-packed bytes from `q4.bin`/`q8.bin`, so that is what this loads.
//!
//! Determinism matters as much as correctness: a path that is right twice
//! and garbage the third time cannot be planned around, and mixed-precision
//! placement depends on being able to trust Q4 storage on the card.
use burn::tensor::{Tensor, TensorData};
use mummu::backend;
use mummu::pack::{Pack, Precision, Role};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

const ROUNDS: usize = 3;

fn main() {
    let dir = std::env::var("PACK_DIR").unwrap_or_else(|_| {
        r"D:\Docker Containers\mummu\models\qwen3.8-27b-ud-q4ks\pack".to_string()
    });
    let pack = Pack::open(&PathBuf::from(&dir)).expect("open pack");
    let gpu = backend::gpu_device();
    let cpu = backend::cpu_device();

    // The widest 2-D weights in the pack — the ones that actually matter for
    // residency, and the ones in the broken-matmul regime.
    let mut picks: Vec<_> = pack
        .manifest
        .tensors
        .iter()
        .filter(|t| matches!(t.role, Role::Linear | Role::Expert { .. }) && t.shape.len() == 2)
        .filter(|t| {
            t.precisions.contains_key(&Precision::Q4) && t.precisions.contains_key(&Precision::Q8)
        })
        .collect();
    // Cap the size: a 1.27 G-parameter tensor needs a 5 GB f32 reference on
    // the host, and this probe running out of memory looks exactly like a
    // broken kernel. Largest tensors UNDER the cap are the interesting ones.
    const MAX_PARAMS: usize = 128 << 20;
    picks.retain(|t| t.shape.iter().product::<usize>() <= MAX_PARAMS);
    picks.sort_by_key(|t| std::cmp::Reverse(t.shape.iter().product::<usize>()));
    picks.truncate(2);

    let last: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
    let hook = std::panic::take_hook();
    {
        let last = Arc::clone(&last);
        std::panic::set_hook(Box::new(move |info| {
            *last.lock().unwrap_or_else(|e| e.into_inner()) = info.to_string();
        }));
    }

    for entry in picks {
        let [k, n] = [entry.shape[0], entry.shape[1]];
        println!("\n{} [{k}, {n}]  role={:?}", entry.name, entry.role);

        // Reference: the pack's own f32 bytes, multiplied on the CPU.
        let f32s = pack.read_f32(entry).expect("read f32");
        let xs: Vec<f32> = (0..k).map(|i| ((i % 13) as f32 - 6.0) / 6.0).collect();
        let want = {
            let x = Tensor::<2>::from_data(TensorData::new(xs.clone(), [1, k]), &cpu);
            let w = Tensor::<2>::from_data(TensorData::new(f32s, [k, n]), &cpu);
            x.matmul(w)
                .into_data()
                .convert::<f32>()
                .try_to_vec::<f32>()
                .unwrap()
        };
        let scale = want.iter().map(|v| v.abs()).fold(1e-6, f32::max);

        for precision in [Precision::Q4, Precision::Q8] {
            for (path, deq) in [("native", false), ("dequantize", true)] {
                for round in 1..=ROUNDS {
                    let xs = xs.clone();
                    let gpu = gpu.clone();
                    let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let w = pack.tensor::<2>(entry, precision, &gpu).ok()?;
                        let w = if deq { w.dequantize() } else { w };
                        let x = Tensor::<2>::from_data(
                            TensorData::new(xs, [1, k]),
                            (&gpu, backend::float_dtype(&gpu)),
                        );
                        // Warm first (autotune), then time the second call.
                        let first = x
                            .clone()
                            .matmul(w.clone())
                            .into_data()
                            .convert::<f32>()
                            .try_to_vec::<f32>()
                            .ok()?;
                        let started = std::time::Instant::now();
                        let _ = x
                            .matmul(w)
                            .into_data()
                            .convert::<f32>()
                            .try_to_vec::<f32>()
                            .ok()?;
                        Some((first, started.elapsed().as_secs_f32() * 1000.0))
                    }))
                    .ok()
                    .flatten();
                    let verdict = match got {
                        None => {
                            let msg = last.lock().unwrap_or_else(|e| e.into_inner()).clone();
                            // Only the last line carries the reason; the rest is
                            // a file/line prefix that repeats.
                            let reason = msg.lines().last().unwrap_or("?").trim().to_string();
                            format!("PANIC: {}", &reason[..reason.len().min(110)])
                        }
                        Some((g, _)) if g.len() != want.len() => "WRONG SHAPE".to_string(),
                        Some((g, ms)) => {
                            let worst = g
                                .iter()
                                .zip(&want)
                                .map(|(a, b)| (a - b).abs() / scale)
                                .fold(0.0f32, f32::max);
                            format!(
                                "{} rel {worst:.4}, warm {ms:6.1} ms",
                                if worst > 0.5 { "GARBAGE" } else { "ok" }
                            )
                        }
                    };
                    println!("  {precision:?} {path:<10} round {round}/{ROUNDS}: {verdict}");
                }
            }
        }
    }
    std::panic::set_hook(hook);
}
