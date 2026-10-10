//! Parity + speed of the native ternary GEMV (`Q2S`, `[128, 1]` blocks along
//! the input axis — `nn::packed_gemv::is_kblock`) against today's Q4S
//! block-32 packed GEMV, at the 27B's production shapes, on the GPU.
//!
//! Both weights take the production path: built on the host, uploaded as
//! packed bytes (`Tensor::from_data` of quantized `TensorData`) — never
//! quantized on the device, which measures the probe and OOMs.
//!
//! `cargo run --release -p mummu --features cuda --example ternary-gemv-probe`
//! (CUDA when built with `cuda`, else the default wgpu device).
//! `PROBE_DEQUANT=1` also checks burn's own dequantize of the ternary layout
//! on the device — what a prefill fallback would lean on.

#![warn(clippy::pedantic, clippy::nursery, clippy::all)]

use burn::tensor::{Tensor, TensorData};
use mummu::backend;
use mummu::nn::packed_gemv::{KBLOCK, kblock_scheme};
use mummu::nn::try_q4s_gemv;
use mummu::pack::{Precision, quantize_blocks, quantized_tensor_data};
use mummu::quant::{QuantPolicy, SchemeExt};
use mummu_num::{f32_from_u64, f64_from_usize};
use std::time::Instant;

/// `SplitMix64`.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `[-1, 1)`.
    fn uniform(&mut self) -> f32 {
        f32_from_u64(self.next_u64() >> 40) / f32_from_u64(1u64 << 23) - 1.0
    }
}

/// Mean ms per call, each call read back before the next is queued — the
/// readback (~tens of µs) is included on purpose: a queued-only loop let the
/// first Q4 run report 5 TB/s on a ~670 GB/s card, which is not a GEMV.
fn bench(label: &str, iters: usize, f: &mut dyn FnMut() -> Tensor<2>) -> f64 {
    for _ in 0..3 {
        let _ = f().into_data(); // warm (compile, autotune, pools)
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = f().into_data();
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / f64_from_usize(iters);
    println!("  {label}: {ms:.3} ms/call (synced)");
    ms
}

fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    let peak = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
    got.iter()
        .zip(want)
        .fold(0f32, |a, (g, w)| a.max((g - w).abs()))
        / peak
}

fn main() {
    #[cfg(feature = "cuda")]
    let gpu = backend::cuda_device();
    #[cfg(not(feature = "cuda"))]
    let gpu = backend::gpu_device();
    let cpu = backend::cpu_device();
    println!("device: {gpu:?}");
    let mut rng = Rng(42);
    // The 27B's projection shapes: FFN gate/up and down, attention q (gated,
    // 24 heads x 256 x 2), DeltaNet qkv.
    for (k, n, label) in [
        (5120usize, 17408usize, "ffn gate/up"),
        (17408, 5120, "ffn down"),
        (5120, 12288, "attn q+gate"),
        (5120, 10240, "deltanet qkv"),
    ] {
        println!("[{label}] W [{k}, {n}], x [1, {k}]");
        let codes: Vec<i8> = (0..k * n)
            .map(|_| match rng.next_u64() % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            })
            .collect();
        let scales: Vec<f32> = (0..k / KBLOCK * n)
            .map(|_| 0.002 + 0.01 * rng.uniform().abs())
            .collect();
        let exact: Vec<f32> = (0..k * n)
            .map(|i| f32::from(codes[i]) * scales[(i / n) / KBLOCK * n + i % n])
            .collect();
        let xv: Vec<f32> = (0..k).map(|_| rng.uniform()).collect();

        let t = Tensor::<2>::from_data(
            quantized_tensor_data(&codes, &scales, [k, n], kblock_scheme()),
            &gpu,
        );
        drop(codes);
        let x = Tensor::<2>::from_data(TensorData::new(xv.clone(), [1, k]), &gpu);

        // Reference on the host from the exact values (f32 matmul, flex).
        let want = Tensor::<2>::from_data(TensorData::new(xv.clone(), [1, k]), &cpu)
            .matmul(Tensor::<2>::from_data(
                TensorData::new(exact.clone(), [k, n]),
                &cpu,
            ))
            .into_data()
            .try_to_vec::<f32>()
            .expect("f32");
        let got = try_q4s_gemv(&x, &t)
            .expect("ternary path must engage")
            .into_data()
            .try_to_vec::<f32>()
            .expect("f32");
        println!("  ternary parity vs exact reference: rel {:.2e}", rel_err(&got, &want));

        // Prefill: a 300-row span takes our own dequantize + f32 matmul
        // (burn's dequantize is wrong for these blocks — see below).
        let m = 300;
        let xm: Vec<f32> = (0..m * k).map(|_| rng.uniform()).collect();
        let want_m = Tensor::<2>::from_data(TensorData::new(xm.clone(), [m, k]), &cpu)
            .matmul(Tensor::<2>::from_data(TensorData::new(exact.clone(), [k, n]), &cpu))
            .into_data()
            .try_to_vec::<f32>()
            .expect("f32");
        let got_m = try_q4s_gemv(&Tensor::<2>::from_data(TensorData::new(xm, [m, k]), &gpu), &t)
            .expect("ternary prefill path")
            .into_data()
            .try_to_vec::<f32>()
            .expect("f32");
        println!("  ternary prefill [{m} rows] parity: rel {:.2e}", rel_err(&got_m, &want_m));

        if std::env::var("PROBE_DEQUANT").is_ok() {
            let deq = t
                .clone()
                .dequantize()
                .into_data()
                .try_to_vec::<f32>()
                .expect("f32");
            println!("  burn device dequantize vs exact: rel {:.2e}", rel_err(&deq, &exact));
        }

        // Today's level: the same values through the pack's Q4S block-32
        // quantizer, uploaded as packed bytes exactly as `Pack::tensor` does.
        let (q, s) = quantize_blocks(&exact, n, Precision::Q4);
        let scheme = QuantPolicy::Q4.scheme().expect("q4 scheme");
        let q4 = Tensor::<2>::from_data(quantized_tensor_data(&q, &s, [k, n], scheme), &gpu);
        let q4_vals: Vec<f32> = q
            .iter()
            .enumerate()
            .map(|(i, &v)| f32::from(v) * s[i / 32])
            .collect();
        let want_q4 = Tensor::<2>::from_data(TensorData::new(xv.clone(), [1, k]), &cpu)
            .matmul(Tensor::<2>::from_data(TensorData::new(q4_vals, [k, n]), &cpu))
            .into_data()
            .try_to_vec::<f32>()
            .expect("f32");
        let got_q4 = try_q4s_gemv(&x, &q4)
            .expect("q4")
            .into_data()
            .try_to_vec::<f32>()
            .expect("f32");
        println!("  q4 parity vs its own dequant: rel {:.2e}", rel_err(&got_q4, &want_q4));

        let tern = bench("ternary kblock gemv", 50, &mut || {
            try_q4s_gemv(&x, &t).expect("ternary")
        });
        let q4ms = bench("q4s block-32 gemv ", 50, &mut || {
            try_q4s_gemv(&x, &q4).expect("q4")
        });
        let bytes_t = f64_from_usize(k * n / 4 + k / KBLOCK * n * 4);
        let bytes_q4 = f64_from_usize(k * n / 2 + k * n / 32 * 4);
        println!(
            "  ternary {:.1} GB/s ({:.1} MB), q4 {:.1} GB/s ({:.1} MB), speedup {:.2}x",
            bytes_t / tern / 1e6,
            bytes_t / 1e6,
            bytes_q4 / q4ms / 1e6,
            bytes_q4 / 1e6,
            q4ms / tern
        );
    }

    // The honest clock: an FFN decode chain, x -> W_up -> W_down -> W_up ...
    // Every call consumes the previous output, so nothing can be skipped,
    // deduplicated or overlapped, and one readback at the end covers all.
    println!("[chain] 100 x (up [5120, 17408] then down [17408, 5120])");
    let (h, f) = (5120usize, 17408usize);
    let mut make = |k: usize, n: usize| {
        let codes: Vec<i8> = (0..k * n)
            .map(|_| match rng.next_u64() % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            })
            .collect();
        // Scales sized so the chain neither explodes nor vanishes.
        let scales: Vec<f32> = (0..k / KBLOCK * n)
            .map(|_| (1.0 + 0.1 * rng.uniform()) / (f32::from(u16::try_from(k).expect("k fits")).sqrt()))
            .collect();
        let exact: Vec<f32> = (0..k * n)
            .map(|i| f32::from(codes[i]) * scales[(i / n) / KBLOCK * n + i % n])
            .collect();
        let tern = Tensor::<2>::from_data(
            quantized_tensor_data(&codes, &scales, [k, n], kblock_scheme()),
            &gpu,
        );
        let (q, sc) = quantize_blocks(&exact, n, Precision::Q4);
        let scheme = QuantPolicy::Q4.scheme().expect("q4 scheme");
        let q4 = Tensor::<2>::from_data(quantized_tensor_data(&q, &sc, [k, n], scheme), &gpu);
        (tern, q4)
    };
    let (up_t, up_q4) = make(h, f);
    let (down_t, down_q4) = make(f, h);
    let x0 = Tensor::<2>::from_data(
        TensorData::new((0..h).map(|_| rng.uniform()).collect::<Vec<f32>>(), [1, h]),
        &gpu,
    );
    let chain = |label: &str, up: &Tensor<2>, down: &Tensor<2>| {
        let step = |x: Tensor<2>| try_q4s_gemv(&try_q4s_gemv(&x, up).expect("up"), down).expect("down");
        let _ = step(x0.clone()).into_data(); // warm
        let t0 = Instant::now();
        let mut x = x0.clone();
        for _ in 0..100 {
            x = step(x);
        }
        let v = x.into_data().try_to_vec::<f32>().expect("f32");
        let ms = t0.elapsed().as_secs_f64() * 1e3 / 200.0;
        println!("  {label}: {ms:.3} ms per GEMV (finite: {})", v.iter().all(|a| a.is_finite()));
        ms
    };
    // Interleaved and repeated: the card is shared with production, so one
    // sample of each is noise. Best-of is the kernel; the spread is the box.
    let (mut t, mut q) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..5 {
        t = t.min(chain("ternary", &up_t, &down_t));
        q = q.min(chain("q4     ", &up_q4, &down_q4));
    }
    println!("  best of 5: ternary {t:.3} ms, q4 {q:.3} ms — ternary speedup over q4: {:.2}x", q / t);
}
