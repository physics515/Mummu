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
//! on the device (wrong on CUDA, 2026-10-10 — which is why the prefill path
//! has its own).
//!
//! Speed comes from a dependent chain (up, down, up, ...): a loop that
//! repeats one call lets fusion merge the reads, and it "measured" 5 TB/s on
//! a 672 GB/s card.

#![warn(clippy::pedantic, clippy::nursery, clippy::all)]

use burn::tensor::{Device, Tensor, TensorData};
use mummu::backend;
use mummu::nn::packed_gemv::{KBLOCK, kblock_scheme};
use mummu::nn::try_q4s_gemv;
use mummu::pack::{Precision, quantize_blocks, quantized_tensor_data};
use mummu::quant::{QuantPolicy, SchemeExt};
use mummu_num::{f32_from_u64, f32_from_usize};
use std::time::Instant;

/// `SplitMix64`.
struct Rng(u64);
impl Rng {
    const fn next_u64(&mut self) -> u64 {
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
    fn uniform_vec(&mut self, len: usize) -> Vec<f32> {
        (0..len).map(|_| self.uniform()).collect()
    }
}

fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    let peak = want.iter().fold(0f32, |acc, v| acc.max(v.abs())).max(1e-6);
    got.iter()
        .zip(want)
        .fold(0f32, |acc, (g, r)| acc.max((g - r).abs()))
        / peak
}

/// One weight in both forms, plus its exact values on the host.
struct Weight {
    ternary: Tensor<2>,
    q4: Tensor<2>,
    exact: Vec<f32>,
    q4_values: Vec<f32>,
}

/// A random ternary `[k_len, width]` weight: codes `−1/0/+1`, a scale per
/// 128 inputs of each column drawn from `scale`, uploaded natively and
/// through the pack's Q4 quantizer.
fn weight(
    rng: &mut Rng,
    k_len: usize,
    width: usize,
    scale: &mut dyn FnMut(&mut Rng) -> f32,
    gpu: &Device,
) -> Weight {
    let codes: Vec<i8> = (0..k_len * width)
        .map(|_| match rng.next_u64() % 3 {
            0 => -1,
            1 => 0,
            _ => 1,
        })
        .collect();
    let scales: Vec<f32> = (0..k_len / KBLOCK * width).map(|_| scale(rng)).collect();
    let exact: Vec<f32> = (0..k_len * width)
        .map(|i| f32::from(codes[i]) * scales[(i / width) / KBLOCK * width + i % width])
        .collect();
    let ternary = Tensor::<2>::from_data(
        quantized_tensor_data(&codes, &scales, [k_len, width], kblock_scheme()),
        gpu,
    );
    let (q, q_scales) = quantize_blocks(&exact, width, Precision::Q4);
    let q4_values = q
        .iter()
        .enumerate()
        .map(|(i, &v)| f32::from(v) * q_scales[i / 32])
        .collect();
    let scheme = QuantPolicy::Q4.scheme().expect("q4 scheme");
    let q4 = Tensor::<2>::from_data(
        quantized_tensor_data(&q, &q_scales, [k_len, width], scheme),
        gpu,
    );
    Weight {
        ternary,
        q4,
        exact,
        q4_values,
    }
}

/// `x · w` on the host, the reference.
fn host_matmul(x: &[f32], rows: usize, w: &[f32], k_len: usize, width: usize) -> Vec<f32> {
    let cpu = backend::cpu_device();
    Tensor::<2>::from_data(TensorData::new(x.to_vec(), [rows, k_len]), &cpu)
        .matmul(Tensor::<2>::from_data(
            TensorData::new(w.to_vec(), [k_len, width]),
            &cpu,
        ))
        .into_data()
        .try_to_vec::<f32>()
        .expect("f32")
}

fn device_gemv(x: &[f32], rows: usize, k_len: usize, w: &Tensor<2>, gpu: &Device) -> Vec<f32> {
    let input = Tensor::<2>::from_data(TensorData::new(x.to_vec(), [rows, k_len]), gpu);
    try_q4s_gemv(&input, w)
        .expect("packed path engages")
        .into_data()
        .try_to_vec::<f32>()
        .expect("f32")
}

/// A scale in `[0.002, 0.012)`.
fn small_scale(rng: &mut Rng) -> f32 {
    let jitter = 0.01 * rng.uniform().abs();
    jitter + 0.002
}

/// A scale near `1/sqrt(fan_in)`, so a chain neither explodes nor vanishes.
fn unit_scale(rng: &mut Rng, fan_in: usize) -> f32 {
    let jitter = 0.1 * rng.uniform();
    (jitter + 1.0) / f32_from_usize(fan_in).sqrt()
}

/// Parity of every path at one production shape.
fn parity(rng: &mut Rng, k_len: usize, width: usize, label: &str, gpu: &Device) {
    println!("[{label}] W [{k_len}, {width}]");
    let w = weight(rng, k_len, width, &mut small_scale, gpu);
    let row = rng.uniform_vec(k_len);
    let got = device_gemv(&row, 1, k_len, &w.ternary, gpu);
    let want = host_matmul(&row, 1, &w.exact, k_len, width);
    println!("  ternary decode parity: rel {:.2e}", rel_err(&got, &want));

    // Prefill: a 300-row span takes our own dequantize + f32 matmul.
    let rows = 300;
    let batch = rng.uniform_vec(rows * k_len);
    let got = device_gemv(&batch, rows, k_len, &w.ternary, gpu);
    let want = host_matmul(&batch, rows, &w.exact, k_len, width);
    println!(
        "  ternary prefill [{rows} rows] parity: rel {:.2e}",
        rel_err(&got, &want)
    );

    let got = device_gemv(&row, 1, k_len, &w.q4, gpu);
    let want = host_matmul(&row, 1, &w.q4_values, k_len, width);
    println!(
        "  q4 decode parity vs its own grid: rel {:.2e}",
        rel_err(&got, &want)
    );

    if std::env::var("PROBE_DEQUANT").is_ok() {
        let deq = w
            .ternary
            .dequantize()
            .into_data()
            .try_to_vec::<f32>()
            .expect("f32");
        println!(
            "  burn device dequantize vs exact: rel {:.2e}",
            rel_err(&deq, &w.exact)
        );
    }
}

/// Mean ms per GEMV of 100 `up` -> `down` steps from `x0`.
fn run_chain(x0: &Tensor<2>, up: &Tensor<2>, down: &Tensor<2>) -> f64 {
    let step = |x: Tensor<2>| try_q4s_gemv(&try_q4s_gemv(&x, up).expect("up"), down).expect("down");
    let _ = step(x0.clone()).into_data(); // warm
    let t0 = Instant::now();
    let mut x = x0.clone();
    for _ in 0..100 {
        x = step(x);
    }
    let finite = x
        .into_data()
        .try_to_vec::<f32>()
        .expect("f32")
        .iter()
        .all(|v| v.is_finite());
    assert!(finite, "the chain stays finite");
    t0.elapsed().as_secs_f64() * 1e3 / 200.0
}

/// The honest clock: an FFN decode chain, x -> up -> down -> up ... Every
/// call consumes the previous output, so nothing can be merged, skipped or
/// overlapped, and one readback at the end covers all. Interleaved and
/// repeated because the card is shared with production: best-of is the
/// kernel, the spread is the box.
fn chain(rng: &mut Rng, gpu: &Device) {
    let (hidden, inter) = (5120usize, 17408usize);
    println!("[chain] 100 x (up [{hidden}, {inter}] then down [{inter}, {hidden}])");
    let up = weight(rng, hidden, inter, &mut |r| unit_scale(r, hidden), gpu);
    let down = weight(rng, inter, hidden, &mut |r| unit_scale(r, inter), gpu);
    let x0 = Tensor::<2>::from_data(TensorData::new(rng.uniform_vec(hidden), [1, hidden]), gpu);
    let (mut tern, mut q4) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..5 {
        tern = tern.min(run_chain(&x0, &up.ternary, &down.ternary));
        q4 = q4.min(run_chain(&x0, &up.q4, &down.q4));
    }
    println!(
        "  best of 5: ternary {tern:.3} ms, q4 {q4:.3} ms per GEMV — ternary speedup over q4: {:.2}x",
        q4 / tern
    );
}

fn main() {
    #[cfg(feature = "cuda")]
    let gpu = backend::cuda_device();
    #[cfg(not(feature = "cuda"))]
    let gpu = backend::gpu_device();
    println!("device: {gpu:?}");
    let mut rng = Rng(42);
    // The 27B's projection shapes: FFN gate/up and down, attention q (gated,
    // 24 heads x 256 x 2), DeltaNet qkv.
    for (k_len, width, label) in [
        (5120usize, 17408usize, "ffn gate/up"),
        (17408, 5120, "ffn down"),
        (5120, 12288, "attn q+gate"),
        (5120, 10240, "deltanet qkv"),
    ] {
        parity(&mut rng, k_len, width, label, &gpu);
    }
    chain(&mut rng, &gpu);
}
