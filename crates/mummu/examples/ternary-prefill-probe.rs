//! Where a 1024-row ternary prefill projection spends its time on the
//! card: the whole `try_q4s_gemv` path against a bare matmul on an
//! already-dequantized weight, in f32 and f16.

#![warn(clippy::pedantic, clippy::nursery, clippy::all)]

use burn::tensor::{Tensor, TensorData};
use mummu::backend;
use mummu::nn::packed_gemv::{KBLOCK, kblock_scheme};
use mummu::nn::try_q4s_gemv;
use mummu::pack::quantized_tensor_data;
use mummu_num::f32_from_usize;
use std::time::Instant;

/// Waits for the result: reads one element back (a drop of the lazy handle times nothing).
fn touch(t: &Tensor<2>) {
    let [r, c] = t.dims();
    let _ = t
        .clone()
        .slice([r - 1..r, c - 1..c])
        .into_data()
        .try_to_vec::<f32>()
        .expect("f32");
}

fn timed(label: &str, flops: f64, mut run: impl FnMut() -> Tensor<2>) {
    touch(&run());
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t0 = Instant::now();
        touch(&run());
        best = best.min(t0.elapsed().as_secs_f64());
    }
    println!(
        "  {label}: {:.1} ms, {:.1} TFLOPS",
        best * 1e3,
        flops / best / 1e12
    );
}

fn main() {
    #[cfg(feature = "cuda")]
    let gpu = backend::cuda_device();
    #[cfg(not(feature = "cuda"))]
    let gpu = backend::gpu_device();
    let (rows, k_len, width) = (1024usize, 5120usize, 17408usize);
    let flops = 2.0
        * f64::from(f32_from_usize(rows))
        * f64::from(f32_from_usize(k_len))
        * f64::from(f32_from_usize(width));
    println!("device {gpu:?}; x [{rows}, {k_len}] . w [{k_len}, {width}]");
    let codes: Vec<i8> = (0..k_len * width).map(|i| [-1i8, 0, 1][i % 3]).collect();
    let scales: Vec<f32> = (0..k_len / KBLOCK * width)
        .map(|i| {
            let t = f32_from_usize(i % 7) * 1e-3;
            0.01 + t
        })
        .collect();
    let w = Tensor::<2>::from_data(
        quantized_tensor_data(&codes, &scales, [k_len, width], kblock_scheme()),
        &gpu,
    );
    let xv: Vec<f32> = (0..rows * k_len)
        .map(|i| {
            let t = f32_from_usize(i % 13) * 0.1;
            t - 0.6
        })
        .collect();
    let x = Tensor::<2>::from_data(TensorData::new(xv, [rows, k_len]), &gpu);
    for m in [8usize, 32, 128, 256, 512, 1024] {
        let xm = x.clone().slice([0..m, 0..k_len]);
        let fl = flops * f64::from(f32_from_usize(m)) / f64::from(f32_from_usize(rows));
        timed(&format!("try_q4s_gemv m={m}"), fl, || {
            try_q4s_gemv(&xm, &w).expect("engages")
        });
    }
    let xm = x.clone().slice([0..37, 0..k_len]);
    let got = try_q4s_gemv(&xm, &w).expect("engages").into_data();
    let got = got.try_to_vec::<f32>().expect("f32");
    let mut worst = 0.0f32;
    for r in [0usize, 1, 5, 36] {
        let one = try_q4s_gemv(&x.clone().slice([r..r + 1, 0..k_len]), &w)
            .expect("engages")
            .into_data()
            .try_to_vec::<f32>()
            .expect("f32");
        for (c, v) in one.iter().enumerate() {
            let d = (v - got[r * width + c]).abs() / v.abs().max(1.0);
            worst = worst.max(d);
        }
    }
    println!("  multi-row vs single-row, worst rel diff {worst:.2e}");
}
