//! Measures the two host kernels this branch adds, at 27B-shaped
//! dimensions, against the costs they replace:
//!
//! 1. `flex::gdn::gdn_step` — the fused GDN middle. The tensor path's
//!    middle measured ~10 ms/layer/token in the production flamegraph
//!    (host GDN mixers 229 ms over 23 host layers, minus the projections'
//!    ~1.5 ms); the fused function's floor is two passes over the per-head
//!    state (~3 MB/layer at these shapes) plus one activation sweep.
//! 2. `flex::kernels::gemm_q4n_auto` — the packed GEMM at prompt shapes,
//!    against the row-looped GEMV it replaces (m separate weight streams).
//!
//! ```text
//! cargo run --release -p mummu --example fused-middle-probe --no-default-features --features vulkan-spirv
//! ```
//!
//! Quiet-box rules apply; read ratios, not absolutes.

use mummu::flex::gdn::{GdnMiddle, gdn_step};
use mummu::flex::kernels::{PackedQ4, gemm_q4n_auto, gemv_q4n_auto};

fn wave(len: usize, f: f32) -> Vec<f32> {
    (0..len).map(|i| ((i as f32) * f).sin()).collect()
}

fn median_ms(reps: usize, mut f: impl FnMut()) -> f64 {
    f(); // warm
    let mut v: Vec<f64> = (0..reps)
        .map(|_| {
            let t0 = std::time::Instant::now();
            f();
            t0.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    // 27B-shaped GDN dims (hidden 5120; qkv mix ~10 k wide; out 6144):
    // hk = 16 k-heads, hv = 48 v-heads, ds = 128, conv kernel 4.
    let (hk, hv, ds, kk) = (16usize, 48usize, 128usize, 4usize);
    let key_dim = hk * ds;
    let d_inner = hv * ds;
    let conv_dim = 2 * key_dim + d_inner;
    let p = GdnMiddle {
        hk,
        hv,
        ds,
        kk,
        conv_dim,
        key_dim,
        d_inner,
        l2_eps: 1e-6,
        norm_eps: 1e-6,
        scale: 1.0 / (ds as f32).sqrt(),
        conv_w: wave(conv_dim * kk, 0.13),
        dt_bias: wave(hv, 0.7),
        a: wave(hv, 0.3).iter().map(|v| -v.abs() - 0.1).collect(),
        gamma: wave(ds, 0.11),
    };
    let mixed = wave(conv_dim, 0.017);
    let z = wave(d_inner, 0.023);
    let beta = wave(hv, 0.5);
    let alpha = wave(hv, 0.9);
    let mut ring = vec![0.0f32; p.ring_len()];
    let mut state = wave(p.state_len(), 0.007);
    let mut gated = vec![0.0f32; d_inner];
    let ms = median_ms(200, || {
        gdn_step(
            &p, &mixed, &z, &beta, &alpha, &mut ring, &mut state, &mut gated,
        );
        std::hint::black_box(&gated);
    });
    let state_mb = (p.state_len() * 4) as f64 / 1e6;
    println!(
        "fused GDN middle [hk {hk} hv {hv} ds {ds} conv_dim {conv_dim}]: {ms:.3} ms/layer/token \
         (state {state_mb:.1} MB, two passes = {:.1} MB moved)",
        2.0 * state_mb
    );
    println!(
        "  vs the tensor middle's ~10 ms/layer in the production flamegraph -> ~{:.0}x on the middle\n",
        10.0 / ms
    );

    // The packed GEMM at prompt shapes vs the row loop it replaced.
    let (k, n) = (5120usize, 17408usize);
    let w = PackedQ4::from_f32(&wave(k * n, 0.11), k, n);
    for m in [4usize, 16, 36, 64, 256] {
        let x = wave(m * k, 0.009);
        let mut out = vec![0.0f32; m * n];
        let gemm = median_ms(9, || {
            gemm_q4n_auto(&w, &x, m, &mut out);
            std::hint::black_box(&out);
        });
        let mut row_out = vec![0.0f32; n];
        let rows = median_ms(3, || {
            for r in 0..m {
                gemv_q4n_auto(&w, &x[r * k..(r + 1) * k], &mut row_out);
            }
            std::hint::black_box(&row_out);
        });
        println!(
            "gemm [{k} x {n}] m={m:>3}: {gemm:>8.2} ms vs row-loop {rows:>8.2} ms = {:>5.1}x \
             ({:.0} tok/s-equivalent per layer)",
            rows / gemm,
            1e3 / (gemm / m as f64),
        );
    }
}
