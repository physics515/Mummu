//! SPEC 1's acceptance instrument: measured effective bandwidth of the
//! packed-nibble VNNI GEMV against the DRAM roofline, at the 27B's
//! production shapes, next to the incumbent i8 flex path.
//!
//! Run on a QUIET machine (the repo's standing measurement rule) with
//!
//! ```text
//! cargo run --release -p mummu --example vnni-gemv-probe --no-default-features --features vulkan-spirv
//! ```
//!
//! Prints, per shape: ms/call, effective GB/s counted as the bytes each path
//! actually streams (packed nibbles + f16 scales for the VNNI path; i8 slab
//! + f32 scales for the incumbent), and the ratio. The DRAM roofline for the
//! ratio's denominator is whatever this box's memory system sustains —
//! measure it with a plain stream over the same buffer (printed first) so
//! the roofline fraction is honest for THIS machine, not a spec sheet.

use mummu::flex::kernels::{self, PackedQ4, Q8Acts};

fn wave(len: usize, f: f32) -> Vec<f32> {
    (0..len).map(|i| ((i as f32) * f).sin()).collect()
}

/// Best-of-N wall time for one closure, milliseconds.
fn time_ms(reps: usize, mut f: impl FnMut()) -> f64 {
    // One warm call outside the clock (page-in, rayon pool spin-up).
    f();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t0 = std::time::Instant::now();
        f();
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
    }
    best
}

fn main() {
    println!("vnni available: {}", kernels::vnni_available());
    println!("rayon threads:  {}", rayon::current_num_threads());

    // A DRAM read roofline for this box: sum a buffer far past every cache
    // (1 GiB), threaded, best of 3. This is the honest denominator.
    {
        let words = (1usize << 30) / 8;
        let buf: Vec<u64> = (0..words as u64).collect();
        let ms = time_ms(3, || {
            use rayon::prelude::*;
            let s: u64 = buf
                .par_chunks(1 << 16)
                .map(|c| c.iter().fold(0u64, |a, &b| a.wrapping_add(b)))
                .reduce(|| 0, u64::wrapping_add);
            std::hint::black_box(s);
        });
        let gbs = (words * 8) as f64 / (ms * 1e6);
        println!("dram read roofline (1 GiB threaded sum): {ms:.2} ms = {gbs:.1} GB/s\n");
    }

    // The 27B's production GEMV shapes: gate/up [5120, 17408], down
    // [17408, 5120], qkv mix [5120, 10240ish], o/out [6144, 5120].
    for (k, n, tag) in [
        (5120usize, 17408usize, "gate/up"),
        (17408, 5120, "down"),
        (5120, 6144, "qkv"),
        (6144, 5120, "out"),
    ] {
        let vals = wave(k * n, 0.000037);
        let w = PackedQ4::from_f32(&vals, k, n);
        let x = wave(k, 0.011);
        let acts = Q8Acts::quantize(&x);
        let mut out = vec![0.0f32; n];

        let ms_vnni = time_ms(20, || {
            kernels::gemv_q4n_vnni(&w, &acts, &mut out);
            std::hint::black_box(&out);
        });
        let bytes_vnni = w.streamed_bytes();
        let gbs_vnni = bytes_vnni as f64 / (ms_vnni * 1e6);

        // The incumbent's traffic model: 1 i8/elem + f32 scale per 32.
        let i8s: Vec<i8> = vals
            .iter()
            .map(|&v| (v * 7.0).round().clamp(-7.0, 7.0) as i8)
            .collect();
        let scales = vec![0.14f32; k * n / 32];
        let mut out2 = vec![0.0f32; n];
        let ms_i8 = time_ms(20, || {
            incumbent_i8(&i8s, &scales, &x, &mut out2, k, n);
            std::hint::black_box(&out2);
        });
        let bytes_i8 = k * n + (k * n / 32) * 4;
        let gbs_i8 = bytes_i8 as f64 / (ms_i8 * 1e6);

        // The DRAM regime: production cycles ~22 host layers (~3 GB packed)
        // per token, so no tensor stays L3-resident (128 MB on this part).
        // Round-robin over enough distinct copies to exceed L3 and the
        // warm-cache number above becomes the honest streaming number.
        let copies = (192 * 1024 * 1024 / bytes_vnni).max(2) + 1;
        let ws: Vec<PackedQ4> = (0..copies)
            .map(|c| {
                let v = wave(k * n, 0.000037 + c as f32 * 1e-6);
                PackedQ4::from_f32(&v, k, n)
            })
            .collect();
        let mut idx = 0usize;
        let ms_stream = time_ms(3 * copies, || {
            kernels::gemv_q4n_vnni(&ws[idx % copies], &acts, &mut out);
            idx += 1;
            std::hint::black_box(&out);
        });
        let gbs_stream = bytes_vnni as f64 / (ms_stream * 1e6);

        println!(
            "[{tag}] [{k} x {n}]  vnni warm {ms_vnni:.3} ms ({gbs_vnni:.1} GB/s of {:.1} MB)  |  \
			 vnni DRAM {ms_stream:.3} ms ({gbs_stream:.1} GB/s)  |  \
			 i8 {ms_i8:.3} ms ({gbs_i8:.1} GB/s of {:.1} MB)  |  speedup {:.2}x (DRAM {:.2}x)",
            bytes_vnni as f64 / 1e6,
            bytes_i8 as f64 / 1e6,
            ms_i8 / ms_vnni,
            ms_i8 / ms_stream,
        );
    }
}

/// The incumbent flex inner loop, verbatim shape (rayon over 32-aligned
/// output chunks, k-outer, LLVM-autovectorized i8 widen + FMA) so the
/// comparison is against what production runs today, not a strawman.
fn incumbent_i8(wq: &[i8], scales: &[f32], xs: &[f32], out: &mut [f32], k_len: usize, n: usize) {
    use rayon::prelude::*;
    let blocks = n / 32;
    let chunk_blocks = blocks.div_ceil(rayon::current_num_threads().max(1)).max(4);
    out.iter_mut().for_each(|o| *o = 0.0);
    out.par_chunks_mut(chunk_blocks * 32)
        .enumerate()
        .for_each(|(t, chunk)| {
            let b0 = t * chunk_blocks;
            let nb = chunk.len() / 32;
            for k in 0..k_len {
                let xk = xs[k];
                let row = &wq[k * n + b0 * 32..k * n + b0 * 32 + nb * 32];
                let srow = &scales[k * blocks + b0..k * blocks + b0 + nb];
                for b in 0..nb {
                    let xs_s = xk * srow[b];
                    let src = &row[b * 32..b * 32 + 32];
                    let dst = &mut chunk[b * 32..b * 32 + 32];
                    for j in 0..32 {
                        dst[j] += xs_s * f32::from(src[j]);
                    }
                }
            }
        });
}
