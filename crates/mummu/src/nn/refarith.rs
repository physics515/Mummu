//! Reference-arithmetic emulation: a DIAGNOSTIC that makes our forward
//! compute the way llama.cpp's CPU backend does.
//!
//! The point is that a parity investigation can tell a port bug from the
//! reference's own rounding noise. Off unless `MUMMU_REF_ARITH=1` or
//! [`set_enabled`]; every hook is a no-op when off.
//!
//! Why it exists: llama.cpp's CPU `mul_mat` on a quantized weight first
//! quantizes the ACTIVATION to the weight's `vec_dot_type` (`Q8_0` for `Q8_0`,
//! `Q8_K` for `Q4_K/Q5_K/Q6_K`, `Q8_1` for `Q5_1`) and dots integer blocks, and its
//! flash attention reads an f16 KV cache with Q converted to f16 and the value
//! sum accumulated in f16. That noise is chaotic across 48 qwen4exp layers:
//! llama.cpp's own mathematically equivalent settings (`--no-repack`,
//! `-fa off`) move the first-forward tail logprobs by up to ~1 nat. With this
//! emulation plus [`set_perturbation`] our forward samples the same noise
//! process, so its spread can be compared with the reference's.
//!
//! What is emulated (checked against ggml at llama.cpp 930e2fa59):
//! - `Q8_0` activations (`quantize_row_q8_0`, AVX): `d = amax/127` stored f16,
//!   `q = round_nearest(x·127/amax)`; the dot uses the f16 `d`.
//! - `Q8_K` activations (`quantize_row_q8_K_ref`, also the repacked 4x8 grid):
//!   signed-max scale `-127/max`, `q = min(127, nearest(iscale·x))`.
//! - `Q8_1` (`Q5_1` weights) as `Q8_0`, plus [`q8_1_min_term_rounding`]: ggml
//!   stores `s = d·Σq` (with the f32 `d`) as f16 and dots the weight block's
//!   min against that rounded `s`. Left out, it moved Flash-Next's
//!   Q5_1-down expert outputs by ~5e-4 relative against a teacher-forced
//!   llama.cpp dump (the Q8_0-down layers matched to 2e-6).
//! - Flash attention: `ggml_compute_forward_flash_attn_ext_f16_one_chunk`
//!   (the path for fewer than 64 query rows and fewer than 512 cached cells).
//!
//! Not emulated: ggml's SIMD summation order (bit-exactness is out of reach
//! and unnecessary for distributions), the fused flex GDN decode step and the
//! packed decode GEMV (prefill-shaped forwards only).

use std::collections::HashMap;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use burn::tensor::{Tensor, TensorData};
use mummu_num::{f64_from_u64, narrow};

use crate::gguf::GgmlType;

/// 0 = read the env on first use, 1 = on, 2 = off.
static STATE: AtomicU8 = AtomicU8::new(0);

/// Whether the emulation is on (`MUMMU_REF_ARITH=1` or [`set_enabled`]).
pub fn enabled() -> bool {
    match STATE.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let on = std::env::var("MUMMU_REF_ARITH").is_ok_and(|v| v == "1");
            STATE.store(if on { 1 } else { 2 }, Ordering::Relaxed);
            on
        }
    }
}

/// Turn the emulation on or off for every later forward in this PROCESS
/// (global: never toggle it from a unit test that shares a process with
/// other model tests).
pub fn set_enabled(on: bool) {
    STATE.store(if on { 1 } else { 2 }, Ordering::Relaxed);
}

/// Seed + 1 of the embedding perturbation (0 = none).
static SEED: AtomicU64 = AtomicU64::new(0);
/// Relative perturbation size, as f64 bits.
static EPS: AtomicU64 = AtomicU64::new(0);

/// Scale every later embedding by `1 + eps·N(0,1)` drawn from `seed`
/// (`None` = exact embeddings).
///
/// With the emulation on, a relative `eps` of
/// 1e-5 flips enough activation-rounding decisions within a few layers to
/// give an independent noise realization; with it off the logits move by
/// less than 1e-3 nats.
pub fn set_perturbation(seed: Option<u64>, eps: f64) {
    SEED.store(seed.map_or(0, |s| s + 1), Ordering::Relaxed);
    EPS.store(eps.to_bits(), Ordering::Relaxed);
}

/// Whether an embedding perturbation is currently configured. A parity gate
/// asserts this is false: a verdict measured on a perturbed forward is not a
/// verdict about the port.
#[must_use]
pub fn perturbation_active() -> bool {
    SEED.load(Ordering::Relaxed) != 0
}

/// Apply the configured embedding perturbation (identity when none is set).
pub fn perturb_embedding(x: Tensor<3>) -> Tensor<3> {
    let s = SEED.load(Ordering::Relaxed);
    if s == 0 {
        return x;
    }
    let eps = f64::from_bits(EPS.load(Ordering::Relaxed));
    let dims = x.dims();
    let device = x.device();
    let dt = x.dtype();
    let mut v = host(x.into_data());
    perturb_in_place(&mut v, s, eps);
    Tensor::from_data(TensorData::new(v, dims), (&device, dt))
}

/// `SplitMix64` normals (Box-Muller): deterministic per seed, no dependency.
fn perturb_in_place(v: &mut [f32], seed: u64, eps: f64) {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut uniform = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (f64_from_u64(z >> 11) + 0.5) / f64_from_u64(1u64 << 53)
    };
    for x in v.iter_mut() {
        let (u1, u2) = (uniform(), uniform());
        let g = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        // Two roundings on purpose: the perturbation is a reproducible
        // noise process, not an accuracy target.
        let rel = eps * g;
        *x = narrow(f64::from(*x) * (1.0 + rel));
    }
}

fn host(d: TensorData) -> Vec<f32> {
    d.convert::<f32>()
        .try_into_vec::<f32>()
        .expect("refarith readback")
}

/// `(in, out)` -> the file dtype of the linear with that shape, or `None`
/// when two linears of that shape disagree (then neither is emulated).
type ShapeRegistry = HashMap<(usize, usize), Option<GgmlType>>;

static SHAPES: RwLock<Option<ShapeRegistry>> = RwLock::new(None);

/// Record that a linear `inp -> out` is stored as `dtype` in the file. The
/// matmul hooks only see shapes, so a shape registered with two dtypes is
/// marked ambiguous (and warned about) instead of guessing.
///
/// # Panics
/// If the shape registry's lock is poisoned — a previous registration
/// panicked while holding it.
pub fn register_linear(inp: usize, out: usize, dtype: GgmlType) {
    note_dtype(
        SHAPES
            .write()
            .expect("refarith registry")
            .get_or_insert_with(HashMap::new),
        inp,
        out,
        dtype,
    );
}

/// [`register_linear`]'s body against an unlocked registry.
fn note_dtype(m: &mut ShapeRegistry, inp: usize, out: usize, dtype: GgmlType) {
    match m.get(&(inp, out)) {
        None => {
            m.insert((inp, out), Some(dtype));
        }
        Some(Some(prev)) if *prev != dtype => {
            eprintln!(
                "[refarith] linears {inp}->{out} are stored as both {prev:?} and {dtype:?}; \
                 that shape is not emulated"
            );
            m.insert((inp, out), None);
        }
        Some(_) => {}
    }
}

fn dtype_of(inp: usize, out: usize) -> Option<GgmlType> {
    SHAPES
        .read()
        .expect("refarith registry")
        .as_ref()
        .and_then(|m| m.get(&(inp, out)).copied().flatten())
}

fn f16r(v: f32) -> f32 {
    half::f16::from_f32(v).to_f32()
}

/// `Q8_0` (and `Q8_1`) activation blocks as ggml's AVX `quantize_row_q8_0`
/// produces them, dequantized with the f16 scale the dot product uses.
fn fq_q8_0(x: &mut [f32]) {
    for b in x.chunks_mut(32) {
        let amax = b.iter().fold(0f32, |a, v| a.max(v.abs()));
        let id = if amax == 0.0 { 0.0 } else { 127.0 / amax };
        let df = f16r(amax / 127.0);
        for v in b.iter_mut() {
            *v = (*v * id).round_ties_even() * df;
        }
    }
}

/// `Q8_K` activation blocks (`quantize_row_q8_K_ref`): the scale comes from the
/// signed value of largest magnitude, so the grid is `amax/127` with the
/// positive end clipped at 127.
fn fq_q8_k(x: &mut [f32]) {
    for b in x.chunks_mut(256) {
        let (mut amax, mut max) = (0f32, 0f32);
        for &v in b.iter() {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        if amax == 0.0 {
            b.fill(0.0);
            continue;
        }
        let iscale = -127.0 / max;
        let d = 1.0 / iscale;
        for v in b.iter_mut() {
            *v = (iscale * *v).round_ties_even().min(127.0) * d;
        }
    }
}

/// The part of ggml's `vec_dot_q5_1_q8_1` that fake-quantized activations
/// miss.
///
/// Per row of `xs` (rows of `width`, a multiple of 32) and per 32-block,
/// `f16(d·Σq) − f16(d)·Σq`, where `d = amax/127` and `q` is the block's `Q8_1`
/// quantization. A `Q5_1` weight block with min `m` adds `m` times this to
/// its dot product; the caller owns the mins. Row-major `[rows · width/32]`.
#[must_use]
pub fn q8_1_min_term_rounding(xs: &[f32], width: usize) -> Vec<f32> {
    xs.chunks(width)
        .flat_map(|row| row.chunks(32))
        .map(|b| {
            let amax = b.iter().fold(0f32, |a, v| a.max(v.abs()));
            let d = amax / 127.0;
            let id = if amax == 0.0 { 0.0 } else { 127.0 / amax };
            let sum_q: f32 = b.iter().map(|v| (*v * id).round_ties_even()).sum();
            // Two roundings on purpose: the term IS the difference between
            // two separately rounded products.
            let fake_sum = f16r(d) * sum_q;
            f16r(d * sum_q) - fake_sum
        })
        .collect()
}

/// Fake-quantize host activations `xs` (rows of `width`) the way llama.cpp
/// quantizes them before a matmul with a weight stored as `dtype`.
///
/// Other dtypes (F32/F16/BF16 weights) leave `xs` untouched. Rows are
/// independent.
pub fn fake_quant_rows(xs: &mut [f32], width: usize, dtype: GgmlType) {
    match dtype {
        GgmlType::Q80 | GgmlType::Q51 | GgmlType::Q40 | GgmlType::Q50 | GgmlType::Q41 => {
            xs.chunks_mut(width).for_each(fq_q8_0);
        }
        GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::Q3K | GgmlType::Q2K => {
            xs.chunks_mut(width).for_each(fq_q8_k);
        }
        _ => {}
    }
}

/// The activation a registered quantized linear `d_in -> d_out` sees
/// (`x` unchanged when the emulation is off or the shape is unregistered).
#[must_use]
pub fn linear_input2(x: Tensor<2>, d_in: usize, d_out: usize) -> Tensor<2> {
    if !enabled() {
        return x;
    }
    let Some(dtype) = dtype_of(d_in, d_out) else {
        return x;
    };
    let dims = x.dims();
    let device = x.device();
    let dt = x.dtype();
    let mut v = host(x.into_data());
    fake_quant_rows(&mut v, d_in, dtype);
    Tensor::from_data(TensorData::new(v, dims), (&device, dt))
}

/// Causal attention as ggml's CPU flash attention computes it over an f16
/// KV cache.
///
/// `q [b, nh, t, hd]` (post-RoPE), `k`/`v [b, nkv, T, hd]` holding
/// every cached position with the new rows last, query head `h` reading kv
/// head `h / (nh / nkv)`. Returns `[b, nh, t, hd]`.
#[must_use]
pub fn flash_attn_f16(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, scale: f32) -> Tensor<4> {
    let device = q.device();
    let dt = q.dtype();
    let [batch, nh, rows, hd] = q.dims();
    let [_, nkv, tt, _] = k.dims();
    let out = flash_attn_f16_host(
        &host(q.into_data()),
        &host(k.into_data()),
        &host(v.into_data()),
        [batch, nh, rows, hd],
        [nkv, tt],
        scale,
    );
    Tensor::from_data(TensorData::new(out, [batch, nh, rows, hd]), (&device, dt))
}

/// Host body of [`flash_attn_f16`]. The arithmetic follows ggml's
/// `flash_attn_ext_f16_one_chunk` line by line (the one `mul_add` is ggml's
/// own fused f16 value accumulate); everything else is two roundings on
/// purpose.
fn flash_attn_f16_host(
    qv: &[f32],
    kv: &[f32],
    vv: &[f32],
    [batch, nh, rows, hd]: [usize; 4],
    [nkv, tt]: [usize; 2],
    scale: f32,
) -> Vec<f32> {
    use half::f16;
    let past = tt - rows;
    let group = nh / nkv;
    // The cache stores K and V as f16; Q is converted by `q_to_vec_dot`.
    let kf: Vec<f16> = kv.iter().map(|&x| f16::from_f32(x)).collect();
    let vf: Vec<f16> = vv.iter().map(|&x| f16::from_f32(x)).collect();
    let mut out = vec![0f32; batch * nh * rows * hd];
    let mut acc = vec![f16::ZERO; hd]; // VKQ16
    let mut qq = vec![f16::ZERO; hd];
    for bi in 0..batch {
        for head in 0..nh {
            let kh = head / group;
            for row in 0..rows {
                let qo = ((bi * nh + head) * rows + row) * hd;
                for (dst, src) in qq.iter_mut().zip(&qv[qo..qo + hd]) {
                    *dst = f16::from_f32(*src);
                }
                acc.fill(f16::ZERO);
                // Online softmax: M is the running max score, S the sum of
                // exp(s - M), and acc the value sum in f16.
                let (mut s_sum, mut run_max) = (0f32, f32::NEG_INFINITY);
                for pos in 0..=past + row {
                    let ko = ((bi * nkv + kh) * tt + pos) * hd;
                    let dot: f64 = (0..hd)
                        .map(|dim| f64::from(kf[ko + dim].to_f32() * qq[dim].to_f32()))
                        .sum();
                    let score = narrow(dot) * scale;
                    let (mut ms, mut vs) = (1f32, 1f32);
                    if score > run_max {
                        ms = (run_max - score).exp();
                        run_max = score;
                        for slot in &mut acc {
                            *slot = f16::from_f32(slot.to_f32() * ms);
                        }
                    } else {
                        vs = (score - run_max).exp();
                    }
                    for dim in 0..hd {
                        let prev = acc[dim].to_f32();
                        acc[dim] = f16::from_f32(vf[ko + dim].to_f32().mul_add(vs, prev));
                    }
                    let rescaled = s_sum * ms;
                    s_sum = rescaled + vs;
                }
                let inv = if s_sum == 0.0 { 0.0 } else { 1.0 / s_sum };
                for dim in 0..hd {
                    out[qo + dim] = acc[dim].to_f32() * inv;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use mummu_num::{f32_from_i32, f32_from_usize, f64_from_usize};

    #[test]
    fn q8_0_blocks_land_on_the_f16_scaled_127_grid() {
        let mut x: Vec<f32> = (0..64)
            .map(|i| (f32_from_i32(i) * 0.37).sin() * 3.0)
            .collect();
        x[5] = -4.0; // block 0's amax
        let orig = x.clone();
        fq_q8_0(&mut x);
        for (blk, (qb, ob)) in x.chunks(32).zip(orig.chunks(32)).enumerate() {
            let amax = ob.iter().fold(0f32, |a, v| a.max(v.abs()));
            let d = f16r(amax / 127.0);
            let half_step = 0.5 * amax / 127.0;
            let slack = 1e-3 * amax;
            for (q, o) in qb.iter().zip(ob) {
                let steps = q / d;
                assert!(
                    (steps - steps.round()).abs() < 1e-3 && steps.abs() <= 127.5,
                    "block {blk}: {q} is not on the {d} grid"
                );
                assert!((q - o).abs() <= half_step + slack, "block {blk}");
            }
        }
        let full_scale = 127.0 * f16r(4.0 / 127.0);
        assert!((x[5] + full_scale).abs() < 1e-6, "amax maps to -127·d");
    }

    /// ggml's `Q8_1` `s` is `f16(d·Σq)` with the f32 `d`, while fake-quantized
    /// activations sum to `f16(d)·Σq`; the returned term is exactly that
    /// difference, per 32-block, recomputed here from the textbook grid.
    #[test]
    fn the_q8_1_min_term_is_the_f16_rounding_of_the_block_sum() {
        let x: Vec<f32> = (0..64)
            .map(|i| {
                let wave = (f32_from_i32(i) * 0.61).cos() * 0.0137;
                wave + 0.004
            })
            .collect();
        let got = q8_1_min_term_rounding(&x, 64);
        assert_eq!(got.len(), 2);
        for (b, block) in x.chunks(32).enumerate() {
            let amax = block.iter().fold(0f32, |a, v| a.max(v.abs()));
            let sum_q: f32 = block.iter().map(|v| (v * 127.0 / amax).round()).sum();
            let d = amax / 127.0;
            let fake_sum = half::f16::from_f32(d).to_f32() * sum_q;
            let want = half::f16::from_f32(d * sum_q).to_f32() - fake_sum;
            assert!(sum_q != 0.0, "the block must have a nonzero quant sum");
            assert!(
                (got[b] - want).abs() <= 1e-9,
                "block {b}: {} vs {want}",
                got[b]
            );
        }
        assert!(
            got.iter().any(|g| g.abs() > 0.0),
            "these blocks' sums are not f16-exact, so the term must not vanish"
        );
        assert_eq!(q8_1_min_term_rounding(&[0.0; 32], 32), vec![0.0]);
    }

    #[test]
    fn q8_k_uses_one_signed_max_scale_per_256_block() {
        let mut x: Vec<f32> = (0..512).map(|i| (f32_from_i32(i) * 0.11).cos()).collect();
        x[7] = 5.0; // positive max: iscale is negative, the grid is 5/127
        let orig = x.clone();
        fq_q8_k(&mut x);
        let d = 5.0f32 / 127.0;
        let half_step = 0.5 * d;
        for (q, o) in x[..256].iter().zip(&orig[..256]) {
            let steps = q / d;
            assert!((steps - steps.round()).abs() < 1e-3, "{q} off the {d} grid");
            assert!((q - o).abs() <= half_step + 1e-5);
        }
        assert!((x[7] - 5.0).abs() < 1e-5, "the max element is exact");
        // Block 1 has its own scale (amax ~1).
        let amax1 = orig[256..].iter().fold(0f32, |a, v| a.max(v.abs()));
        assert!(
            x[256..]
                .iter()
                .zip(&orig[256..])
                .all(|(q, o)| (q - o).abs() <= 0.5 * amax1 / 127.0 + 1e-5)
        );
    }

    #[test]
    fn f32_and_f16_weights_leave_activations_exact() {
        let x: Vec<f32> = (0..64).map(|i| f32_from_i32(i) * 0.013).collect();
        for dt in [GgmlType::F32, GgmlType::F16, GgmlType::BF16] {
            let mut y = x.clone();
            fake_quant_rows(&mut y, 64, dt);
            assert_eq!(x, y, "{dt:?}");
        }
    }

    /// The online f16 softmax equals exact causal attention up to f16
    /// rounding, and the new rows only read positions up to their own.
    #[test]
    fn f16_flash_attention_matches_exact_causal_attention_within_f16_rounding() {
        let (batch, nh, nkv, hd, rows, past) = (1usize, 4usize, 2usize, 8usize, 3usize, 2usize);
        let tt = past + rows;
        let wave = |len: usize, phase: f32| -> Vec<f32> {
            (0..len)
                .map(|i| {
                    let arg = f32_from_usize(i) * 0.7;
                    (arg + phase).sin() * 0.9
                })
                .collect()
        };
        let (q, k, v) = (
            wave(batch * nh * rows * hd, 0.1),
            wave(batch * nkv * tt * hd, 1.3),
            wave(batch * nkv * tt * hd, 2.9),
        );
        let scale = 1.0 / f32_from_usize(hd).sqrt();
        let got = flash_attn_f16_host(&q, &k, &v, [batch, nh, rows, hd], [nkv, tt], scale);
        for head in 0..nh {
            let kh = head / (nh / nkv);
            for row in 0..rows {
                let qo = (head * rows + row) * hd;
                let scores: Vec<f64> = (0..=past + row)
                    .map(|pos| {
                        let ko = (kh * tt + pos) * hd;
                        (0..hd)
                            .map(|dim| f64::from(q[qo + dim] * k[ko + dim]))
                            .sum::<f64>()
                            * f64::from(scale)
                    })
                    .collect();
                let mx = scores.iter().copied().fold(f64::MIN, f64::max);
                let denom: f64 = scores.iter().map(|score| (score - mx).exp()).sum();
                for dim in 0..hd {
                    let want: f64 = scores
                        .iter()
                        .enumerate()
                        .map(|(pos, score)| {
                            (score - mx).exp() / denom * f64::from(v[(kh * tt + pos) * hd + dim])
                        })
                        .sum();
                    let got_dim = f64::from(got[qo + dim]);
                    assert!(
                        (got_dim - want).abs() < 2e-3,
                        "head {head} row {row} dim {dim}: {got_dim} vs {want}"
                    );
                }
            }
        }
        // Causality: a change to the last cached row leaves row 0 untouched.
        let mut k2 = k;
        for x in &mut k2[(tt - 1) * hd..tt * hd] {
            *x += 0.5;
        }
        let got2 = flash_attn_f16_host(&q, &k2, &v, [batch, nh, rows, hd], [nkv, tt], scale);
        assert_eq!(got[..hd], got2[..hd], "row 0 read a future position");
    }

    #[test]
    fn the_perturbation_is_deterministic_per_seed_and_relative() {
        let x: Vec<f32> = (1..=1000).map(f32_from_i32).collect();
        let (mut a, mut b, mut c) = (x.clone(), x.clone(), x.clone());
        perturb_in_place(&mut a, 3, 1e-3);
        perturb_in_place(&mut b, 3, 1e-3);
        perturb_in_place(&mut c, 4, 1e-3);
        assert_eq!(a, b, "same seed, same draw");
        assert_ne!(a, c, "another seed, another draw");
        let rel: Vec<f64> = a
            .iter()
            .zip(&x)
            .map(|(p, o)| f64::from(p / o) - 1.0)
            .collect();
        let rms = (rel.iter().map(|r| r * r).sum::<f64>() / f64_from_usize(rel.len())).sqrt();
        assert!((0.8e-3..1.2e-3).contains(&rms), "relative rms {rms}");
    }

    #[test]
    fn a_shape_stored_with_two_dtypes_is_not_emulated() {
        // Shapes no real model registers, so the global registry is safe to share.
        register_linear(7_777_001, 3, GgmlType::Q80);
        assert_eq!(dtype_of(7_777_001, 3), Some(GgmlType::Q80));
        register_linear(7_777_001, 3, GgmlType::Q80);
        assert_eq!(dtype_of(7_777_001, 3), Some(GgmlType::Q80));
        register_linear(7_777_001, 3, GgmlType::Q4K);
        assert_eq!(dtype_of(7_777_001, 3), None);
    }
}
