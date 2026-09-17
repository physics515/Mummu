//! Reference-arithmetic emulation: a DIAGNOSTIC that makes our forward
//! compute the way llama.cpp's CPU backend does, so a parity investigation
//! can tell a port bug from the reference's own rounding noise. Off unless
//! `MUMMU_REF_ARITH=1` or [`set_enabled`]; every hook is a no-op when off.
//!
//! Why it exists: llama.cpp's CPU `mul_mat` on a quantized weight first
//! quantizes the ACTIVATION to the weight's `vec_dot_type` (Q8_0 for Q8_0,
//! Q8_K for Q4_K/Q5_K/Q6_K, Q8_1 for Q5_1) and dots integer blocks, and its
//! flash attention reads an f16 KV cache with Q converted to f16 and the value
//! sum accumulated in f16. That noise is chaotic across 48 qwen4exp layers:
//! llama.cpp's own mathematically equivalent settings (`--no-repack`,
//! `-fa off`) move the first-forward tail logprobs by up to ~1 nat. With this
//! emulation plus [`set_perturbation`] our forward samples the same noise
//! process, so its spread can be compared with the reference's.
//!
//! What is emulated (checked against ggml at llama.cpp 930e2fa59):
//! - Q8_0 activations (`quantize_row_q8_0`, AVX): `d = amax/127` stored f16,
//!   `q = round_nearest(x·127/amax)`; the dot uses the f16 `d`.
//! - Q8_K activations (`quantize_row_q8_K_ref`, also the repacked 4x8 grid):
//!   signed-max scale `-127/max`, `q = min(127, nearest(iscale·x))`.
//! - Q8_1 (Q5_1 weights) as Q8_0, plus [`q8_1_min_term_rounding`]: ggml
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
/// (`None` = exact embeddings). With the emulation on, a relative `eps` of
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

/// SplitMix64 normals (Box-Muller): deterministic per seed, no dependency.
fn perturb_in_place(v: &mut [f32], seed: u64, eps: f64) {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut uniform = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    for x in v.iter_mut() {
        let (u1, u2) = (uniform(), uniform());
        let g = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        *x = (f64::from(*x) * (1.0 + eps * g)) as f32;
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
pub fn register_linear(inp: usize, out: usize, dtype: GgmlType) {
    let mut g = SHAPES.write().expect("refarith registry");
    let m = g.get_or_insert_with(HashMap::new);
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

/// Q8_0 (and Q8_1) activation blocks as ggml's AVX `quantize_row_q8_0`
/// produces them, dequantized with the f16 scale the dot product uses.
fn fq_q8_0(x: &mut [f32]) {
    for b in x.chunks_mut(32) {
        let amax = b.iter().fold(0f32, |a, v| a.max(v.abs()));
        let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
        let df = f16r(amax / 127.0);
        for v in b.iter_mut() {
            *v = (*v * id).round_ties_even() * df;
        }
    }
}

/// Q8_K activation blocks (`quantize_row_q8_K_ref`): the scale comes from the
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
/// miss: per row of `xs` (rows of `width`, a multiple of 32) and per 32-block,
/// `f16(d·Σq) − f16(d)·Σq`, where `d = amax/127` and `q` is the block's Q8_1
/// quantization. A Q5_1 weight block with min `m` adds `m` times this to
/// its dot product; the caller owns the mins. Row-major `[rows · width/32]`.
#[must_use]
pub fn q8_1_min_term_rounding(xs: &[f32], width: usize) -> Vec<f32> {
    xs.chunks(width)
        .flat_map(|row| row.chunks(32))
        .map(|b| {
            let amax = b.iter().fold(0f32, |a, v| a.max(v.abs()));
            let d = amax / 127.0;
            let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
            let sum_q: f32 = b.iter().map(|v| (*v * id).round_ties_even()).sum();
            f16r(d * sum_q) - f16r(d) * sum_q
        })
        .collect()
}

/// Fake-quantize host activations `xs` (rows of `width`) the way llama.cpp
/// quantizes them before a matmul with a weight stored as `dtype`; other
/// dtypes (F32/F16/BF16 weights) leave `xs` untouched. Rows are independent.
pub fn fake_quant_rows(xs: &mut [f32], width: usize, dtype: GgmlType) {
    match dtype {
        GgmlType::Q8_0 | GgmlType::Q5_1 | GgmlType::Q4_0 | GgmlType::Q5_0 | GgmlType::Q4_1 => {
            xs.chunks_mut(width).for_each(fq_q8_0);
        }
        GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K | GgmlType::Q3_K | GgmlType::Q2_K => {
            xs.chunks_mut(width).for_each(fq_q8_k);
        }
        _ => {}
    }
}

/// The activation a registered quantized linear `d_in -> d_out` sees
/// (`x` unchanged when the emulation is off or the shape is unregistered).
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
/// KV cache: `q [b, nh, t, hd]` (post-RoPE), `k`/`v [b, nkv, T, hd]` holding
/// every cached position with the new rows last, query head `h` reading kv
/// head `h / (nh / nkv)`. Returns `[b, nh, t, hd]`.
pub fn flash_attn_f16(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, scale: f32) -> Tensor<4> {
    let device = q.device();
    let dt = q.dtype();
    let [b, nh, t, hd] = q.dims();
    let [_, nkv, tt, _] = k.dims();
    let out = flash_attn_f16_host(
        &host(q.into_data()),
        &host(k.into_data()),
        &host(v.into_data()),
        [b, nh, t, hd],
        [nkv, tt],
        scale,
    );
    Tensor::from_data(TensorData::new(out, [b, nh, t, hd]), (&device, dt))
}

/// Host body of [`flash_attn_f16`].
fn flash_attn_f16_host(
    qv: &[f32],
    kv: &[f32],
    vv: &[f32],
    [b, nh, t, hd]: [usize; 4],
    [nkv, tt]: [usize; 2],
    scale: f32,
) -> Vec<f32> {
    use half::f16;
    let past = tt - t;
    let group = nh / nkv;
    // The cache stores K and V as f16; Q is converted by `q_to_vec_dot`.
    let kf: Vec<f16> = kv.iter().map(|&x| f16::from_f32(x)).collect();
    let vf: Vec<f16> = vv.iter().map(|&x| f16::from_f32(x)).collect();
    let mut out = vec![0f32; b * nh * t * hd];
    let mut acc = vec![f16::ZERO; hd]; // VKQ16
    let mut qq = vec![f16::ZERO; hd];
    for bi in 0..b {
        for h in 0..nh {
            let kh = h / group;
            for i in 0..t {
                let qo = ((bi * nh + h) * t + i) * hd;
                for (d, x) in qq.iter_mut().zip(&qv[qo..qo + hd]) {
                    *d = f16::from_f32(*x);
                }
                acc.fill(f16::ZERO);
                // Online softmax: M is the running max score, S the sum of
                // exp(s - M), and acc the value sum in f16.
                let (mut s_sum, mut m) = (0f32, f32::NEG_INFINITY);
                for j in 0..=past + i {
                    let ko = ((bi * nkv + kh) * tt + j) * hd;
                    let dot: f64 = (0..hd)
                        .map(|d| f64::from(kf[ko + d].to_f32() * qq[d].to_f32()))
                        .sum();
                    let s = dot as f32 * scale;
                    let (mut ms, mut vs) = (1f32, 1f32);
                    if s > m {
                        ms = (m - s).exp();
                        m = s;
                        for a in acc.iter_mut() {
                            *a = f16::from_f32(a.to_f32() * ms);
                        }
                    } else {
                        vs = (s - m).exp();
                    }
                    for d in 0..hd {
                        let y = acc[d].to_f32();
                        acc[d] = f16::from_f32(vf[ko + d].to_f32().mul_add(vs, y));
                    }
                    s_sum = s_sum * ms + vs;
                }
                let inv = if s_sum == 0.0 { 0.0 } else { 1.0 / s_sum };
                for d in 0..hd {
                    out[qo + d] = acc[d].to_f32() * inv;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_blocks_land_on_the_f16_scaled_127_grid() {
        let mut x: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.37).sin() * 3.0).collect();
        x[5] = -4.0; // block 0's amax
        let orig = x.clone();
        fq_q8_0(&mut x);
        for (blk, (qb, ob)) in x.chunks(32).zip(orig.chunks(32)).enumerate() {
            let amax = ob.iter().fold(0f32, |a, v| a.max(v.abs()));
            let d = f16r(amax / 127.0);
            for (q, o) in qb.iter().zip(ob) {
                let steps = q / d;
                assert!(
                    (steps - steps.round()).abs() < 1e-3 && steps.abs() <= 127.5,
                    "block {blk}: {q} is not on the {d} grid"
                );
                assert!(
                    (q - o).abs() <= 0.5 * amax / 127.0 + 1e-3 * amax,
                    "block {blk}"
                );
            }
        }
        assert!(
            (x[5] + 127.0 * f16r(4.0 / 127.0)).abs() < 1e-6,
            "amax maps to -127·d"
        );
    }

    /// ggml's Q8_1 `s` is `f16(d·Σq)` with the f32 `d`, while fake-quantized
    /// activations sum to `f16(d)·Σq`; the returned term is exactly that
    /// difference, per 32-block, recomputed here from the textbook grid.
    #[test]
    fn the_q8_1_min_term_is_the_f16_rounding_of_the_block_sum() {
        let x: Vec<f32> = (0..64)
            .map(|i| ((i as f32) * 0.61).cos() * 0.0137 + 0.004)
            .collect();
        let got = q8_1_min_term_rounding(&x, 64);
        assert_eq!(got.len(), 2);
        for (b, block) in x.chunks(32).enumerate() {
            let amax = block.iter().fold(0f32, |a, v| a.max(v.abs()));
            let sum_q: f32 = block.iter().map(|v| (v * 127.0 / amax).round()).sum();
            let d = amax / 127.0;
            let want =
                half::f16::from_f32(d * sum_q).to_f32() - half::f16::from_f32(d).to_f32() * sum_q;
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
        let mut x: Vec<f32> = (0..512).map(|i| ((i as f32) * 0.11).cos()).collect();
        x[7] = 5.0; // positive max: iscale is negative, the grid is 5/127
        let orig = x.clone();
        fq_q8_k(&mut x);
        let d = 5.0f32 / 127.0;
        for (q, o) in x[..256].iter().zip(&orig[..256]) {
            let steps = q / d;
            assert!((steps - steps.round()).abs() < 1e-3, "{q} off the {d} grid");
            assert!((q - o).abs() <= 0.5 * d + 1e-5);
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
        let x: Vec<f32> = (0..64).map(|i| i as f32 * 0.013).collect();
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
        let (b, nh, nkv, hd, t, past) = (1usize, 4usize, 2usize, 8usize, 3usize, 2usize);
        let tt = past + t;
        let r = |n: usize, s: f32| -> Vec<f32> {
            (0..n).map(|i| ((i as f32 * 0.7 + s).sin()) * 0.9).collect()
        };
        let (q, k, v) = (
            r(b * nh * t * hd, 0.1),
            r(b * nkv * tt * hd, 1.3),
            r(b * nkv * tt * hd, 2.9),
        );
        let scale = 1.0 / (hd as f32).sqrt();
        let got = flash_attn_f16_host(&q, &k, &v, [b, nh, t, hd], [nkv, tt], scale);
        for h in 0..nh {
            let kh = h / (nh / nkv);
            for i in 0..t {
                let qo = (h * t + i) * hd;
                let scores: Vec<f64> = (0..=past + i)
                    .map(|j| {
                        let ko = (kh * tt + j) * hd;
                        (0..hd)
                            .map(|d| f64::from(q[qo + d] * k[ko + d]))
                            .sum::<f64>()
                            * f64::from(scale)
                    })
                    .collect();
                let mx = scores.iter().copied().fold(f64::MIN, f64::max);
                let z: f64 = scores.iter().map(|s| (s - mx).exp()).sum();
                for d in 0..hd {
                    let want: f64 = scores
                        .iter()
                        .enumerate()
                        .map(|(j, s)| (s - mx).exp() / z * f64::from(v[(kh * tt + j) * hd + d]))
                        .sum();
                    let g = f64::from(got[qo + d]);
                    assert!(
                        (g - want).abs() < 2e-3,
                        "head {h} row {i} dim {d}: {g} vs {want}"
                    );
                }
            }
        }
        // Causality: a change to the last cached row leaves row 0 untouched.
        let mut k2 = k.clone();
        for x in &mut k2[(tt - 1) * hd..tt * hd] {
            *x += 0.5;
        }
        let got2 = flash_attn_f16_host(&q, &k2, &v, [b, nh, t, hd], [nkv, tt], scale);
        assert_eq!(got[..hd], got2[..hd], "row 0 read a future position");
    }

    #[test]
    fn the_perturbation_is_deterministic_per_seed_and_relative() {
        let x: Vec<f32> = (1..=1000).map(|i| i as f32).collect();
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
        let rms = (rel.iter().map(|r| r * r).sum::<f64>() / rel.len() as f64).sqrt();
        assert!((0.8e-3..1.2e-3).contains(&rms), "relative rms {rms}");
    }

    #[test]
    fn a_shape_stored_with_two_dtypes_is_not_emulated() {
        // Shapes no real model registers, so the global registry is safe to share.
        register_linear(7_777_001, 3, GgmlType::Q8_0);
        assert_eq!(dtype_of(7_777_001, 3), Some(GgmlType::Q8_0));
        register_linear(7_777_001, 3, GgmlType::Q8_0);
        assert_eq!(dtype_of(7_777_001, 3), Some(GgmlType::Q8_0));
        register_linear(7_777_001, 3, GgmlType::Q4_K);
        assert_eq!(dtype_of(7_777_001, 3), None);
    }
}
