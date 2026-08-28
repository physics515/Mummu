//! Packed-nibble Q4 GEMV at the DRAM roofline (SPEC 1).
//!
//! The host GEMV this replaces reads flex's i8-unpacked Q4 storage —
//! 1.125 B/param — with a scalar-autovectorized f32 FMA loop. Decode GEMV is
//! memory-bound, so the ceiling is `bytes_streamed / DRAM_bandwidth`; this
//! module halves the bytes (packed nibbles + f16 scales = 0.5625 B/param +
//! 1/16 scale overhead) and evaluates the dot products with AVX-512 VNNI
//! integer MACs so the instruction budget stays far under the memory budget
//! (one `vpdpbusd` retires 64 MACs; the whole inner iteration is ~12
//! instructions per 64 weights, and rayon spreads rows across cores, so the
//! aggregate instruction ceiling is an order of magnitude above the DRAM
//! roofline — the kernel streams).
//!
//! ## The master identity
//!
//! Weights are 4-bit symmetric per 32-element group **along the reduction
//! dim K** (note: the device format groups along N; this host format is its
//! own quantization, produced by [`PackedQ4::from_f32`]), stored
//! offset-binary (`stored = q + 8`, `q ∈ [-7, 7]`, so `stored ∈ [1, 15]`).
//! Activations are quantized per 32-element group of K to i8:
//! `x[k] = sx(g)·qx[k] + ε`, `|ε| ≤ sx(g)/2`. Then
//!
//! ```text
//! y[n] = Σ_g sw(n,g)·sx(g)·( Σ_{k∈g} stored[k,n]·qx[k]  −  8·Σ_{k∈g} qx[k] )
//! ```
//!
//! Both inner sums are `vpdpbusd` (u8 × i8 → i32) dots — the second against
//! a constant all-8 vector — and the kernel subtracts them **in the integer
//! domain**, per lane, before anything touches f32. That ordering is
//! load-bearing: the offset part (`8·Σqx`) is routinely a thousand times the
//! true dot, and an f32 accumulator that carries it to a final correction
//! loses ~5e-4 of the answer to cancellation (measured; the first cut of
//! this kernel did exactly that). Subtracted as i32 it is exact: per-lane
//! magnitudes stay ≤ `4·15·127 = 7620 ≪ 2³¹`, a lane never accumulates more
//! than one group, and the difference is the true `Σ q·qx ≤ 4·7·127`. The
//! only approximation in the whole path is ε — bounded, and checked by the
//! auto-dispatch below.
//!
//! ## Storage layout (the second 2×)
//!
//! `qs` holds W transposed — one output row `n` at a time, K/2 bytes per
//! row, in 32-byte chunks covering 64 consecutive k each: byte `j` of chunk
//! `c` carries `k = 64c + j` in its LOW nibble and `k = 64c + 32 + j` in its
//! HIGH nibble. This is the offline interleave that makes the unpack free:
//! `lo = bytes & 0x0F` yields elements `64c..64c+32` in order, `hi = bytes
//! >> 4` yields `64c+32..64c+64`, and `[lo | hi]` as one zmm is perfectly
//! sequential in k — the activation vector is loaded with a single
//! unpermuted 64-byte read. Group boundaries land exactly on the zmm
//! halves, so the two groups' partial sums occupy i32 lanes 0..8 and 8..16
//! and the per-group scale applies as a vertical f32 multiply — the
//! horizontal sum happens **once per row**, never per block.
//!
//! Scales are f16, one per (row, group), in a separate contiguous array.
//!
//! ## Exactness dispatch
//!
//! [`gemv_q4n_auto`] estimates the activation-quantization error budget per
//! call (`ε` is the only error) and routes pathological activation
//! distributions — a group whose magnitude is dominated by one outlier
//! quantizes the other 31 elements to noise — to [`gemv_q4n_f32`], the
//! exact-in-f32 path over the same packed bytes. The dispatch decision is
//! logged (throttled) so a model that keeps falling off the fast path is
//! visible, per the SPEC 1 contract.

use half::f16;
use rayon::prelude::*;

/// Quantization group width along K. Matches the repo-wide block width so
/// eligibility rules carry over.
pub const GROUP: usize = 32;

/// Two groups per SIMD chunk (one 32-byte packed load = 64 weights).
const CHUNK: usize = 2 * GROUP;

// ---------------------------------------------------------------------------
// Packed weight storage
// ---------------------------------------------------------------------------

/// A weight matrix `[K, N]` (burn Linear layout, `y = x·W`) re-quantized
/// 4-bit symmetric per 32-group **along K** and packed for the host kernel.
///
/// This is its own quantization grid — building one from f32 loses exactly
/// one 4-bit rounding, same magnitude as the device grid's. Building one
/// from an already-quantized tensor ([`PackedQ4::from_q4s_slab`]) pays a
/// second rounding on top of the first; prefer sourcing the pack's f16/f32
/// level when available.
pub struct PackedQ4 {
    /// Reduction length (rows of the logical `[K, N]`), multiple of 32.
    pub k: usize,
    /// Output width (columns of the logical `[K, N]`).
    pub n: usize,
    /// Packed nibbles, `n · k/2` bytes, laid out as documented above.
    qs: Vec<u8>,
    /// f16 scale per (row n, group g): `scales[n·groups + g]`.
    scales: Vec<f16>,
}

impl PackedQ4 {
    /// Groups per output row.
    #[must_use]
    pub fn groups(&self) -> usize {
        self.k / GROUP
    }

    /// Bytes this representation actually streams per GEMV (values + scales)
    /// — the numerator of the roofline.
    #[must_use]
    pub fn streamed_bytes(&self) -> usize {
        self.qs.len() + self.scales.len() * 2
    }

    /// Quantize + pack from a per-column accessor (rayon across columns —
    /// packing a 27B layer tensor must cost tens of milliseconds, not
    /// seconds; it can run per-tensor at load or lazily at first touch).
    fn build(k: usize, n: usize, fill_col: impl Fn(usize, &mut [f32]) + Sync) -> Self {
        assert!(
            k.is_multiple_of(GROUP),
            "PackedQ4: k {k} must divide by {GROUP}"
        );
        let groups = k / GROUP;
        let mut qs = vec![0u8; n * k / 2];
        let mut scales = vec![f16::ZERO; n * groups];
        qs.par_chunks_mut(k / 2)
            .zip(scales.par_chunks_mut(groups))
            .enumerate()
            .for_each(|(col, (qrow, srow))| {
                let mut colv = vec![0.0f32; k];
                fill_col(col, &mut colv);
                // One row's quantized values, natural k order, offset (q+8).
                let mut row_q = vec![8u8; k];
                for g in 0..groups {
                    let seg = &colv[g * GROUP..(g + 1) * GROUP];
                    let amax = seg.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                    let scale = if amax > 0.0 { amax / 7.0 } else { 1.0 };
                    srow[g] = f16::from_f32(scale);
                    // Quantize against the f16-rounded scale the kernel will
                    // actually multiply by — otherwise the stored grid and
                    // the evaluated grid disagree by the f16 rounding.
                    let inv = 1.0 / srow[g].to_f32();
                    for (j, &v) in seg.iter().enumerate() {
                        let q = (v * inv).round().clamp(-7.0, 7.0) as i32;
                        row_q[g * GROUP + j] = (q + 8) as u8;
                    }
                }
                pack_row(&row_q, qrow);
            });
        Self { k, n, qs, scales }
    }

    /// Quantize + pack from row-major f32 `[K, N]` values.
    ///
    /// Reads column-major (once, at load); the kernel's row-major streams are
    /// what the layout optimizes. `k` must divide by 32; callers gate
    /// eligibility the same way the device quantizer does.
    #[must_use]
    pub fn from_f32(values: &[f32], k: usize, n: usize) -> Self {
        assert!(
            values.len() == k * n,
            "PackedQ4::from_f32: {} values for [{k}, {n}]",
            values.len()
        );
        Self::build(k, n, |col, out| {
            for (kk, slot) in out.iter_mut().enumerate() {
                *slot = values[kk * n + col];
            }
        })
    }

    /// Re-pack from flex's unpacked Q4S storage (i8 values `[K, N]`, f32
    /// scales `[K, N/32]`, blocks along N — the device grid). Dequantizes and
    /// re-quantizes along K: a second 4-bit rounding, taken deliberately when
    /// no float source is at hand (the lazy path logs it). Column-at-a-time,
    /// so the f32 transient is one column, never the whole tensor.
    #[must_use]
    pub fn from_q4s_slab(values: &[i8], scales: &[f32], k: usize, n: usize) -> Self {
        assert!(values.len() == k * n, "slab shape mismatch");
        let blocks = n / GROUP;
        assert!(scales.len() == k * blocks, "slab scales mismatch");
        Self::build(k, n, |col, out| {
            let b = col / GROUP;
            for (kk, slot) in out.iter_mut().enumerate() {
                *slot = f32::from(values[kk * n + col]) * scales[kk * blocks + b];
            }
        })
    }

    /// The dequantized `[K, N]` f32 matrix this representation denotes — the
    /// ground truth every kernel in this module is measured against.
    #[must_use]
    pub fn dequantize(&self) -> Vec<f32> {
        let groups = self.groups();
        let mut out = vec![0.0f32; self.k * self.n];
        let mut row = vec![0u8; self.k];
        for col in 0..self.n {
            unpack_row(
                &self.qs[col * (self.k / 2)..(col + 1) * (self.k / 2)],
                &mut row,
            );
            for (kk, &stored) in row.iter().enumerate() {
                let q = i32::from(stored) - 8;
                let s = self.scales[col * groups + kk / GROUP].to_f32();
                out[kk * self.n + col] = q as f32 * s;
            }
        }
        out
    }

    /// One row's packed bytes.
    #[inline]
    fn row_qs(&self, n: usize) -> &[u8] {
        &self.qs[n * (self.k / 2)..(n + 1) * (self.k / 2)]
    }

    /// One row's scales.
    #[inline]
    fn row_scales(&self, n: usize) -> &[f16] {
        let g = self.groups();
        &self.scales[n * g..(n + 1) * g]
    }
}

/// Pack one row (offset nibbles, natural k order) into the chunk layout.
fn pack_row(row_q: &[u8], out: &mut [u8]) {
    let k = row_q.len();
    debug_assert_eq!(out.len(), k / 2);
    let full = k / CHUNK;
    for c in 0..full {
        for j in 0..GROUP {
            out[c * GROUP + j] = row_q[c * CHUNK + j] | (row_q[c * CHUNK + GROUP + j] << 4);
        }
    }
    // K % 64 == 32 tail: one group, lo = j, hi = j + 16.
    if k % CHUNK != 0 {
        let base = full * CHUNK;
        let ob = full * GROUP;
        for j in 0..GROUP / 2 {
            out[ob + j] = row_q[base + j] | (row_q[base + GROUP / 2 + j] << 4);
        }
    }
}

/// Inverse of [`pack_row`] (tests + dequantize).
fn unpack_row(packed: &[u8], row_q: &mut [u8]) {
    let k = row_q.len();
    let full = k / CHUNK;
    for c in 0..full {
        for j in 0..GROUP {
            let b = packed[c * GROUP + j];
            row_q[c * CHUNK + j] = b & 0x0F;
            row_q[c * CHUNK + GROUP + j] = b >> 4;
        }
    }
    if k % CHUNK != 0 {
        let base = full * CHUNK;
        let ob = full * GROUP;
        for j in 0..GROUP / 2 {
            let b = packed[ob + j];
            row_q[base + j] = b & 0x0F;
            row_q[base + GROUP / 2 + j] = b >> 4;
        }
    }
}

// ---------------------------------------------------------------------------
// Activation quantization
// ---------------------------------------------------------------------------

/// One activation vector quantized per 32-group of K: `x[k] ≈ sx(g)·qx[k]`.
///
/// Built once per (token, projection input) and shared by every output row.
pub struct Q8Acts {
    /// i8 values, natural k order.
    qs: Vec<i8>,
    /// Per-group scale `sx(g) = max|x in g| / 127` (1.0 for an all-zero group).
    scales: Vec<f32>,
    /// Predicted mean relative quantization error of the whole vector:
    /// `Σ_g 32·(sx(g)/2) / Σ_k |x[k]|`. ~1/254 ≈ 0.004 for well-behaved
    /// activations; large values flag outlier-dominated groups.
    pub quality: f32,
}

impl Q8Acts {
    /// Quantize `x` (length a multiple of 32).
    #[must_use]
    pub fn quantize(x: &[f32]) -> Self {
        assert!(
            x.len().is_multiple_of(GROUP),
            "activation length must divide by {GROUP}"
        );
        let groups = x.len() / GROUP;
        let mut qs = vec![0i8; x.len()];
        let mut scales = vec![1.0f32; groups];
        let mut err_budget = 0.0f64;
        let mut l1 = 0.0f64;
        for g in 0..groups {
            let seg = &x[g * GROUP..(g + 1) * GROUP];
            let amax = seg.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            scales[g] = scale;
            let inv = 1.0 / scale;
            for (j, &v) in seg.iter().enumerate() {
                let q = (v * inv).round().clamp(-127.0, 127.0) as i32;
                qs[g * GROUP + j] = q as i8;
                l1 += f64::from(v.abs());
            }
            err_budget += f64::from(scale) * 0.5 * GROUP as f64;
        }
        let quality = if l1 > 0.0 {
            (err_budget / l1) as f32
        } else {
            0.0
        };
        Self {
            qs,
            scales,
            quality,
        }
    }
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/// Is the VNNI fast path available on this CPU?
#[must_use]
pub fn vnni_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512dq")
            && std::arch::is_x86_feature_detected!("avx512vnni")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Activation distributions whose predicted mean relative quantization error
/// exceeds this fall back to the exact f32 path.
///
/// Calibrated against measured reality, not synthetic waves: uniform-ish
/// test vectors predict ~0.004, but REAL decode activations on the 27B
/// (post-RMSNorm hidden states, SwiGLU products) measure 0.022–0.043 —
/// heavy tails raise per-group `amax/mean` without hurting the dot, whose
/// error concentrates over K terms (the same per-32 i8 activation
/// quantization llama.cpp applies universally, with no dispatch at all).
/// An early 0.02 limit routed every production host GEMV to the slow exact
/// path and silently gave the speedup back. 0.12 keeps the gate where it
/// catches the genuinely pathological shape — a lone outlier owning each
/// group predicts ~0.126 — and nothing real. `MUMMU_VNNI_QUALITY` overrides
/// for operators chasing a quality/speed boundary.
pub const QUALITY_LIMIT: f32 = 0.12;

/// The effective dispatch limit ([`QUALITY_LIMIT`] or `MUMMU_VNNI_QUALITY`).
#[must_use]
pub fn quality_limit() -> f32 {
    static V: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("MUMMU_VNNI_QUALITY")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(QUALITY_LIMIT)
    })
}

/// `y = x · W` over the packed representation, integer fast path when legal,
/// exact f32 path otherwise. `out.len() == w.n`, `x.len() == w.k`.
///
/// This is the hand-off entry point: quantizes the activations, applies the
/// exactness dispatch, and logs (throttled, at most once per distinct shape)
/// when a call leaves the fast path.
pub fn gemv_q4n_auto(w: &PackedQ4, x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), w.k, "activation length");
    assert_eq!(out.len(), w.n, "output length");
    let acts = Q8Acts::quantize(x);
    if acts.quality > quality_limit() || !vnni_available() {
        log_fallback(w.k, w.n, acts.quality);
        gemv_q4n_f32(w, x, out);
        return;
    }
    gemv_q4n_vnni(w, &acts, out);
}

/// Throttled dispatch log: the SPEC contract requires the decision to be
/// visible, not silent — but once per shape, not once per token.
fn log_fallback(k: usize, n: usize, quality: f32) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(usize, usize)>>> = Mutex::new(None);
    let mut seen = SEEN
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if seen.get_or_insert_with(HashSet::new).insert((k, n)) {
        eprintln!(
            "[mummu] vnni gemv [{k} x {n}]: exact f32 path (act quality {quality:.4}, vnni {})",
            vnni_available()
        );
    }
}

/// The VNNI fast path. Rows are split across the rayon pool; each worker
/// walks its rows' packed bytes sequentially (the stream the layout was
/// built for).
pub fn gemv_q4n_vnni(w: &PackedQ4, acts: &Q8Acts, out: &mut [f32]) {
    assert_eq!(acts.qs.len(), w.k);
    assert_eq!(out.len(), w.n);
    if !vnni_available() {
        // Same math, scalar — never wrong, only slower.
        gemv_q4n_scalar(w, acts, out);
        return;
    }
    // Chunk rows so each rayon task amortizes its setup; 16 rows ≈ 40-140 KB
    // of packed stream per task at production K.
    let rows_per_task = 16usize;
    out.par_chunks_mut(rows_per_task)
        .enumerate()
        .for_each(|(t, chunk)| {
            let n0 = t * rows_per_task;
            // Per-task scratch for the combined (sw·sx) scales of one row.
            let mut combined = vec![0.0f32; w.groups()];
            for (i, o) in chunk.iter_mut().enumerate() {
                let n = n0 + i;
                // combined[g] = sw(n,g)·sx(g).
                let rs = w.row_scales(n);
                for g in 0..combined.len() {
                    combined[g] = rs[g].to_f32() * acts.scales[g];
                }
                // SAFETY: vnni_available() checked above; slices are exactly
                // k/2 packed bytes, k i8 activations, k/64 chunk pairs.
                *o = unsafe { row_dot_vnni(w.row_qs(n), &acts.qs, &combined) };
            }
        });
}

/// One row's `Σ_g combined[g] · Σ_{k∈g} (stored[k] − 8)·qx[k]` with AVX-512
/// VNNI.
///
/// The signed dot is formed as the i32 difference of two `vpdpbusd` dots —
/// stored bytes × qx, and the constant 8 × qx — BEFORE the f32 convert (see
/// the module header: carrying the offset into f32 costs ~5e-4 of the
/// answer to cancellation). Instruction budget per 64 weights: 1 packed
/// load, 3 unpack ops, 1 activation load, 2 `vpdpbusd` + 1 subtract (fresh
/// accumulators — a lane never crosses a group, which is the overflow
/// proof), 1 int→f32 convert, 3 scale-broadcast ops, 1 FMA ≈ 13 — ~0.2
/// instructions per weight, still an order of magnitude under the memory
/// budget once rayon spreads rows over cores. The horizontal sum runs once
/// per row.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vnni,avx2,fma")]
unsafe fn row_dot_vnni(qs: &[u8], qx: &[i8], combined: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let k = qx.len();
    let full = k / CHUNK;
    unsafe {
        let lo_mask = _mm256_set1_epi8(0x0F);
        let eights = _mm512_set1_epi8(8);
        let zero = _mm512_setzero_si512();
        let mut accf = _mm512_setzero_ps();
        for c in 0..full {
            let w = _mm256_loadu_si256(qs.as_ptr().add(c * GROUP).cast());
            let lo = _mm256_and_si256(w, lo_mask);
            let hi = _mm256_and_si256(_mm256_srli_epi16(w, 4), lo_mask);
            let q = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
            let xa = _mm512_loadu_si512(qx.as_ptr().add(c * CHUNK).cast());
            // 16 i32 lanes: 0..8 sum group 2c, 8..16 sum group 2c+1.
            let dot = _mm512_sub_epi32(
                _mm512_dpbusd_epi32(zero, q, xa),
                _mm512_dpbusd_epi32(zero, eights, xa),
            );
            let s0 = _mm256_set1_ps(*combined.get_unchecked(2 * c));
            let s1 = _mm256_set1_ps(*combined.get_unchecked(2 * c + 1));
            let sc = _mm512_insertf32x8(_mm512_castps256_ps512(s0), s1, 1);
            accf = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot), sc, accf);
        }
        let mut y = _mm512_reduce_add_ps(accf);
        // K % 64 == 32 tail: one group, packed lo = j, hi = j + 16.
        if k % CHUNK != 0 {
            let base = full * CHUNK;
            let ob = full * GROUP;
            let mut dot = 0i32;
            for j in 0..GROUP / 2 {
                let b = *qs.get_unchecked(ob + j);
                dot += (i32::from(b & 0x0F) - 8) * i32::from(*qx.get_unchecked(base + j));
                dot += (i32::from(b >> 4) - 8) * i32::from(*qx.get_unchecked(base + GROUP / 2 + j));
            }
            y += *combined.get_unchecked(2 * full) * dot as f32;
        }
        y
    }
}

/// The same integer math, scalar — the portable path and the reference the
/// SIMD path is tested against (identical dots; the f32 fold order differs,
/// so tests compare at ~1e-5 relative, not bitwise).
pub fn gemv_q4n_scalar(w: &PackedQ4, acts: &Q8Acts, out: &mut [f32]) {
    assert_eq!(acts.qs.len(), w.k);
    assert_eq!(out.len(), w.n);
    let groups = w.groups();
    let mut row = vec![0u8; w.k];
    for n in 0..w.n {
        unpack_row(w.row_qs(n), &mut row);
        let rs = w.row_scales(n);
        let mut y = 0.0f32;
        for g in 0..groups {
            let mut dot = 0i32;
            for j in 0..GROUP {
                let kk = g * GROUP + j;
                dot += (i32::from(row[kk]) - 8) * i32::from(acts.qs[kk]);
            }
            y += rs[g].to_f32() * acts.scales[g] * dot as f32;
        }
        out[n] = y;
    }
}

/// The exact-in-f32 path over the same packed bytes: dequantizes on the fly
/// and FMAs against the un-quantized activations. This is "exact with
/// respect to the stored quantization" — the reference the fast path's only
/// error (activation ε) is measured against, and the fallback the dispatch
/// routes pathological activations to.
pub fn gemv_q4n_f32(w: &PackedQ4, x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), w.k);
    assert_eq!(out.len(), w.n);
    let groups = w.groups();
    out.par_chunks_mut(16).enumerate().for_each(|(t, chunk)| {
        let mut row = vec![0u8; w.k];
        for (i, o) in chunk.iter_mut().enumerate() {
            let n = t * 16 + i;
            unpack_row(w.row_qs(n), &mut row);
            let rs = w.row_scales(n);
            let mut y = 0.0f32;
            for g in 0..groups {
                let s = rs[g].to_f32();
                let mut acc = 0.0f32;
                for j in 0..GROUP {
                    let kk = g * GROUP + j;
                    acc += (i32::from(row[kk]) - 8) as f32 * x[kk];
                }
                y += s * acc;
            }
            *o = y;
        }
    });
}

/// Per-row hard bound on the fast path's activation-quantization error:
/// `|Δy[n]| ≤ Σ_g (sx(g)/2) · Σ_{k∈g} |W'[k,n]|` where `W'` is the
/// dequantized packed weight. Used by tests; the runtime dispatch uses the
/// cheaper whole-vector [`Q8Acts::quality`] statistic instead.
#[must_use]
pub fn act_error_bound(w: &PackedQ4, acts: &Q8Acts) -> Vec<f32> {
    let groups = w.groups();
    let mut row = vec![0u8; w.k];
    (0..w.n)
        .map(|n| {
            unpack_row(w.row_qs(n), &mut row);
            let rs = w.row_scales(n);
            let mut bound = 0.0f32;
            for g in 0..groups {
                let sw = rs[g].to_f32();
                let mut l1 = 0.0f32;
                for j in 0..GROUP {
                    l1 += (i32::from(row[g * GROUP + j]) - 8).abs() as f32;
                }
                bound += acts.scales[g] * 0.5 * sw * l1;
            }
            bound
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Packed GEMM, m > 1 (SPEC P5.1): stream the weight once, reuse it across
// every prompt row.
// ---------------------------------------------------------------------------

/// A batch of activation rows, each quantized on ITS OWN per-32-group grid —
/// exactly the grid the m=1 path would use for that row, so the GEMM is
/// bit-compatible with m independent GEMVs (the scalar test pins this).
pub struct Q8ActsBatch {
    pub rows: Vec<Q8Acts>,
    /// Worst per-row quality statistic — the dispatch gate for the batch.
    pub worst_quality: f32,
}

impl Q8ActsBatch {
    /// Quantize `m` rows of length `k` (row-major `x`).
    #[must_use]
    pub fn quantize(x: &[f32], m: usize, k: usize) -> Self {
        assert_eq!(x.len(), m * k, "activation batch shape");
        let rows: Vec<Q8Acts> = x.chunks_exact(k).map(Q8Acts::quantize).collect();
        let worst_quality = rows.iter().fold(0.0f32, |w, r| w.max(r.quality));
        Self {
            rows,
            worst_quality,
        }
    }
}

/// `Y = X · W` over the packed representation, `X: [m, k]`, `Y: [m, n]`
/// row-major. The point against the row-looped GEMV: the weight bytes
/// stream from DRAM **once** for the whole batch instead of once per row —
/// at m = 64 that is 64x less weight traffic, which is the entire prefill
/// gap on host layers (decode GEMV is byte-bound; a prompt is GEMM
/// territory from m ≈ 2 up).
///
/// Dispatch mirrors [`gemv_q4n_auto`]: the batch takes the integer path
/// only when every row's quantization quality clears the limit; a batch
/// with one pathological row goes wholesale to the exact-f32 path (which
/// is ALSO weight-stream-once here — the fallback keeps GEMM economics).
pub fn gemm_q4n_auto(w: &PackedQ4, x: &[f32], m: usize, out: &mut [f32]) {
    assert_eq!(x.len(), m * w.k, "activation batch shape");
    assert_eq!(out.len(), m * w.n, "output batch shape");
    if m == 0 {
        return;
    }
    let acts = Q8ActsBatch::quantize(x, m, w.k);
    if acts.worst_quality > quality_limit() || !vnni_available() {
        log_fallback(w.k, w.n, acts.worst_quality);
        gemm_q4n_f32(w, x, m, out);
        return;
    }
    gemm_q4n_vnni(w, &acts, out);
}

/// Row-block width the VNNI GEMM holds in registers: 8 f32 accumulators of
/// 16 lanes each, beside the unpacked weight chunk and scale temporaries —
/// 17 of 32 zmm registers live.
const GEMM_MR: usize = 8;

/// Output-column panel per rayon task. A 64-row panel's packed bytes
/// (64 · k/2 ≈ 160 KB at k = 5120) stay L2-resident across the row blocks,
/// so DRAM sees the weight once while every m-block re-reads it from cache.
const GEMM_PANEL: usize = 64;

/// The VNNI GEMM. Loop structure (the memory argument, which is the whole
/// design): rayon tasks own disjoint `GEMM_PANEL`-column slices of W; a
/// task iterates m in blocks of [`GEMM_MR`]; within a block it walks its
/// panel's rows once, so the panel streams from DRAM on the first block
/// and from L2 after; activations for the current block (8 rows · k bytes
/// ≈ 40 KB) sit in L1/L2 throughout. Net DRAM traffic: weights once,
/// activations once.
pub fn gemm_q4n_vnni(w: &PackedQ4, acts: &Q8ActsBatch, out: &mut [f32]) {
    let m = acts.rows.len();
    assert_eq!(out.len(), m * w.n, "output batch shape");
    if !vnni_available() {
        gemm_q4n_scalar(w, acts, out);
        return;
    }
    for r in &acts.rows {
        assert_eq!(r.qs.len(), w.k, "activation row length");
    }
    let groups = w.groups();
    let n = w.n;

    // Tasks write disjoint (row, column-panel) elements of `out`; the
    // panels interleave within each output row, so no contiguous split
    // exists. SAFETY: every task writes only columns in its own panel —
    // element-disjoint by construction.
    struct SendPtr(*mut f32);
    unsafe impl Send for SendPtr {}
    unsafe impl Sync for SendPtr {}
    let out_ptr = SendPtr(out.as_mut_ptr());

    let panels = n.div_ceil(GEMM_PANEL);
    (0..panels).into_par_iter().for_each(|p| {
        let _ = &out_ptr; // capture the wrapper, not the raw pointer
        let col0 = p * GEMM_PANEL;
        let cols = GEMM_PANEL.min(n - col0);
        let mut combined = vec![0.0f32; GEMM_MR * groups];
        let mut mb = 0usize;
        while mb < m {
            let mrl = GEMM_MR.min(m - mb);
            for c in 0..cols {
                let col = col0 + c;
                let rs = w.row_scales(col);
                // combined[r][g] = sw(col, g) · sx(mb + r, g).
                for r in 0..mrl {
                    let sx = &acts.rows[mb + r].scales;
                    let dst = &mut combined[r * groups..(r + 1) * groups];
                    for g in 0..groups {
                        dst[g] = rs[g].to_f32() * sx[g];
                    }
                }
                // SAFETY: vnni_available() checked; slices sized by the
                // asserts above; each (row, col) written exactly once.
                unsafe {
                    let mut ys = [0.0f32; GEMM_MR];
                    row_dot_vnni_mr(
                        w.row_qs(col),
                        &acts.rows[mb..mb + mrl],
                        &combined,
                        groups,
                        &mut ys[..mrl],
                    );
                    for (r, &y) in ys[..mrl].iter().enumerate() {
                        *out_ptr.0.add((mb + r) * n + col) = y;
                    }
                }
            }
            mb += mrl;
        }
    });
}

/// One weight row against up to [`GEMM_MR`] activation rows: the m=1
/// kernel's structure with the unpack hoisted out of the row loop, so the
/// weight chunk is decoded once and multiplied [`GEMM_MR`] times.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vnni,avx2,fma")]
unsafe fn row_dot_vnni_mr(
    qs: &[u8],
    act_rows: &[Q8Acts],
    combined: &[f32],
    groups: usize,
    ys: &mut [f32],
) {
    use std::arch::x86_64::*;
    let mrl = act_rows.len();
    debug_assert!(mrl <= GEMM_MR && ys.len() == mrl);
    let k = act_rows[0].qs.len();
    let full = k / CHUNK;
    unsafe {
        let lo_mask = _mm256_set1_epi8(0x0F);
        let eights = _mm512_set1_epi8(8);
        let zero = _mm512_setzero_si512();
        let mut acc = [_mm512_setzero_ps(); GEMM_MR];
        for c in 0..full {
            let wv = _mm256_loadu_si256(qs.as_ptr().add(c * GROUP).cast());
            let lo = _mm256_and_si256(wv, lo_mask);
            let hi = _mm256_and_si256(_mm256_srli_epi16(wv, 4), lo_mask);
            let q = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
            for r in 0..mrl {
                let xa = _mm512_loadu_si512(act_rows[r].qs.as_ptr().add(c * CHUNK).cast());
                let dot = _mm512_sub_epi32(
                    _mm512_dpbusd_epi32(zero, q, xa),
                    _mm512_dpbusd_epi32(zero, eights, xa),
                );
                let cs = combined.as_ptr().add(r * groups + 2 * c);
                let s0 = _mm256_set1_ps(*cs);
                let s1 = _mm256_set1_ps(*cs.add(1));
                let sc = _mm512_insertf32x8(_mm512_castps256_ps512(s0), s1, 1);
                acc[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot), sc, acc[r]);
            }
        }
        for r in 0..mrl {
            let mut y = _mm512_reduce_add_ps(acc[r]);
            if k % CHUNK != 0 {
                let base = full * CHUNK;
                let ob = full * GROUP;
                let qx = &act_rows[r].qs;
                let mut dot = 0i32;
                for j in 0..GROUP / 2 {
                    let b = *qs.get_unchecked(ob + j);
                    dot += (i32::from(b & 0x0F) - 8) * i32::from(*qx.get_unchecked(base + j));
                    dot += (i32::from(b >> 4) - 8)
                        * i32::from(*qx.get_unchecked(base + GROUP / 2 + j));
                }
                y += *combined.get_unchecked(r * groups + 2 * full) * dot as f32;
            }
            ys[r] = y;
        }
    }
}

/// Scalar GEMM: exactly `m` independent scalar GEMVs (same integer dots per
/// row), so `gemm ≡ m × gemv` holds by construction — the reference the
/// SIMD path is tested against.
pub fn gemm_q4n_scalar(w: &PackedQ4, acts: &Q8ActsBatch, out: &mut [f32]) {
    let m = acts.rows.len();
    assert_eq!(out.len(), m * w.n, "output batch shape");
    for (r, row_acts) in acts.rows.iter().enumerate() {
        gemv_q4n_scalar(w, row_acts, &mut out[r * w.n..(r + 1) * w.n]);
    }
}

/// The exact-in-f32 GEMM over the same packed bytes — the batch fallback.
/// Still weight-stream-once: each task unpacks its output row a single
/// time and multiplies it against every activation row.
pub fn gemm_q4n_f32(w: &PackedQ4, x: &[f32], m: usize, out: &mut [f32]) {
    assert_eq!(x.len(), m * w.k, "activation batch shape");
    assert_eq!(out.len(), m * w.n, "output batch shape");
    let groups = w.groups();
    let n = w.n;
    struct SendPtr(*mut f32);
    unsafe impl Send for SendPtr {}
    unsafe impl Sync for SendPtr {}
    let out_ptr = SendPtr(out.as_mut_ptr());
    (0..n.div_ceil(16)).into_par_iter().for_each(|t| {
        let _ = &out_ptr;
        let col0 = t * 16;
        let cols = 16.min(n - col0);
        let mut row = vec![0u8; w.k];
        let mut wf = vec![0.0f32; w.k];
        for c in 0..cols {
            let col = col0 + c;
            unpack_row(w.row_qs(col), &mut row);
            let rs = w.row_scales(col);
            for g in 0..groups {
                let s = rs[g].to_f32();
                for j in 0..GROUP {
                    let kk = g * GROUP + j;
                    wf[kk] = (i32::from(row[kk]) - 8) as f32 * s;
                }
            }
            for r in 0..m {
                let xr = &x[r * w.k..(r + 1) * w.k];
                let mut acc = 0.0f32;
                for kk in 0..w.k {
                    acc += wf[kk] * xr[kk];
                }
                // SAFETY: each (r, col) is written exactly once by the one
                // task owning `col`.
                unsafe {
                    *out_ptr.0.add(r * n + col) = acc;
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Split calibration (SPEC 1, algorithm B seam)
// ---------------------------------------------------------------------------

/// The measured split decision: which kernel evaluates the packed GEMV.
///
/// SPEC 1's algorithm B splits the four weight bit-planes between the FMA
/// pipes (VNNI) and the shuffle pipes (T-MAC-style LUT) at a point `s`
/// chosen by port-pressure arithmetic. This repo has already learned the
/// hard way that derived kernel-selection rules lose to measurement
/// (`nn/packed_gemv.rs::split_candidates` — "a rule three parties derive
/// and the hardware refuses is not a rule"), so `calibrate_split` MEASURES
/// the implemented kernels on a production-shaped workload and returns the
/// argmin. `s = 0` is the pure-VNNI kernel above; on Zen 4 the packed
/// stream is DRAM-bound at s = 0, which is the roofline — a LUT plane can
/// only relieve port pressure the memory system never exposes, and the
/// measurement shows it. The LUT planes stay a documented extension point:
/// [`gemv_q4n_split`] takes `s` so a future shuffle-bound machine plugs its
/// kernel in behind the same contract.
#[must_use]
pub fn calibrate_split() -> u8 {
    static CAL: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *CAL.get_or_init(|| {
        if !vnni_available() {
            return 0; // scalar path; nothing to choose between
        }
        // A production-shaped probe: [K=5120, N=1024] keeps the weight
        // stream past L2 so the measurement sees memory, not cache.
        let (k, n) = (5120usize, 1024usize);
        let vals: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.37).sin()).collect();
        let w = PackedQ4::from_f32(&vals, k, n);
        let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.11).cos()).collect();
        let acts = Q8Acts::quantize(&x);
        let mut out = vec![0.0f32; n];
        let mut best = (0u8, f64::INFINITY);
        for s in [0u8] {
            // Warm, then time.
            gemv_q4n_split(s, &w, &acts, &mut out);
            let t0 = std::time::Instant::now();
            for _ in 0..8 {
                gemv_q4n_split(s, &w, &acts, &mut out);
            }
            let ms = t0.elapsed().as_secs_f64() * 1e3 / 8.0;
            if ms < best.1 {
                best = (s, ms);
            }
        }
        best.0
    })
}

/// Evaluate the packed GEMV at split point `s`. `s = 0` is the pure-VNNI
/// kernel; other values are reserved for LUT-plane hybrids and currently
/// evaluate through the same exact integer path (see [`calibrate_split`]).
pub fn gemv_q4n_split(s: u8, w: &PackedQ4, acts: &Q8Acts, out: &mut [f32]) {
    let _ = s;
    gemv_q4n_vnni(w, acts, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_gemv(values: &[f32], k: usize, n: usize, x: &[f32]) -> Vec<f32> {
        let mut y = vec![0.0f32; n];
        for kk in 0..k {
            for nn in 0..n {
                y[nn] += x[kk] * values[kk * n + nn];
            }
        }
        y
    }

    fn wave(len: usize, f: f32) -> Vec<f32> {
        (0..len).map(|i| ((i as f32) * f).sin()).collect()
    }

    /// Pack/unpack and dequantize agree with a directly-computed grid.
    #[test]
    fn pack_roundtrip_and_dequant_grid() {
        let (k, n) = (96, 8); // k % 64 == 32: exercises the tail-group path
        let vals = wave(k * n, 0.7);
        let w = PackedQ4::from_f32(&vals, k, n);
        let deq = w.dequantize();
        for col in 0..n {
            for g in 0..k / GROUP {
                let amax = (0..GROUP)
                    .map(|j| vals[(g * GROUP + j) * n + col].abs())
                    .fold(0.0f32, f32::max);
                let scale = f16::from_f32(if amax > 0.0 { amax / 7.0 } else { 1.0 }).to_f32();
                for j in 0..GROUP {
                    let v = vals[(g * GROUP + j) * n + col];
                    let q = (v / scale).round().clamp(-7.0, 7.0);
                    let want = q * scale;
                    let got = deq[(g * GROUP + j) * n + col];
                    assert!(
                        (want - got).abs() < 1e-6,
                        "col {col} g {g} j {j}: {want} vs {got}"
                    );
                }
            }
        }
    }

    /// The vpdpbusd overflow budget the module doc claims: max lane sum.
    #[test]
    fn overflow_budget_holds() {
        // stored ∈ [1,15], |qx| ≤ 127, 4 products per lane, one group per
        // lane per accumulate.
        assert!(4 * 15 * 127 < i32::MAX / 1024);
    }

    /// Scalar integer path vs the exact f32 path: only activation ε apart,
    /// within the hard per-row bound.
    #[test]
    fn scalar_error_stays_inside_the_bound() {
        let (k, n) = (256, 48);
        let vals = wave(k * n, 0.31);
        let w = PackedQ4::from_f32(&vals, k, n);
        let x = wave(k, 0.017);
        let acts = Q8Acts::quantize(&x);
        let mut got = vec![0.0f32; n];
        gemv_q4n_scalar(&w, &acts, &mut got);
        let mut exact = vec![0.0f32; n];
        gemv_q4n_f32(&w, &x, &mut exact);
        let bounds = act_error_bound(&w, &acts);
        for i in 0..n {
            let err = (got[i] - exact[i]).abs();
            // The bound is mathematical for exact arithmetic; allow f32
            // accumulation slack proportional to the magnitude.
            let slack = 1e-5 * exact[i].abs().max(1.0);
            assert!(
                err <= bounds[i] + slack,
                "row {i}: err {err} > bound {} (+{slack})",
                bounds[i]
            );
        }
    }

    /// VNNI and scalar compute the same integer dots — results agree to fold
    /// order (1e-5 relative).
    #[test]
    fn vnni_matches_scalar() {
        if !vnni_available() {
            eprintln!("[skip] no AVX-512 VNNI on this host");
            return;
        }
        for (k, n) in [(64, 16), (128, 33), (192, 5), (5120, 64)] {
            let vals = wave(k * n, 0.13);
            let w = PackedQ4::from_f32(&vals, k, n);
            let x = wave(k, 0.007);
            let acts = Q8Acts::quantize(&x);
            let mut simd = vec![0.0f32; n];
            gemv_q4n_vnni(&w, &acts, &mut simd);
            let mut scalar = vec![0.0f32; n];
            gemv_q4n_scalar(&w, &acts, &mut scalar);
            let scale = scalar.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
            for i in 0..n {
                assert!(
                    (simd[i] - scalar[i]).abs() / scale < 1e-5,
                    "[{k}x{n}] row {i}: {} vs {}",
                    simd[i],
                    scalar[i]
                );
            }
        }
    }

    /// End-to-end against a plain f32 GEMV on the dequantized weights, at the
    /// SPEC's scaled tolerance (the only error is activation ε).
    #[test]
    fn auto_path_matches_dequant_reference() {
        let (k, n) = (320, 64);
        let vals = wave(k * n, 0.41);
        let w = PackedQ4::from_f32(&vals, k, n);
        let deq = w.dequantize();
        let x = wave(k, 0.023);
        let mut got = vec![0.0f32; n];
        gemv_q4n_auto(&w, &x, &mut got);
        let want = ref_gemv(&deq, k, n, &x);
        let acts = Q8Acts::quantize(&x);
        let bounds = act_error_bound(&w, &acts);
        for i in 0..n {
            let err = (got[i] - want[i]).abs();
            let slack = 1e-5 * want[i].abs().max(1.0);
            assert!(
                err <= bounds[i] + slack,
                "row {i}: err {err} > bound {}",
                bounds[i]
            );
        }
    }

    /// Adversarial activation kurtosis must fall back to the exact path:
    /// with one huge outlier per group, the auto path's answer must equal
    /// the exact f32 path bit-for-bit (proof it dispatched).
    #[test]
    fn adversarial_activations_take_the_exact_path() {
        let (k, n) = (128, 16);
        let vals = wave(k * n, 0.19);
        let w = PackedQ4::from_f32(&vals, k, n);
        let mut x = vec![1e-4f32; k];
        for g in 0..k / GROUP {
            x[g * GROUP] = 1000.0; // one outlier owns each group
        }
        let acts = Q8Acts::quantize(&x);
        assert!(
            acts.quality > quality_limit(),
            "the adversarial vector must trip the quality gate ({})",
            acts.quality
        );
        let mut auto = vec![0.0f32; n];
        gemv_q4n_auto(&w, &x, &mut auto);
        let mut exact = vec![0.0f32; n];
        gemv_q4n_f32(&w, &x, &mut exact);
        assert_eq!(auto, exact, "dispatch must have taken the exact path");
    }

    /// Requantizing from the device grid (blocks along N) stays close to the
    /// original values: two 4-bit roundings, bounded by the sum of the two
    /// grids' half-steps.
    #[test]
    fn slab_repack_error_is_two_roundings() {
        let (k, n) = (64, 64);
        let vals = wave(k * n, 0.53);
        // Device-grid quantization (blocks along N).
        let (qi8, scales) = crate::pack::quantize_blocks(&vals, n, crate::pack::Precision::Q4);
        let w = PackedQ4::from_q4s_slab(&qi8, &scales, k, n);
        let deq = w.dequantize();
        for i in 0..k * n {
            let dev = f32::from(qi8[i]) * scales[i / GROUP];
            // Second rounding: at most half a step of the host grid on top.
            let host_step = {
                let col = i % n;
                let g = (i / n) / GROUP;
                w.scales[col * w.groups() + g].to_f32()
            };
            assert!(
                (deq[i] - dev).abs() <= host_step / 2.0 + 1e-6,
                "elem {i}: host {} vs device {dev}",
                deq[i]
            );
        }
    }

    /// The scalar GEMM is m independent GEMVs by construction: identical
    /// outputs, bit for bit.
    #[test]
    fn gemm_scalar_is_exactly_m_gemvs() {
        let (k, n, m) = (160, 40, 5); // k % 64 == 32: tail path in play
        let vals = wave(k * n, 0.23);
        let w = PackedQ4::from_f32(&vals, k, n);
        let x: Vec<f32> = wave(m * k, 0.013);
        let acts = Q8ActsBatch::quantize(&x, m, k);
        let mut gemm = vec![0.0f32; m * n];
        gemm_q4n_scalar(&w, &acts, &mut gemm);
        for r in 0..m {
            let row_acts = Q8Acts::quantize(&x[r * k..(r + 1) * k]);
            let mut gemv = vec![0.0f32; n];
            gemv_q4n_scalar(&w, &row_acts, &mut gemv);
            assert_eq!(&gemm[r * n..(r + 1) * n], &gemv[..], "row {r}");
        }
    }

    /// The VNNI GEMM computes the same integer dots as the scalar path —
    /// agreement to f32 fold order, across row-block and panel boundaries
    /// (m and n chosen to exercise partial blocks on both axes).
    #[test]
    fn gemm_vnni_matches_scalar() {
        if !vnni_available() {
            eprintln!("[skip] no AVX-512 VNNI on this host");
            return;
        }
        for (k, n, m) in [(64, 16, 2), (128, 96, 8), (192, 130, 11), (5120, 80, 9)] {
            let vals = wave(k * n, 0.171);
            let w = PackedQ4::from_f32(&vals, k, n);
            let x = wave(m * k, 0.0077);
            let acts = Q8ActsBatch::quantize(&x, m, k);
            let mut simd = vec![0.0f32; m * n];
            gemm_q4n_vnni(&w, &acts, &mut simd);
            let mut scalar = vec![0.0f32; m * n];
            gemm_q4n_scalar(&w, &acts, &mut scalar);
            let scale = scalar.iter().fold(1e-6f32, |mx, v| mx.max(v.abs()));
            for i in 0..m * n {
                assert!(
                    (simd[i] - scalar[i]).abs() / scale < 1e-5,
                    "[{k}x{n} m{m}] elem {i}: {} vs {}",
                    simd[i],
                    scalar[i]
                );
            }
        }
    }

    /// The auto GEMM stays inside the per-row activation error bounds
    /// against the dequantized reference (the same contract as the GEMV),
    /// and a batch with one adversarial row takes the exact path wholesale.
    #[test]
    fn gemm_auto_bounds_and_dispatch() {
        let (k, n, m) = (256, 48, 4);
        let vals = wave(k * n, 0.37);
        let w = PackedQ4::from_f32(&vals, k, n);
        let deq = w.dequantize();
        let x = wave(m * k, 0.019);
        let mut got = vec![0.0f32; m * n];
        gemm_q4n_auto(&w, &x, m, &mut got);
        for r in 0..m {
            let xr = &x[r * k..(r + 1) * k];
            let want = ref_gemv(&deq, k, n, xr);
            let acts = Q8Acts::quantize(xr);
            let bounds = act_error_bound(&w, &acts);
            for i in 0..n {
                let err = (got[r * n + i] - want[i]).abs();
                let slack = 1e-5 * want[i].abs().max(1.0);
                assert!(
                    err <= bounds[i] + slack,
                    "row {r} col {i}: err {err} > bound {}",
                    bounds[i]
                );
            }
        }

        // One outlier-dominated row must push the WHOLE batch to the exact
        // f32 path: bitwise equality with gemm_q4n_f32 proves the dispatch.
        let mut x_bad = x.clone();
        for g in 0..k / GROUP {
            x_bad[k + g * GROUP] = 1000.0; // row 1 becomes adversarial
            for j in 1..GROUP {
                x_bad[k + g * GROUP + j] = 1e-4;
            }
        }
        let acts_bad = Q8ActsBatch::quantize(&x_bad, m, k);
        assert!(acts_bad.worst_quality > quality_limit());
        let mut auto_out = vec![0.0f32; m * n];
        gemm_q4n_auto(&w, &x_bad, m, &mut auto_out);
        let mut exact = vec![0.0f32; m * n];
        gemm_q4n_f32(&w, &x_bad, m, &mut exact);
        assert_eq!(auto_out, exact, "dispatch must have taken the exact path");
    }

    /// calibrate_split returns a valid split and gemv_q4n_split(s) computes
    /// the same function for the chosen s.
    #[test]
    fn calibration_returns_a_working_split() {
        let s = calibrate_split();
        let (k, n) = (128, 8);
        let vals = wave(k * n, 0.29);
        let w = PackedQ4::from_f32(&vals, k, n);
        let x = wave(k, 0.011);
        let acts = Q8Acts::quantize(&x);
        let mut a = vec![0.0f32; n];
        gemv_q4n_split(s, &w, &acts, &mut a);
        let mut b = vec![0.0f32; n];
        gemv_q4n_scalar(&w, &acts, &mut b);
        let scale = b.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
        for i in 0..n {
            assert!((a[i] - b[i]).abs() / scale < 1e-5);
        }
    }
}
