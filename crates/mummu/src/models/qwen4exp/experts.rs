//! Routed mixture-of-experts for qwen4exp: the softmax top-k router and the
//! routed experts computed straight from the GGUF expert banks, which are
//! never loaded whole.
//!
//! **Why the banks stay on disk.** Every one of the 48 blocks carries three
//! banks (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`) of 512 experts;
//! quantized they are ~77 GB of the 111 GB file, and dequantized to f32 an
//! expert alone is ~20 MB (`3 · 640 · 2560` weights). A token touches 10 of
//! 512 per layer. So an expert is located by arithmetic — expert `e` of a
//! bank is the byte range `[e·out·row_bytes, (e+1)·out·row_bytes)` from the
//! bank's payload start — read with one positional read per projection
//! (0.9 MB for a Q4_K gate/up, 1.2 MB Q5_1 down, 1.7 MB Q8_0 down), and
//! multiplied row-chunk by row-chunk: each chunk of quantized rows is
//! dequantized with [`gguf::dequantize`] (whole blocks only) and dotted with
//! the inputs, so the f32 rows of one expert never exist all at once.
//!
//! **Host-slice API.** Everything here takes `&[f32]` and returns
//! `Vec<f32>`, row-major (`[n x width]`: token `t` is
//! `xs[t*width..(t+1)*width]`). No Burn tensors: the shared expert, its
//! sigmoid gate and the router `Linear` are ordinary trunk weights the
//! integrator computes with Burn, handing this module the router logits and
//! the HC-mixed input and adding its output to the shared path.
//!
//! **Numerics (llama.cpp `build_moe_ffn`, qwen4exp.cpp `build_layer_ffn`).**
//! Gating is `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX` over all experts;
//! selection is the `k` largest probabilities; `norm_w = true` divides the
//! selected probabilities by their sum clamped to at least
//! [`WEIGHT_SUM_FLOOR`] (the smallest positive F16, llama.cpp's guard); and
//! qwen4exp never loads `expert_weights_scale`, so it stays 0 and **no scale
//! is applied**. Each expert is `down(silu(gate·x) · (up·x))` with
//! `silu(x) = x·sigmoid(x)` (`LLM_FFN_SILU` with a gate: `swiglu_split`),
//! weighted after the FFN (`weight_before_ffn` is llama4-only), and summed.
//!
//! **Determinism.** Every output element is one [`dot`] of one weight row
//! with one input row, independent of chunking, threading, batch
//! composition and of whether the row came from the quantized bytes or the
//! f32 cache. That is what makes the cache bit-exact and makes a token's
//! routed output bit-identical whether it arrives alone or in a prefill
//! batch (the property the chunked-prefill invariant in the port spec §7
//! needs from this block).

use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use rayon::prelude::*;

use crate::gguf::{self, GgmlType, GgufFile};

/// Environment knob for the f32 expert cache ceiling, in GiB (fractions
/// allowed; `0` disables the cache).
pub const CACHE_ENV: &str = "MUMMU_QWEN4EXP_EXPERT_CACHE_GB";

/// Cache ceiling when [`CACHE_ENV`] is unset or unparseable: OFF.
///
/// Measured on the real model (release, 15-token prompts + 24-token greedy,
/// warm page cache, two sessions on 2026-09-16, co-tenant load 7-14): the
/// 2 GiB cache this default used to be made prefill 4.4x slower (9.4 s vs
/// 2.1 s) and decode 2.3x slower (~2.0 vs ~0.87 s/token) with bit-identical
/// logits, and raised peak RSS from 19.2 to 22.3 GiB. A miss dequantizes and
/// stores the whole 19.7 MB expert, and ~100 cached experts churn against
/// the hundreds a prompt routes to per layer, so hits are too rare to repay
/// it. Opt back in with [`CACHE_ENV`] to measure a workload where they are
/// not.
pub const DEFAULT_CACHE_GB: f64 = 0.0;

/// Floor on the selected-probability sum before renormalizing: llama.cpp's
/// `ggml_clamp(weights_sum, 6.103515625e-5, INF)` — the smallest positive
/// F16 value, so the division can never blow up.
pub const WEIGHT_SUM_FLOOR: f32 = 1.0 / 16384.0;

/// f32 elements per dequantized row chunk (~256 KiB): L2-sized, so a chunk's
/// rows are dotted while still hot, and big enough that a rayon task
/// amortizes its dequant call. Gate/up (2560 wide) chunk 25 rows, down (640
/// wide) 102.
const CHUNK_ELEMS: usize = 1 << 16;

/// Which of an expert's three projections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BankKind {
    /// `ffn_gate_exps`: `ne = [hidden, ffn, n_expert]`.
    Gate,
    /// `ffn_up_exps`: `ne = [hidden, ffn, n_expert]`.
    Up,
    /// `ffn_down_exps`: `ne = [ffn, hidden, n_expert]`.
    Down,
}

impl BankKind {
    pub const ALL: [Self; 3] = [Self::Gate, Self::Up, Self::Down];

    /// The GGUF tensor name of this bank in block `layer`.
    #[must_use]
    pub fn tensor_name(self, layer: usize) -> String {
        let part = match self {
            Self::Gate => "gate",
            Self::Up => "up",
            Self::Down => "down",
        };
        format!("blk.{layer}.ffn_{part}_exps.weight")
    }
}

/// Where a bank's bytes live.
#[derive(Debug)]
enum Payload {
    /// A (shared) handle on the shard holding the bank, and the ABSOLUTE
    /// file offset of the bank's first byte.
    File {
        path: PathBuf,
        file: Arc<File>,
        offset: u64,
    },
    /// The whole bank in memory (tests, and tiny synthetic models).
    Memory(Vec<u8>),
}

/// One expert bank: `n_experts` matrices of `out_dim` rows x `in_dim`
/// columns, stored as ggml `ne = [in_dim, out_dim, n_experts]` (the first ne
/// varies fastest, so the bytes are expert-major, then row-major, and each
/// row is `in_dim / block_size` whole blocks).
#[derive(Debug)]
pub struct ExpertBank {
    dtype: GgmlType,
    in_dim: usize,
    out_dim: usize,
    n_experts: usize,
    row_bytes: usize,
    payload: Payload,
}

impl ExpertBank {
    /// Validate the geometry and return `(row_bytes, total_bytes)`.
    fn geometry(dtype: GgmlType, ne: [usize; 3]) -> Result<(usize, usize), String> {
        let [in_dim, out_dim, n_experts] = ne;
        if in_dim == 0 || out_dim == 0 || n_experts == 0 {
            return Err(format!("expert bank ne {ne:?} has a zero dimension"));
        }
        let block = usize::try_from(dtype.block_size()).map_err(|_| "block size over usize")?;
        let bpb = usize::try_from(dtype.bytes_per_block()).map_err(|_| "block bytes over usize")?;
        // Negative space: a row that is not whole blocks cannot be sliced
        // out of the bank, and dequantizing a misaligned slice would return
        // garbage rather than fail.
        if !in_dim.is_multiple_of(block) {
            return Err(format!(
                "expert bank ne {ne:?}: row width {in_dim} is not whole {dtype:?} blocks of {block}"
            ));
        }
        let row_bytes = in_dim / block * bpb;
        let total = n_experts
            .checked_mul(out_dim)
            .and_then(|v| v.checked_mul(row_bytes))
            .ok_or_else(|| format!("expert bank ne {ne:?} overflows usize bytes"))?;
        Ok((row_bytes, total))
    }

    /// A bank held in memory: `bytes` is the whole tensor payload.
    ///
    /// # Errors
    /// On a bad geometry or a byte count that is not exactly the bank.
    pub fn from_bytes(dtype: GgmlType, ne: [usize; 3], bytes: Vec<u8>) -> Result<Self, String> {
        let (row_bytes, total) = Self::geometry(dtype, ne)?;
        if bytes.len() != total {
            return Err(format!(
                "expert bank ne {ne:?} {dtype:?} is {total} bytes, got {}",
                bytes.len()
            ));
        }
        Ok(Self {
            dtype,
            in_dim: ne[0],
            out_dim: ne[1],
            n_experts: ne[2],
            row_bytes,
            payload: Payload::Memory(bytes),
        })
    }

    /// A bank read on demand from `file` (already open on `path`), whose
    /// payload starts at the absolute file `offset`.
    ///
    /// # Errors
    /// On a bad geometry, or when the file is too short to hold the bank —
    /// a truncated shard must fail here, not as a short read mid-decode.
    pub fn from_file(
        path: &Path,
        file: Arc<File>,
        offset: u64,
        dtype: GgmlType,
        ne: [usize; 3],
    ) -> Result<Self, String> {
        let (row_bytes, total) = Self::geometry(dtype, ne)?;
        let len = file
            .metadata()
            .map_err(|e| format!("{}: {e}", path.display()))?
            .len();
        let end = offset
            .checked_add(total as u64)
            .ok_or_else(|| format!("{}: bank end overflows u64", path.display()))?;
        if end > len {
            return Err(format!(
                "{}: expert bank needs bytes up to {end}, file is {len}",
                path.display()
            ));
        }
        Ok(Self {
            dtype,
            in_dim: ne[0],
            out_dim: ne[1],
            n_experts: ne[2],
            row_bytes,
            payload: Payload::File {
                path: path.to_path_buf(),
                file,
                offset,
            },
        })
    }

    #[must_use]
    pub fn dtype(&self) -> GgmlType {
        self.dtype
    }

    /// ggml `ne`: `[in_dim, out_dim, n_experts]`.
    #[must_use]
    pub fn ne(&self) -> [usize; 3] {
        [self.in_dim, self.out_dim, self.n_experts]
    }

    /// Columns of each expert matrix (the input width).
    #[must_use]
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    /// Rows of each expert matrix (the output width).
    #[must_use]
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    #[must_use]
    pub fn n_experts(&self) -> usize {
        self.n_experts
    }

    /// Bytes of one quantized row: `in_dim / block_size * bytes_per_block`.
    #[must_use]
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// Bytes of one expert: `out_dim * row_bytes`.
    #[must_use]
    pub fn expert_bytes_len(&self) -> usize {
        self.out_dim * self.row_bytes
    }

    /// The shard file and absolute payload offset, for a file-backed bank.
    #[must_use]
    pub fn location(&self) -> Option<(&Path, u64)> {
        match &self.payload {
            Payload::File { path, offset, .. } => Some((path, *offset)),
            Payload::Memory(_) => None,
        }
    }

    /// The quantized bytes of expert `e` (borrowed for an in-memory bank,
    /// one positional read for a file-backed one).
    ///
    /// # Errors
    /// On an out-of-range expert or an I/O failure.
    pub fn expert_bytes(&self, e: usize) -> Result<Cow<'_, [u8]>, String> {
        if e >= self.n_experts {
            return Err(format!(
                "expert {e} out of range (bank has {})",
                self.n_experts
            ));
        }
        let len = self.expert_bytes_len();
        match &self.payload {
            Payload::Memory(bytes) => Ok(Cow::Borrowed(&bytes[e * len..(e + 1) * len])),
            Payload::File { path, file, offset } => {
                let mut buf = vec![0u8; len];
                let at = offset + (e as u64) * (len as u64);
                read_exact_at(file, &mut buf, at)
                    .map_err(|err| format!("{} @ {at} (+{len}): {err}", path.display()))?;
                Ok(Cow::Owned(buf))
            }
        }
    }

    /// Expert `e` dequantized to f32, `[out_dim x in_dim]` row-major.
    /// Row chunks are dequantized in parallel and concatenated in order, so
    /// the values are exactly [`gguf::dequantize`] of the expert's bytes.
    ///
    /// # Errors
    /// As [`Self::expert_bytes`], or on a dtype without a dequantizer.
    pub fn dequantize_expert(&self, e: usize) -> Result<Vec<f32>, String> {
        let bytes = self.expert_bytes(e)?;
        let chunk_rows = (CHUNK_ELEMS / self.in_dim).max(1);
        let parts: Vec<Vec<f32>> = bytes
            .par_chunks(chunk_rows * self.row_bytes)
            .map(|chunk| gguf::dequantize(self.dtype, chunk))
            .collect::<Result<_, _>>()?;
        let out: Vec<f32> = parts.concat();
        assert_eq!(out.len(), self.out_dim * self.in_dim, "whole expert out");
        Ok(out)
    }

    /// `W_e · x` for `n` input rows: `xs` is `[n x in_dim]`, the result
    /// `[n x out_dim]`. Rows are dequantized chunk by chunk, never the whole
    /// bank and never the whole expert at once.
    ///
    /// # Errors
    /// On a mis-sized `xs`, an out-of-range expert or an I/O failure.
    pub fn matvec(&self, e: usize, xs: &[f32], n: usize) -> Result<Vec<f32>, String> {
        check_len("matvec input", xs.len(), n, self.in_dim)?;
        if n == 0 {
            return Ok(Vec::new());
        }
        let bytes = self.expert_bytes(e)?;
        let chunk_rows = (CHUNK_ELEMS / self.in_dim).max(1);
        matvec_rows(
            Rows::Quant {
                dtype: self.dtype,
                bytes: &bytes,
                row_bytes: self.row_bytes,
            },
            self.in_dim,
            self.out_dim,
            xs,
            n,
            chunk_rows,
        )
    }
}

/// `read_exact` at an absolute offset through a shared handle — positional,
/// so parallel expert reads through one per-shard `File` never race on a
/// cursor (the twin of `pack.rs`'s helper).
fn read_exact_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    fn read_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
        #[cfg(windows)]
        return std::os::windows::fs::FileExt::seek_read(file, buf, offset);
        #[cfg(unix)]
        return std::os::unix::fs::FileExt::read_at(file, buf, offset);
    }
    while !buf.is_empty() {
        match read_at(file, buf, offset) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "file ended before the expert's bytes",
                ));
            }
            Ok(n) => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

fn check_len(what: &str, got: usize, n: usize, width: usize) -> Result<(), String> {
    match n.checked_mul(width) {
        Some(want) if want == got => Ok(()),
        _ => Err(format!("{what}: {got} values is not {n} rows x {width}")),
    }
}

/// The weight rows a matvec reads: quantized bytes or dequantized f32.
enum Rows<'a> {
    Quant {
        dtype: GgmlType,
        bytes: &'a [u8],
        row_bytes: usize,
    },
    Dense(&'a [f32]),
}

/// The one matvec kernel both the quantized and the cached path run, so the
/// two are bit-identical by construction: output element `(t, r)` is
/// `dot(row r, x_t)` whatever the chunking (`chunk_rows`) or threading.
fn matvec_rows(
    rows: Rows<'_>,
    in_dim: usize,
    out_dim: usize,
    xs: &[f32],
    n: usize,
    chunk_rows: usize,
) -> Result<Vec<f32>, String> {
    debug_assert_eq!(xs.len(), n * in_dim, "caller checked the input");
    if n == 0 {
        return Ok(Vec::new());
    }
    let chunk_rows = chunk_rows.max(1);
    // Row-major scratch (`rm[r*n + t]`) so each rayon task owns a contiguous
    // slice of whole weight rows; transposed to token-major at the end.
    let mut rm = vec![0f32; out_dim * n];
    let fill = |w: &[f32], dst: &mut [f32]| {
        for (row, d) in w.chunks_exact(in_dim).zip(dst.chunks_exact_mut(n)) {
            for (x, o) in xs.chunks_exact(in_dim).zip(d.iter_mut()) {
                *o = dot(row, x);
            }
        }
    };
    match rows {
        Rows::Quant {
            dtype,
            bytes,
            row_bytes,
        } => {
            assert_eq!(bytes.len(), out_dim * row_bytes, "one expert of rows");
            bytes
                .par_chunks(chunk_rows * row_bytes)
                .zip(rm.par_chunks_mut(chunk_rows * n))
                .try_for_each(|(qb, dst)| {
                    let w = gguf::dequantize(dtype, qb)?;
                    fill(&w, dst);
                    Ok::<(), String>(())
                })?;
        }
        Rows::Dense(w) => {
            assert_eq!(w.len(), out_dim * in_dim, "one expert of rows");
            w.par_chunks(chunk_rows * in_dim)
                .zip(rm.par_chunks_mut(chunk_rows * n))
                .for_each(|(w, dst)| fill(w, dst));
        }
    }
    if n == 1 {
        return Ok(rm);
    }
    let mut out = vec![0f32; n * out_dim];
    for (r, col) in rm.chunks_exact(n).enumerate() {
        for (t, &v) in col.iter().enumerate() {
            out[t * out_dim + r] = v;
        }
    }
    Ok(out)
}

/// Inner product with a fixed 8-lane accumulation order: autovectorizable
/// (the lanes are independent) and deterministic (the order depends only on
/// the length), which the bit-exactness claims above rest on.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0f32; 8];
    let (ca, ra) = a.as_chunks::<8>();
    let (cb, rb) = b.as_chunks::<8>();
    for (x, y) in ca.iter().zip(cb) {
        for ((lane, xl), yl) in acc.iter_mut().zip(x).zip(y) {
            *lane += xl * yl;
        }
    }
    let mut s = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// `x · sigmoid(x)`, written as ggml's `ggml_silu_f32` (`x / (1 + e^-x)`):
/// for very negative `x` the exponential overflows to inf and the result is
/// a clean `-0.0`, never NaN.
#[inline]
#[must_use]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Per-token routing decisions: `k` expert ids and weights per token,
/// `ids[t*k + s]` / `weights[t*k + s]`, slots in descending probability.
#[derive(Debug, Clone, PartialEq)]
pub struct Routing {
    pub n: usize,
    pub k: usize,
    pub ids: Vec<u32>,
    pub weights: Vec<f32>,
}

/// Softmax of one row, the way ggml's CPU `soft_max` does it: shift by the
/// max (so no exponential overflows), accumulate the normalizer in f64,
/// scale by its f32 reciprocal.
fn softmax_row(logits: &[f32], probs: &mut [f32]) {
    debug_assert_eq!(logits.len(), probs.len());
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f64;
    for (p, &l) in probs.iter_mut().zip(logits) {
        let v = (l - max).exp();
        sum += f64::from(v);
        *p = v;
    }
    #[allow(clippy::cast_possible_truncation)] // f64 -> f32 is the ggml scale
    let scale = (1.0 / sum) as f32;
    for p in probs.iter_mut() {
        *p *= scale;
    }
}

/// Divide selected probabilities by their sum, clamped below at
/// [`WEIGHT_SUM_FLOOR`] (llama.cpp `norm_w`). The sum is accumulated in f64
/// like ggml's `sum_rows`.
fn renormalize(top: &mut [f32]) {
    let sum: f64 = top.iter().map(|&p| f64::from(p)).sum();
    #[allow(clippy::cast_possible_truncation)]
    let sum = (sum as f32).max(WEIGHT_SUM_FLOOR);
    for w in top.iter_mut() {
        *w /= sum;
    }
}

/// Route `n` tokens: softmax over all `n_experts` logits per token, the `k`
/// most probable experts, weights renormalized over those `k` (no extra
/// scale — see the module doc).
///
/// Ties: when two probabilities are bit-equal at the selection boundary,
/// the LOWER expert id wins, and equal-probability slots are ordered by
/// ascending id. ggml's order for exact ties is its sort's business (not
/// verified to match); it only matters when two logits are bit-equal.
///
/// # Errors
/// On a mis-sized `logits`, `k` outside `1..=n_experts`, or a non-finite
/// logit (an inf/NaN router output would otherwise route silently to
/// arbitrary experts).
pub fn route(logits: &[f32], n: usize, n_experts: usize, k: usize) -> Result<Routing, String> {
    if n_experts == 0 || k == 0 || k > n_experts {
        return Err(format!("top-{k} of {n_experts} experts is not a routing"));
    }
    check_len("router logits", logits.len(), n, n_experts)?;
    if let Some(i) = logits.iter().position(|v| !v.is_finite()) {
        return Err(format!(
            "router logit for token {} expert {} is {}",
            i / n_experts,
            i % n_experts,
            logits[i]
        ));
    }
    let mut ids = Vec::with_capacity(n * k);
    let mut weights = Vec::with_capacity(n * k);
    let mut probs = vec![0f32; n_experts];
    // (prob, id), descending prob then ascending id; at most k long.
    let mut best: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
    for row in logits.chunks_exact(n_experts) {
        softmax_row(row, &mut probs);
        best.clear();
        for (id, &p) in probs.iter().enumerate() {
            // Ids arrive ascending, so a strict comparison keeps an earlier
            // (lower) id ahead of a later equal probability.
            if best.len() == k && p <= best[k - 1].0 {
                continue;
            }
            let pos = best.partition_point(|&(q, _)| q >= p);
            #[allow(clippy::cast_possible_truncation)] // n_experts is small
            best.insert(pos, (p, id as u32));
            best.truncate(k);
        }
        let start = weights.len();
        for &(p, id) in &best {
            ids.push(id);
            weights.push(p);
        }
        renormalize(&mut weights[start..]);
    }
    Ok(Routing { n, k, ids, weights })
}

/// One expert dequantized to f32 — the cache's unit. `gate`/`up` are
/// `[ffn x hidden]`, `down` is `[hidden x ffn]`, row-major like the banks.
#[derive(Debug)]
struct DenseExpert {
    hidden: usize,
    ffn: usize,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
}

/// Counters and occupancy of the f32 expert cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheStats {
    pub capacity_bytes: usize,
    pub used_bytes: usize,
    pub entries: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

#[derive(Debug)]
struct CacheEntry {
    weights: Arc<DenseExpert>,
    bytes: usize,
    last_touch: u64,
}

#[derive(Debug, Default)]
struct CacheState {
    entries: HashMap<(usize, usize), CacheEntry>,
    used_bytes: usize,
    clock: u64,
}

/// A bounded LRU of dequantized experts keyed `(layer, expert)`. It changes
/// speed only: a hit runs the same [`matvec_rows`] kernel over the same f32
/// row values the quantized path dequantizes chunk by chunk.
#[derive(Debug)]
struct ExpertCache {
    capacity_bytes: usize,
    state: Mutex<CacheState>,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
}

impl ExpertCache {
    fn new(capacity_bytes: usize) -> Self {
        assert!(capacity_bytes > 0, "a zero-capacity cache is no cache");
        Self {
            capacity_bytes,
            state: Mutex::new(CacheState::default()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }

    /// The cached expert, loading it on a miss. `None` when one expert is
    /// larger than the whole cache (the caller takes the quantized path
    /// rather than dequantizing an expert it could never keep).
    fn get_or_load(
        &self,
        key: (usize, usize),
        entry_bytes: usize,
        load: impl FnOnce() -> Result<DenseExpert, String>,
    ) -> Result<Option<Arc<DenseExpert>>, String> {
        if entry_bytes > self.capacity_bytes {
            return Ok(None);
        }
        {
            let mut st = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            st.clock += 1;
            let now = st.clock;
            if let Some(e) = st.entries.get_mut(&key) {
                e.last_touch = now;
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(Some(Arc::clone(&e.weights)));
            }
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        // Dequantize outside the lock: parallel experts must not serialize
        // behind one another's ~20 MB of dequant.
        let weights = Arc::new(load()?);
        let mut st = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        st.clock += 1;
        let now = st.clock;
        if let Some(e) = st.entries.get_mut(&key) {
            // A concurrent caller loaded the same expert first; keep theirs.
            e.last_touch = now;
            return Ok(Some(Arc::clone(&e.weights)));
        }
        while st.used_bytes + entry_bytes > self.capacity_bytes {
            let victim = st
                .entries
                .iter()
                .min_by_key(|(_, e)| e.last_touch)
                .map(|(k, _)| *k)
                .expect("over capacity implies at least one entry");
            let gone = st.entries.remove(&victim).expect("victim present");
            st.used_bytes -= gone.bytes;
            self.evictions.fetch_add(1, Ordering::Relaxed);
        }
        st.used_bytes += entry_bytes;
        st.entries.insert(
            key,
            CacheEntry {
                weights: Arc::clone(&weights),
                bytes: entry_bytes,
                last_touch: now,
            },
        );
        Ok(Some(weights))
    }

    fn stats(&self) -> CacheStats {
        let st = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        CacheStats {
            capacity_bytes: self.capacity_bytes,
            used_bytes: st.used_bytes,
            entries: st.entries.len(),
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
        }
    }
}

/// The cache ceiling from [`CACHE_ENV`] in bytes: GiB, fractional allowed,
/// `0` disables; unset, unparseable, negative or non-finite values fall back
/// to [`DEFAULT_CACHE_GB`].
#[must_use]
pub fn cache_bytes_from_env() -> usize {
    let gb = std::env::var(CACHE_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|g| g.is_finite() && *g >= 0.0)
        .unwrap_or(DEFAULT_CACHE_GB);
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let bytes = (gb * (1u64 << 30) as f64) as usize;
    bytes
}

/// The three banks of one block.
#[derive(Debug)]
pub struct LayerExperts {
    pub gate: ExpertBank,
    pub up: ExpertBank,
    pub down: ExpertBank,
}

impl LayerExperts {
    #[must_use]
    pub fn bank(&self, kind: BankKind) -> &ExpertBank {
        match kind {
            BankKind::Gate => &self.gate,
            BankKind::Up => &self.up,
            BankKind::Down => &self.down,
        }
    }
}

/// An expert's weights as one matvec source: quantized banks or a cache hit.
enum ExpertWeights<'a> {
    Quant(&'a LayerExperts, usize),
    Dense(&'a DenseExpert),
}

impl ExpertWeights<'_> {
    fn matvec(&self, kind: BankKind, xs: &[f32], n: usize) -> Result<Vec<f32>, String> {
        match self {
            Self::Quant(layer, e) => layer.bank(kind).matvec(*e, xs, n),
            Self::Dense(d) => {
                let (w, in_dim, out_dim) = match kind {
                    BankKind::Gate => (&d.gate, d.hidden, d.ffn),
                    BankKind::Up => (&d.up, d.hidden, d.ffn),
                    BankKind::Down => (&d.down, d.ffn, d.hidden),
                };
                check_len("expert input", xs.len(), n, in_dim)?;
                if n == 0 {
                    return Ok(Vec::new());
                }
                matvec_rows(
                    Rows::Dense(w),
                    in_dim,
                    out_dim,
                    xs,
                    n,
                    (CHUNK_ELEMS / in_dim).max(1),
                )
            }
        }
    }

    /// `down(silu(gate·x) · (up·x))` over `n` rows.
    fn ffn(&self, xs: &[f32], n: usize) -> Result<Vec<f32>, String> {
        // Diagnostic only (`nn::refarith`, off by default): quantize the
        // activation to the bank's ggml vec_dot grid first, as llama.cpp's
        // CPU mul_mat_id does. Only the quantized banks know their dtype,
        // which is why `expert_ffn` skips the f32 cache while it is on.
        let fq = |kind: BankKind, v: &[f32]| -> Option<Vec<f32>> {
            match self {
                Self::Quant(layer, _) if crate::nn::refarith::enabled() => {
                    let b = layer.bank(kind);
                    let mut o = v.to_vec();
                    crate::nn::refarith::fake_quant_rows(&mut o, b.in_dim(), b.dtype());
                    Some(o)
                }
                _ => None,
            }
        };
        let xg = fq(BankKind::Gate, xs);
        let xu = fq(BankKind::Up, xs);
        let (g, u) = rayon::join(
            || self.matvec(BankKind::Gate, xg.as_deref().unwrap_or(xs), n),
            || self.matvec(BankKind::Up, xu.as_deref().unwrap_or(xs), n),
        );
        let (mut h, u) = (g?, u?);
        for (hi, &ui) in h.iter_mut().zip(&u) {
            *hi = silu(*hi) * ui;
        }
        // Q5_1 banks dot against Q8_1, whose block sum `s` ggml rounds to
        // f16; the fake-quantized activations cannot carry that, so the
        // emulation adds `min · (rounded s − s)` per weight block afterwards.
        let min_rounding = match self {
            Self::Quant(layer, _)
                if crate::nn::refarith::enabled() && layer.down.dtype() == GgmlType::Q5_1 =>
            {
                Some(crate::nn::refarith::q8_1_min_term_rounding(
                    &h,
                    layer.down.in_dim(),
                ))
            }
            _ => None,
        };
        let h = fq(BankKind::Down, &h).unwrap_or(h);
        let mut y = self.matvec(BankKind::Down, &h, n)?;
        if let (Some(delta), Self::Quant(layer, e)) = (min_rounding, self) {
            let bank = &layer.down;
            let (blocks, out) = (bank.in_dim() / 32, bank.out_dim());
            let bytes = bank.expert_bytes(*e)?;
            // block_q5_1 = f16 d | f16 m | u32 qh | 16 B qs (24 bytes).
            let (blocks24, _) = bytes.as_chunks::<24>();
            let mins: Vec<f32> = blocks24
                .iter()
                .map(|blk| half::f16::from_le_bytes([blk[2], blk[3]]).to_f32())
                .collect();
            for (yt, dt) in y.chunks_mut(out).zip(delta.chunks(blocks)) {
                for (r, yr) in yt.iter_mut().enumerate() {
                    let m = &mins[r * blocks..(r + 1) * blocks];
                    *yr += m.iter().zip(dt).map(|(a, b)| a * b).sum::<f32>();
                }
            }
        }
        Ok(y)
    }
}

/// Every block's routed experts, read from the GGUF shards on demand, with
/// an optional bounded f32 cache. Host-slice API (see the module doc).
#[derive(Debug)]
pub struct RoutedExperts {
    layers: Vec<LayerExperts>,
    hidden: usize,
    ffn: usize,
    n_experts: usize,
    open_files: usize,
    cache: Option<ExpertCache>,
}

impl RoutedExperts {
    /// Locate `blk.L.ffn_{gate,up,down}_exps.weight` for `L in 0..n_layers`
    /// in a (possibly split) GGUF, holding ONE open handle per shard that
    /// carries a bank. `cache_bytes = 0` disables the f32 cache; pass
    /// [`cache_bytes_from_env`] for the operator's setting.
    ///
    /// # Errors
    /// A missing bank (named), a non-3-D shape, inconsistent geometry across
    /// banks or layers, a truncated shard, or an open failure.
    pub fn open(gguf: &GgufFile, n_layers: usize, cache_bytes: usize) -> Result<Self, String> {
        let mut files: HashMap<PathBuf, Arc<File>> = HashMap::new();
        let mut layers = Vec::with_capacity(n_layers);
        for layer in 0..n_layers {
            let mut bank = |kind: BankKind| -> Result<ExpertBank, String> {
                let name = kind.tensor_name(layer);
                let idx = gguf
                    .tensor_index(&name)
                    .ok_or_else(|| format!("GGUF has no {name}"))?;
                let info = &gguf.tensors[idx];
                let ne = match info.dims.as_slice() {
                    &[a, b, c] => [a, b, c].map(|d| usize::try_from(d).unwrap_or(usize::MAX)),
                    dims => return Err(format!("{name} has dims {dims:?}, want 3")),
                };
                // Split models keep each tensor's offset relative to its OWN
                // shard's payload base: resolve the file and base together.
                let (path, base) = gguf.payload_location(idx);
                let file = if let Some(f) = files.get(path) {
                    Arc::clone(f)
                } else {
                    let f =
                        Arc::new(File::open(path).map_err(|e| format!("{}: {e}", path.display()))?);
                    files.insert(path.to_path_buf(), Arc::clone(&f));
                    f
                };
                let b = ExpertBank::from_file(path, file, base + info.offset, info.dtype, ne)
                    .map_err(|e| format!("{name}: {e}"))?;
                if b.n_experts as u64 * b.expert_bytes_len() as u64 != info.byte_len() {
                    return Err(format!(
                        "{name}: bank geometry disagrees with the header's byte length"
                    ));
                }
                Ok(b)
            };
            let gate = bank(BankKind::Gate)?;
            let up = bank(BankKind::Up)?;
            let down = bank(BankKind::Down)?;
            layers.push(LayerExperts { gate, up, down });
        }
        let mut out = Self::from_layers(layers, cache_bytes)?;
        out.open_files = files.len();
        Ok(out)
    }

    /// Assemble from already-built banks (in-memory tests and toy models).
    ///
    /// # Errors
    /// No layers, or banks whose shapes disagree: gate and up must be
    /// `[hidden, ffn, n_experts]`, down `[ffn, hidden, n_experts]`, the same
    /// in every layer.
    pub fn from_layers(layers: Vec<LayerExperts>, cache_bytes: usize) -> Result<Self, String> {
        let first = layers.first().ok_or("no expert layers")?;
        let [hidden, ffn, n_experts] = first.gate.ne();
        for (l, layer) in layers.iter().enumerate() {
            let want = [
                (BankKind::Gate, [hidden, ffn, n_experts]),
                (BankKind::Up, [hidden, ffn, n_experts]),
                (BankKind::Down, [ffn, hidden, n_experts]),
            ];
            for (kind, ne) in want {
                let got = layer.bank(kind).ne();
                if got != ne {
                    return Err(format!(
                        "{}: ne {got:?}, expected {ne:?}",
                        kind.tensor_name(l)
                    ));
                }
            }
        }
        if n_experts > u32::MAX as usize {
            return Err(format!("{n_experts} experts do not fit a u32 id"));
        }
        Ok(Self {
            layers,
            hidden,
            ffn,
            n_experts,
            open_files: 0,
            cache: (cache_bytes > 0).then(|| ExpertCache::new(cache_bytes)),
        })
    }

    #[must_use]
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    #[must_use]
    pub fn n_experts(&self) -> usize {
        self.n_experts
    }

    /// Model width (expert input and output).
    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    /// Expert intermediate width.
    #[must_use]
    pub fn ffn_size(&self) -> usize {
        self.ffn
    }

    #[must_use]
    pub fn layer(&self, layer: usize) -> &LayerExperts {
        &self.layers[layer]
    }

    /// Distinct shard files held open (0 for in-memory banks).
    #[must_use]
    pub fn open_files(&self) -> usize {
        self.open_files
    }

    /// Cache counters, when the cache is enabled.
    #[must_use]
    pub fn cache_stats(&self) -> Option<CacheStats> {
        self.cache.as_ref().map(ExpertCache::stats)
    }

    /// f32 bytes of one dequantized expert (the cache's unit).
    #[must_use]
    pub fn dense_expert_bytes(&self) -> usize {
        3 * self.hidden * self.ffn * std::mem::size_of::<f32>()
    }

    fn check_expert(&self, layer: usize, expert: usize) -> Result<(), String> {
        if layer >= self.layers.len() {
            return Err(format!(
                "layer {layer} out of range ({})",
                self.layers.len()
            ));
        }
        if expert >= self.n_experts {
            return Err(format!("expert {expert} out of range ({})", self.n_experts));
        }
        Ok(())
    }

    /// One expert's FFN, `down(silu(gate·x) · (up·x))`, over `n` rows:
    /// `xs` is `[n x hidden]`, the result `[n x hidden]`. Uses the cache
    /// when enabled; the result is bit-identical either way.
    ///
    /// # Errors
    /// Out-of-range layer/expert, a mis-sized `xs`, or an I/O failure.
    pub fn expert_ffn(
        &self,
        layer: usize,
        expert: usize,
        xs: &[f32],
        n: usize,
    ) -> Result<Vec<f32>, String> {
        self.check_expert(layer, expert)?;
        check_len("expert input", xs.len(), n, self.hidden)?;
        if n == 0 {
            return Ok(Vec::new());
        }
        let banks = &self.layers[layer];
        // The emulation needs the quantized banks' dtypes (see `ffn`).
        if let Some(cache) = self
            .cache
            .as_ref()
            .filter(|_| !crate::nn::refarith::enabled())
        {
            let dense = cache.get_or_load((layer, expert), self.dense_expert_bytes(), || {
                let (gu, down) = rayon::join(
                    || {
                        rayon::join(
                            || banks.gate.dequantize_expert(expert),
                            || banks.up.dequantize_expert(expert),
                        )
                    },
                    || banks.down.dequantize_expert(expert),
                );
                Ok(DenseExpert {
                    hidden: self.hidden,
                    ffn: self.ffn,
                    gate: gu.0?,
                    up: gu.1?,
                    down: down?,
                })
            })?;
            if let Some(dense) = dense {
                return ExpertWeights::Dense(&dense).ffn(xs, n);
            }
        }
        ExpertWeights::Quant(banks, expert).ffn(xs, n)
    }

    /// The routed half of the MoE block for `n` tokens: route with
    /// [`route`] (`router_logits` is `[n x n_experts]`, top-`k`), then
    /// [`Self::apply_routing`]. Returns `[n x hidden]`; the integrator adds
    /// `shared · sigmoid(shared_gate)`.
    ///
    /// # Errors
    /// As [`route`] and [`Self::apply_routing`].
    pub fn routed_moe(
        &self,
        layer: usize,
        router_logits: &[f32],
        xs: &[f32],
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, String> {
        let routing = route(router_logits, n, self.n_experts, k)?;
        self.apply_routing(layer, &routing, xs)
    }

    /// Weighted sum of the routed experts' outputs. Token rows are grouped
    /// per selected expert so each expert runs ONCE per call on exactly the
    /// rows routed to it; experts run in parallel; results are scatter-added
    /// in ascending expert id (a fixed order, so the sum is reproducible).
    ///
    /// Addition order differs from llama.cpp's (which sums in slot order);
    /// that is rounding-level, far below the difference its quantized-
    /// activation dot products already introduce.
    ///
    /// Memory: every hit expert's `[m x hidden]` output is held until the
    /// scatter, about `n · k · hidden · 4` bytes — chunk very long prefills.
    ///
    /// # Errors
    /// Out-of-range layer or expert id, a mis-sized `xs`, or an I/O failure.
    pub fn apply_routing(
        &self,
        layer: usize,
        routing: &Routing,
        xs: &[f32],
    ) -> Result<Vec<f32>, String> {
        let (n, k, e_dim) = (routing.n, routing.k, self.hidden);
        if layer >= self.layers.len() {
            return Err(format!(
                "layer {layer} out of range ({})",
                self.layers.len()
            ));
        }
        check_len("routing ids", routing.ids.len(), n, k)?;
        check_len("routing weights", routing.weights.len(), n, k)?;
        check_len("moe input", xs.len(), n, e_dim)?;
        let mut members: Vec<Vec<(usize, f32)>> = vec![Vec::new(); self.n_experts];
        for t in 0..n {
            for s in 0..k {
                let e = routing.ids[t * k + s] as usize;
                if e >= self.n_experts {
                    return Err(format!(
                        "routed expert {e} out of range ({})",
                        self.n_experts
                    ));
                }
                members[e].push((t, routing.weights[t * k + s]));
            }
        }
        let hit: Vec<(usize, Vec<(usize, f32)>)> = members
            .into_iter()
            .enumerate()
            .filter(|(_, m)| !m.is_empty())
            .collect();
        let outputs: Vec<Result<Vec<f32>, String>> = hit
            .par_iter()
            .map(|(e, rows)| {
                let mut gathered = Vec::with_capacity(rows.len() * e_dim);
                for &(t, _) in rows {
                    gathered.extend_from_slice(&xs[t * e_dim..(t + 1) * e_dim]);
                }
                self.expert_ffn(layer, *e, &gathered, rows.len())
            })
            .collect();
        let mut out = vec![0f32; n * e_dim];
        for ((_, rows), y) in hit.iter().zip(outputs) {
            let y = y?;
            for (i, &(t, w)) in rows.iter().enumerate() {
                let dst = &mut out[t * e_dim..(t + 1) * e_dim];
                for (o, &v) in dst.iter_mut().zip(&y[i * e_dim..(i + 1) * e_dim]) {
                    *o += w * v;
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SplitMix64: a tiny deterministic generator for synthetic quant bits.
    struct Rng(u64);

    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn byte(&mut self) -> u8 {
            (self.next_u64() >> 56) as u8
        }
        /// Uniform in `[lo, hi)`.
        fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
            let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
            lo + (hi - lo) * u
        }
        fn uniform_vec(&mut self, len: usize) -> Vec<f32> {
            (0..len).map(|_| self.uniform(-1.0, 1.0)).collect()
        }
    }

    fn f16_bytes(v: f32) -> [u8; 2] {
        half::f16::from_f32(v).to_bits().to_le_bytes()
    }

    /// `blocks` sane blocks of `dtype`: finite, modest f16 scales and random
    /// quant bits, so dequantized weights are O(0.01-1).
    fn synth_blocks(dtype: GgmlType, blocks: usize, rng: &mut Rng) -> Vec<u8> {
        let mut out = Vec::new();
        for _ in 0..blocks {
            match dtype {
                GgmlType::Q8_0 => {
                    out.extend(f16_bytes(rng.uniform(0.002, 0.02)));
                    out.extend((0..32).map(|_| rng.byte()));
                }
                GgmlType::Q5_1 => {
                    out.extend(f16_bytes(rng.uniform(0.005, 0.05)));
                    out.extend(f16_bytes(rng.uniform(-0.4, 0.0)));
                    out.extend((0..20).map(|_| rng.byte()));
                }
                GgmlType::Q4_K => {
                    out.extend(f16_bytes(rng.uniform(0.0005, 0.003)));
                    out.extend(f16_bytes(rng.uniform(0.0005, 0.003)));
                    out.extend((0..140).map(|_| rng.byte()));
                }
                GgmlType::Q5_K => {
                    out.extend(f16_bytes(rng.uniform(0.0003, 0.002)));
                    out.extend(f16_bytes(rng.uniform(0.0003, 0.002)));
                    out.extend((0..172).map(|_| rng.byte()));
                }
                other => panic!("no synthetic blocks for {other:?}"),
            }
        }
        assert_eq!(out.len(), blocks * dtype.bytes_per_block() as usize);
        out
    }

    fn synth_bank(dtype: GgmlType, ne: [usize; 3], rng: &mut Rng) -> ExpertBank {
        let blocks = ne[0] / dtype.block_size() as usize * ne[1] * ne[2];
        ExpertBank::from_bytes(dtype, ne, synth_blocks(dtype, blocks, rng)).expect("sane bank")
    }

    /// Reference `W x` in f64 over the whole dequantized expert.
    fn ref_matvec(w: &[f32], in_dim: usize, xs: &[f32]) -> (Vec<f64>, Vec<f64>) {
        let mut out = Vec::new();
        let mut mag = Vec::new();
        for x in xs.chunks_exact(in_dim) {
            for row in w.chunks_exact(in_dim) {
                let terms = row
                    .iter()
                    .zip(x)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b));
                out.push(terms.clone().sum());
                mag.push(terms.map(|v| v * v).sum::<f64>().sqrt());
            }
        }
        (out, mag)
    }

    /// Relative closeness at 1e-5, measured against the larger of the value
    /// and the root-sum-square of its terms: a sum that cancels to near zero
    /// cannot be held to a tolerance its own rounding cannot meet.
    fn assert_close(got: &[f32], want: &[f64], mag: &[f64], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, ((&g, &w), &m)) in got.iter().zip(want).zip(mag).enumerate() {
            let scale = w.abs().max(m).max(1e-30);
            let rel = (f64::from(g) - w).abs() / scale;
            assert!(rel <= 1e-5, "{what}[{i}]: {g} vs {w} (rel {rel:e})");
        }
    }

    fn check_matvec_against_dequantized_expert(dtype: GgmlType, in_dim: usize) {
        let mut rng = Rng(0x5eed ^ in_dim as u64 ^ (dtype.bytes_per_block() << 20));
        // Two full production chunks plus a ragged tail.
        let out_dim = CHUNK_ELEMS / in_dim * 2 + 3;
        let blocks_per_expert = in_dim / dtype.block_size() as usize * out_dim;
        // Each expert's bytes built on their own, so the reference does not
        // lean on the bank's own offset arithmetic.
        let per_expert: Vec<Vec<u8>> = (0..3)
            .map(|_| synth_blocks(dtype, blocks_per_expert, &mut rng))
            .collect();
        let bank = ExpertBank::from_bytes(dtype, [in_dim, out_dim, 3], per_expert.concat())
            .expect("sane bank");
        let n = 4;
        let xs = rng.uniform_vec(n * in_dim);
        for e in [0, 2] {
            let w = gguf::dequantize(dtype, &per_expert[e]).unwrap();
            assert!(w.iter().all(|v| v.is_finite()), "synthetic weights finite");
            let (want, mag) = ref_matvec(&w, in_dim, &xs);
            let got = bank.matvec(e, &xs, n).expect("matvec");
            assert_close(&got, &want, &mag, &format!("{dtype:?} expert {e}"));
        }
    }

    #[test]
    fn q8_0_matvec_equals_dequantize_then_matmul() {
        check_matvec_against_dequantized_expert(GgmlType::Q8_0, 96);
    }

    #[test]
    fn q5_1_matvec_equals_dequantize_then_matmul() {
        check_matvec_against_dequantized_expert(GgmlType::Q5_1, 64);
    }

    #[test]
    fn q4_k_matvec_equals_dequantize_then_matmul() {
        check_matvec_against_dequantized_expert(GgmlType::Q4_K, 512);
    }

    #[test]
    fn q5_k_matvec_equals_dequantize_then_matmul() {
        check_matvec_against_dequantized_expert(GgmlType::Q5_K, 256);
    }

    #[test]
    fn chunking_and_the_dense_path_do_not_change_a_single_bit() {
        // The cache's bit-exactness rests on this: every element is one
        // `dot` of one row, whatever the chunk size or weight source.
        let mut rng = Rng(7);
        let (in_dim, out_dim, n) = (256, 37, 3);
        let bank = synth_bank(GgmlType::Q4_K, [in_dim, out_dim, 2], &mut rng);
        let xs = rng.uniform_vec(n * in_dim);
        let bytes = bank.expert_bytes(1).unwrap();
        let run =
            |rows: Rows<'_>, chunk| matvec_rows(rows, in_dim, out_dim, &xs, n, chunk).unwrap();
        let quant = |chunk| {
            run(
                Rows::Quant {
                    dtype: GgmlType::Q4_K,
                    bytes: &bytes,
                    row_bytes: bank.row_bytes(),
                },
                chunk,
            )
        };
        let base = quant(1);
        let dense = bank.dequantize_expert(1).unwrap();
        for got in [
            quant(5),
            quant(37),
            quant(1000),
            run(Rows::Dense(&dense), 4),
        ] {
            assert!(
                base.iter()
                    .zip(&got)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "bit-identical across chunking and weight source"
            );
        }
        assert_eq!(base, bank.matvec(1, &xs, n).unwrap());
    }

    #[test]
    fn a_row_that_is_not_whole_blocks_is_refused() {
        // 640-wide rows cannot be K-quant (256-element blocks): slicing such
        // a bank by row would silently misalign.
        let err = ExpertBank::from_bytes(GgmlType::Q4_K, [640, 4, 2], vec![0; 1024]).unwrap_err();
        assert!(err.contains("not whole"), "{err}");
        let err = ExpertBank::from_bytes(GgmlType::Q8_0, [64, 4, 2], vec![0; 10]).unwrap_err();
        assert!(err.contains("bytes"), "{err}");
    }

    #[test]
    fn the_router_picks_the_most_probable_and_renormalizes_without_scale() {
        let logits = [0.0f32, 3.0, 1.0, 2.5, -2.0, 2.0, 0.5, 2.9];
        let r = route(&logits, 1, 8, 3).unwrap();
        assert_eq!(r.ids, vec![1, 7, 3], "descending probability");
        let e = |l: f64| l.exp();
        let z = e(3.0) + e(2.9) + e(2.5);
        let want = [e(3.0) / z, e(2.9) / z, e(2.5) / z];
        for (g, w) in r.weights.iter().zip(want) {
            assert!((f64::from(*g) - w).abs() < 1e-6, "{g} vs {w}");
        }
        let sum: f32 = r.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "weights sum to 1: {sum}");
    }

    #[test]
    fn router_ties_go_to_the_lowest_expert_id() {
        let r = route(&[0.5; 8], 1, 8, 3).unwrap();
        assert_eq!(r.ids, vec![0, 1, 2]);
        assert!(r.weights.iter().all(|w| (w - 1.0 / 3.0).abs() < 1e-6));
        // A tie AT the boundary keeps the lower id.
        let r = route(&[1.0, 0.0, 2.0, 1.0], 1, 4, 2).unwrap();
        assert_eq!(r.ids, vec![2, 0]);
    }

    #[test]
    fn router_is_stable_for_huge_logits_and_many_tokens() {
        let mut logits = vec![-1000.0f32; 3 * 16];
        logits[5] = 1000.0; // token 0
        logits[16 + 9] = 999.0; // token 1
        logits[16 + 3] = 998.0;
        logits[32 + 15] = 30.0; // token 2
        let r = route(&logits, 3, 16, 2).unwrap();
        assert_eq!(r.ids[0], 5);
        assert_eq!(&r.ids[2..4], &[9, 3]);
        assert_eq!(r.ids[4], 15);
        for t in 0..3 {
            let w = &r.weights[t * 2..t * 2 + 2];
            assert!(w.iter().all(|v| v.is_finite()), "finite: {w:?}");
            assert!((w[0] + w[1] - 1.0).abs() < 1e-6, "sum to 1: {w:?}");
        }
        let e = 1f64.exp();
        assert!((f64::from(r.weights[2]) - e / (e + 1.0)).abs() < 1e-6);
    }

    #[test]
    fn router_refuses_non_finite_logits_and_bad_k() {
        let err = route(&[0.0, f32::NAN, 1.0, 2.0], 2, 2, 1).unwrap_err();
        assert!(err.contains("token 0 expert 1"), "{err}");
        assert!(route(&[0.0; 4], 1, 4, 0).is_err());
        assert!(route(&[0.0; 4], 1, 4, 5).is_err());
        assert!(route(&[0.0; 5], 1, 4, 2).is_err());
    }

    #[test]
    fn a_vanishing_weight_sum_is_clamped_like_llama_cpp() {
        let mut w = [1e-6f32, 2e-6];
        renormalize(&mut w);
        assert!((w[0] - 1e-6 / WEIGHT_SUM_FLOOR).abs() < 1e-9);
        let mut w = [0.25f32, 0.25];
        renormalize(&mut w);
        assert_eq!(w, [0.5, 0.5]);
    }

    const TOY_HIDDEN: usize = 256;
    const TOY_FFN: usize = 32;
    const TOY_EXPERTS: usize = 8;

    /// Deterministic toy layers: gate/up Q4_K (256-wide rows), down Q5_1
    /// (32-wide rows) on layer 0 and Q8_0 on layer 1 — the real model's mix.
    fn toy_layers(seed: u64) -> Vec<LayerExperts> {
        let mut rng = Rng(seed);
        let fwd = [TOY_HIDDEN, TOY_FFN, TOY_EXPERTS];
        let back = [TOY_FFN, TOY_HIDDEN, TOY_EXPERTS];
        [GgmlType::Q5_1, GgmlType::Q8_0]
            .into_iter()
            .map(|down| LayerExperts {
                gate: synth_bank(GgmlType::Q4_K, fwd, &mut rng),
                up: synth_bank(GgmlType::Q4_K, fwd, &mut rng),
                down: synth_bank(down, back, &mut rng),
            })
            .collect()
    }

    /// Clone a layer's banks into fresh in-memory banks.
    fn clone_layers(src: &[LayerExperts]) -> Vec<LayerExperts> {
        let copy = |b: &ExpertBank| {
            let bytes: Vec<u8> = (0..b.n_experts())
                .flat_map(|e| b.expert_bytes(e).unwrap().into_owned())
                .collect();
            ExpertBank::from_bytes(b.dtype(), b.ne(), bytes).unwrap()
        };
        src.iter()
            .map(|l| LayerExperts {
                gate: copy(&l.gate),
                up: copy(&l.up),
                down: copy(&l.down),
            })
            .collect()
    }

    #[test]
    fn routed_moe_equals_the_dense_mask_reference() {
        let experts = RoutedExperts::from_layers(toy_layers(11), 0).unwrap();
        let mut rng = Rng(12);
        let n = 5;
        let xs = rng.uniform_vec(n * TOY_HIDDEN);
        let logits: Vec<f32> = (0..n * TOY_EXPERTS)
            .map(|_| rng.uniform(-2.0, 2.0))
            .collect();
        for layer in 0..2 {
            let got = experts.routed_moe(layer, &logits, &xs, n, 2).unwrap();
            // Reference: EVERY expert on EVERY token, in f64 from the whole
            // dequantized matrices, masked by an independently computed
            // top-2 renormalized softmax.
            let banks = experts.layer(layer);
            let mut want = vec![0f64; n * TOY_HIDDEN];
            let mut mag = vec![0f64; n * TOY_HIDDEN];
            for t in 0..n {
                let row = &logits[t * TOY_EXPERTS..(t + 1) * TOY_EXPERTS];
                let z: f64 = row.iter().map(|&l| f64::from(l).exp()).sum();
                let mut order: Vec<(f64, usize)> = row
                    .iter()
                    .enumerate()
                    .map(|(e, &l)| (f64::from(l).exp() / z, e))
                    .collect();
                order.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                let top_sum = order[0].0 + order[1].0;
                let mut mask = [0f64; TOY_EXPERTS];
                for &(p, e) in &order[..2] {
                    mask[e] = p / top_sum;
                }
                let x = &xs[t * TOY_HIDDEN..(t + 1) * TOY_HIDDEN];
                #[allow(clippy::needless_range_loop)]
                // `e` is the expert id the bank reads, not just an index
                for e in 0..TOY_EXPERTS {
                    let deq = |b: &ExpertBank| b.dequantize_expert(e).unwrap();
                    let (g, _) = ref_matvec(&deq(&banks.gate), TOY_HIDDEN, x);
                    let (u, _) = ref_matvec(&deq(&banks.up), TOY_HIDDEN, x);
                    let h: Vec<f32> = g
                        .iter()
                        .zip(&u)
                        .map(|(&g, &u)| (g / (1.0 + (-g).exp()) * u) as f32)
                        .collect();
                    let (y, ym) = ref_matvec(&deq(&banks.down), TOY_FFN, &h);
                    for j in 0..TOY_HIDDEN {
                        want[t * TOY_HIDDEN + j] += mask[e] * y[j];
                        mag[t * TOY_HIDDEN + j] += mask[e] * ym[j];
                    }
                }
            }
            // The reference runs its silu in f64 then narrows, so allow the
            // f32 rounding of the hidden activations through `mag`.
            assert_close(&got, &want, &mag, &format!("layer {layer} routed"));
        }
    }

    #[test]
    fn a_token_routes_bit_identically_alone_or_in_a_batch() {
        // The chunked-prefill invariant, restricted to this block.
        let experts = RoutedExperts::from_layers(toy_layers(21), 0).unwrap();
        let mut rng = Rng(22);
        let n = 6;
        let xs = rng.uniform_vec(n * TOY_HIDDEN);
        let logits: Vec<f32> = (0..n * TOY_EXPERTS)
            .map(|_| rng.uniform(-3.0, 3.0))
            .collect();
        let batch = experts.routed_moe(0, &logits, &xs, n, 3).unwrap();
        for t in 0..n {
            let one = experts
                .routed_moe(
                    0,
                    &logits[t * TOY_EXPERTS..(t + 1) * TOY_EXPERTS],
                    &xs[t * TOY_HIDDEN..(t + 1) * TOY_HIDDEN],
                    1,
                    3,
                )
                .unwrap();
            let row = &batch[t * TOY_HIDDEN..(t + 1) * TOY_HIDDEN];
            assert!(
                row.iter()
                    .zip(&one)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "token {t}"
            );
        }
    }

    #[test]
    fn the_cache_is_bit_exact_through_hits_and_evictions() {
        let off = RoutedExperts::from_layers(toy_layers(31), 0).unwrap();
        let per_expert = off.dense_expert_bytes();
        let big = RoutedExperts::from_layers(clone_layers(&off.layers), 64 * per_expert).unwrap();
        // Room for exactly two experts: k=2 routing over 8 forces evictions.
        let tiny = RoutedExperts::from_layers(clone_layers(&off.layers), 2 * per_expert).unwrap();
        assert!(off.cache_stats().is_none());
        let mut rng = Rng(32);
        let n = 7;
        for round in 0..3 {
            let xs = rng.uniform_vec(n * TOY_HIDDEN);
            let logits: Vec<f32> = (0..n * TOY_EXPERTS)
                .map(|_| rng.uniform(-2.0, 2.0))
                .collect();
            for layer in 0..2 {
                let base = off.routed_moe(layer, &logits, &xs, n, 2).unwrap();
                for cached in [&big, &tiny] {
                    let got = cached.routed_moe(layer, &logits, &xs, n, 2).unwrap();
                    assert!(
                        base.iter()
                            .zip(&got)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "round {round} layer {layer}: cache changed the result"
                    );
                }
            }
        }
        let b = big.cache_stats().unwrap();
        assert!(b.hits > 0 && b.misses > 0, "{b:?}");
        assert_eq!(b.evictions, 0);
        let t = tiny.cache_stats().unwrap();
        assert!(t.evictions > 0, "{t:?}");
        assert!(t.used_bytes <= t.capacity_bytes && t.entries <= 2, "{t:?}");
        // An expert bigger than the whole cache bypasses it, still exact.
        let small = RoutedExperts::from_layers(clone_layers(&off.layers), per_expert - 1).unwrap();
        let xs = rng.uniform_vec(TOY_HIDDEN);
        assert_eq!(
            off.expert_ffn(1, 7, &xs, 1).unwrap(),
            small.expert_ffn(1, 7, &xs, 1).unwrap()
        );
        assert_eq!(small.cache_stats().unwrap().entries, 0);
    }

    #[test]
    fn cache_env_parses_gib_and_zero_disables() {
        // Parsing only (no env mutation: tests run in parallel threads).
        let parse = |v: &str| {
            v.trim()
                .parse::<f64>()
                .ok()
                .filter(|g| g.is_finite() && *g >= 0.0)
                .unwrap_or(DEFAULT_CACHE_GB)
        };
        assert_eq!(parse("0"), 0.0);
        assert_eq!(parse(" 0.5 "), 0.5);
        assert_eq!(parse("-1"), DEFAULT_CACHE_GB);
        assert_eq!(parse("lots"), DEFAULT_CACHE_GB);
        assert_eq!(DEFAULT_CACHE_GB, 0.0, "the measured default is off");
    }

    // ---- GGUF-backed banks ------------------------------------------------

    fn push_str(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    fn type_id(dtype: GgmlType) -> u32 {
        match dtype {
            GgmlType::F32 => 0,
            GgmlType::Q5_1 => 7,
            GgmlType::Q8_0 => 8,
            GgmlType::Q4_K => 12,
            GgmlType::Q5_K => 13,
            other => panic!("no test id for {other:?}"),
        }
    }

    /// One GGUF v3 shard: `kvs` are (key, u32) pairs plus the architecture;
    /// tensors are laid out in order at 32-aligned offsets.
    fn gguf_shard(
        kvs: &[(&str, u32)],
        tensors: &[(String, [usize; 3], GgmlType, Vec<u8>)],
    ) -> Vec<u8> {
        let mut kv = Vec::new();
        push_str(&mut kv, "general.architecture");
        kv.extend_from_slice(&8u32.to_le_bytes());
        push_str(&mut kv, "qwen4exp");
        for (k, v) in kvs {
            push_str(&mut kv, k);
            kv.extend_from_slice(&4u32.to_le_bytes());
            kv.extend_from_slice(&v.to_le_bytes());
        }
        let mut table = Vec::new();
        let mut payload = Vec::new();
        for (name, ne, dtype, bytes) in tensors {
            push_str(&mut table, name);
            table.extend_from_slice(&3u32.to_le_bytes());
            for d in ne {
                table.extend_from_slice(&(*d as u64).to_le_bytes());
            }
            table.extend_from_slice(&type_id(*dtype).to_le_bytes());
            table.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            payload.extend_from_slice(bytes);
            payload.resize(payload.len().div_ceil(32) * 32, 0);
        }
        let mut buf = b"GGUF".to_vec();
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(kvs.len() as u64 + 1).to_le_bytes());
        buf.extend_from_slice(&kv);
        buf.extend_from_slice(&table);
        buf.resize(buf.len().div_ceil(32) * 32, 0);
        buf.extend_from_slice(&payload);
        buf
    }

    fn bank_tensor(
        layer: usize,
        kind: BankKind,
        b: &ExpertBank,
    ) -> (String, [usize; 3], GgmlType, Vec<u8>) {
        let bytes = (0..b.n_experts())
            .flat_map(|e| b.expert_bytes(e).unwrap().into_owned())
            .collect();
        (kind.tensor_name(layer), b.ne(), b.dtype(), bytes)
    }

    #[test]
    fn open_locates_every_bank_across_shards_with_one_handle_each() {
        let mem = RoutedExperts::from_layers(toy_layers(41), 0).unwrap();
        // A filler tensor first so no bank starts at payload offset 0, and
        // layer 1 in the second shard at ITS OWN offsets.
        let filler = (
            "blk.0.ffn_gate_inp.weight".to_string(),
            [4, 2, 1],
            GgmlType::F32,
            vec![1u8; 32],
        );
        let mut s0 = vec![filler];
        let mut s1 = Vec::new();
        for (layer, shard) in [(0, &mut s0), (1, &mut s1)] {
            for kind in BankKind::ALL {
                shard.push(bank_tensor(layer, kind, mem.layer(layer).bank(kind)));
            }
        }
        let split = |no| {
            [
                ("split.count", 2),
                ("split.no", no),
                ("split.tensors.count", 7),
            ]
        };
        let dir =
            std::env::temp_dir().join(format!("mummu-qwen4exp-experts-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p0 = dir.join("toy-00001-of-00002.gguf");
        std::fs::write(&p0, gguf_shard(&split(0), &s0)).unwrap();
        std::fs::write(
            dir.join("toy-00002-of-00002.gguf"),
            gguf_shard(&split(1), &s1),
        )
        .unwrap();

        let result = std::panic::catch_unwind(|| {
            let gguf = GgufFile::open_sharded(&p0).expect("toy split set parses");
            let disk = RoutedExperts::open(&gguf, 2, 0).expect("banks located");
            assert_eq!(disk.open_files(), 2, "one handle per shard");
            assert_eq!(
                (disk.hidden_size(), disk.ffn_size(), disk.n_experts()),
                (256, 32, 8)
            );
            for layer in 0..2 {
                for kind in BankKind::ALL {
                    let (d, m) = (disk.layer(layer).bank(kind), mem.layer(layer).bank(kind));
                    assert_eq!((d.dtype(), d.ne()), (m.dtype(), m.ne()));
                    let (path, offset) = d.location().expect("file-backed");
                    assert!(path.ends_with(format!("toy-0000{}-of-00002.gguf", layer + 1)));
                    assert!(offset > 0);
                    for e in [0, 3, 7] {
                        assert_eq!(d.expert_bytes(e).unwrap(), m.expert_bytes(e).unwrap());
                    }
                }
            }
            let mut rng = Rng(42);
            let n = 3;
            let xs = rng.uniform_vec(n * TOY_HIDDEN);
            let logits: Vec<f32> = (0..n * TOY_EXPERTS)
                .map(|_| rng.uniform(-2.0, 2.0))
                .collect();
            for layer in 0..2 {
                assert_eq!(
                    disk.routed_moe(layer, &logits, &xs, n, 2).unwrap(),
                    mem.routed_moe(layer, &logits, &xs, n, 2).unwrap()
                );
            }
            let err = RoutedExperts::open(&gguf, 3, 0).unwrap_err();
            assert!(
                err.contains("blk.2.ffn_gate_exps.weight"),
                "names the missing bank: {err}"
            );
        });
        let _ = std::fs::remove_dir_all(&dir);
        if let Err(panic) = result {
            std::panic::resume_unwind(panic);
        }
    }

    #[test]
    fn the_expert_store_can_be_shared_across_threads() {
        // The integrator holds one store for the model and calls it from
        // rayon/tokio workers; this must stay true as fields are added.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RoutedExperts>();
        assert_send_sync::<ExpertBank>();
    }

    #[test]
    fn mismatched_bank_shapes_are_refused() {
        let mut layers = toy_layers(51);
        let mut rng = Rng(52);
        // Down with the gate's orientation: [hidden, ffn] instead of [ffn, hidden].
        layers[1].down = synth_bank(GgmlType::Q4_K, [TOY_HIDDEN, TOY_FFN, TOY_EXPERTS], &mut rng);
        let err = RoutedExperts::from_layers(layers, 0).unwrap_err();
        assert!(err.contains("blk.1.ffn_down_exps.weight"), "{err}");
    }

    // ---- The shipped model --------------------------------------------------

    /// Real Qwen3.8-Flash-Next shards (`MUMMU_QWEN4EXP_DIR` = the directory
    /// holding `*-00001-of-00004.gguf`; point it at NVMe). Locates all 144
    /// banks, checks the shipped dtype mix, runs the first and last expert of
    /// the first and last layer against a dequantize-then-matmul reference,
    /// and times the routed path. RAM stays small: one expert at a time,
    /// plus one 5 MB router matrix. Run with `--release` for timings.
    #[test]
    #[ignore = "needs the 111 GB model; set MUMMU_QWEN4EXP_DIR"]
    fn shipped_expert_banks_route_and_time() {
        use std::time::Instant;
        let Some(dir) = std::env::var_os("MUMMU_QWEN4EXP_DIR") else {
            eprintln!("skipping: MUMMU_QWEN4EXP_DIR is unset");
            return;
        };
        let first_shard = std::fs::read_dir(&dir)
            .expect("model dir")
            .filter_map(Result::ok)
            .map(|e| e.path())
            .find(|p| p.to_string_lossy().ends_with("-00001-of-00004.gguf"))
            .expect("first shard in MUMMU_QWEN4EXP_DIR");
        let gguf = GgufFile::open_sharded(&first_shard).expect("split set");
        let t0 = Instant::now();
        let experts = RoutedExperts::open(&gguf, 48, 0).expect("all banks");
        eprintln!(
            "open: {} layers, {} shard handles, {:.1} ms",
            experts.n_layers(),
            experts.open_files(),
            t0.elapsed().as_secs_f64() * 1e3
        );
        assert_eq!(
            (
                experts.hidden_size(),
                experts.ffn_size(),
                experts.n_experts()
            ),
            (2560, 640, 512)
        );
        let mut counts: HashMap<(BankKind, String), usize> = HashMap::new();
        for l in 0..48 {
            let layer = experts.layer(l);
            assert_eq!(
                layer.gate.dtype(),
                layer.up.dtype(),
                "layer {l} gate/up share a dtype"
            );
            for kind in BankKind::ALL {
                let b = layer.bank(kind);
                *counts
                    .entry((kind, format!("{:?}", b.dtype())))
                    .or_default() += 1;
            }
        }
        let count = |k, d: &str| counts.get(&(k, d.to_string())).copied().unwrap_or(0);
        eprintln!("dtype mix: {counts:?}");
        // Spec §0: gate/up Q4_K on 47 layers and Q5_K on 1; down Q5_1 on 43, Q8_0 on 5.
        assert_eq!(
            (count(BankKind::Gate, "Q4_K"), count(BankKind::Gate, "Q5_K")),
            (47, 1)
        );
        assert_eq!(
            (count(BankKind::Up, "Q4_K"), count(BankKind::Up, "Q5_K")),
            (47, 1)
        );
        assert_eq!(
            (count(BankKind::Down, "Q5_1"), count(BankKind::Down, "Q8_0")),
            (43, 5)
        );

        let mut rng = Rng(0xF1A5);
        let unit_rms = |rng: &mut Rng, n: usize| {
            let mut xs: Vec<f32> = rng.uniform_vec(n * 2560);
            for x in xs.as_chunks_mut::<2560>().0 {
                let rms = (x.iter().map(|v| v * v).sum::<f32>() / 2560.0).sqrt();
                x.iter_mut().for_each(|v| *v /= rms);
            }
            xs
        };
        let x1 = unit_rms(&mut rng, 1);
        for (l, e) in [(0, 0), (47, 511)] {
            let t = Instant::now();
            let y = experts.expert_ffn(l, e, &x1, 1).expect("expert ffn");
            let cold = t.elapsed().as_secs_f64() * 1e3;
            assert_eq!(y.len(), 2560);
            assert!(
                y.iter().all(|v| v.is_finite()),
                "layer {l} expert {e} finite"
            );
            let banks = experts.layer(l);
            let (g, _) = ref_matvec(&banks.gate.dequantize_expert(e).unwrap(), 2560, &x1);
            let (u, _) = ref_matvec(&banks.up.dequantize_expert(e).unwrap(), 2560, &x1);
            let h: Vec<f32> = g
                .iter()
                .zip(&u)
                .map(|(&g, &u)| (g / (1.0 + (-g).exp()) * u) as f32)
                .collect();
            let (want, mag) = ref_matvec(&banks.down.dequantize_expert(e).unwrap(), 640, &h);
            assert_close(&y, &want, &mag, &format!("layer {l} expert {e}"));
            let norm = (y.iter().map(|v| v * v).sum::<f32>()).sqrt();
            eprintln!("expert_ffn L{l} E{e}: |y| = {norm:.4}, first call {cold:.2} ms");
        }

        fn time_ms(f: &mut dyn FnMut()) -> f64 {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        }
        fn best_of(f: &mut dyn FnMut(), reps: usize) -> f64 {
            (0..reps)
                .map(|_| time_ms(&mut *f))
                .fold(f64::INFINITY, f64::min)
        }
        let ffn_warm = best_of(&mut || drop(experts.expert_ffn(0, 0, &x1, 1).unwrap()), 5);
        eprintln!("TIMING expert_ffn n=1 (layer 0, Q4_K/Q5_1, warm page cache): {ffn_warm:.2} ms");

        // Real routing: the layer's own router on the random input.
        let layer = 0;
        let router = gguf
            .read_tensor_f32(&format!("blk.{layer}.ffn_gate_inp.weight"))
            .expect("router weight");
        assert_eq!(router.len(), 512 * 2560);
        let logits_for = |xs: &[f32]| -> Vec<f32> {
            xs.as_chunks::<2560>()
                .0
                .iter()
                .flat_map(|x| {
                    router
                        .as_chunks::<2560>()
                        .0
                        .iter()
                        .map(|w| dot(w, x))
                        .collect::<Vec<_>>()
                })
                .collect()
        };
        for n in [1usize, 16] {
            let xs = unit_rms(&mut rng, n);
            let logits = logits_for(&xs);
            let routing = route(&logits, n, 512, 10).unwrap();
            let distinct = {
                let mut ids = routing.ids.clone();
                ids.sort_unstable();
                ids.dedup();
                ids.len()
            };
            let mut out = Vec::new();
            let cold =
                time_ms(&mut || out = experts.routed_moe(layer, &logits, &xs, n, 10).unwrap());
            assert_eq!(out.len(), n * 2560);
            assert!(out.iter().all(|v| v.is_finite()));
            let warm = best_of(
                &mut || drop(experts.routed_moe(layer, &logits, &xs, n, 10).unwrap()),
                3,
            );
            eprintln!(
                "TIMING routed_moe layer {layer} n={n} k=10 ({distinct} distinct experts): first {cold:.2} ms, warm best-of-3 {warm:.2} ms"
            );
        }

        // Cache on: same numbers bit for bit, and what a hit costs.
        let cached = RoutedExperts::open(&gguf, 48, 512 << 20).expect("cached banks");
        let xs = unit_rms(&mut rng, 1);
        let logits = logits_for(&xs);
        let base = experts.routed_moe(layer, &logits, &xs, 1, 10).unwrap();
        let mut first = Vec::new();
        let miss = time_ms(&mut || first = cached.routed_moe(layer, &logits, &xs, 1, 10).unwrap());
        let hit = best_of(
            &mut || drop(cached.routed_moe(layer, &logits, &xs, 1, 10).unwrap()),
            3,
        );
        assert!(
            base.iter()
                .zip(&first)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        eprintln!(
            "TIMING routed_moe n=1 with f32 cache: fill {miss:.2} ms, hit best-of-3 {hit:.2} ms, {:?}",
            cached.cache_stats().unwrap()
        );
    }
}
