//! PLE: the n-gram row hash, the on-demand IQ4_NL row gather, and the gated
//! key/value + dilated depthwise conv block.
//!
//! qwen4exp adds one "per-layer embedding" block (at layer 1 in the shipped
//! model) that injects hashed n-gram features into every residual stream.
//! Three pieces, one per section below:
//!
//! 1. **Row ids** ([`ple_row_ids`], [`PleHash`]) — host-side exact integer
//!    arithmetic, ported from llama.cpp `llm_graph_input_ple::set_input`
//!    (`src/models/qwen4exp.cpp`). Each token hashes its bigram and trigram
//!    (with an EOS-reset window) into `(n_gram-1)·per_gram = 16` rows of one
//!    shared table: heads `0..per_gram` are the BIGRAM heads, the next
//!    `per_gram` the TRIGRAM heads.
//! 2. **The table** ([`PleTable`]) — `per_layer_token_embd.weight`, 320 M rows
//!    of 160 at IQ4_NL (28.8 GB). It is never loaded: each token's 16 rows are
//!    positional reads of 90 bytes each, dequantized on the host.
//! 3. **The block** ([`PleBlock`]) — `res + gated + silu(conv(norm(gated)))`
//!    with a per-stream signed-sqrt sigmoid gate and a kernel-4, dilation-3
//!    depthwise causal conv that carries a 9-column history across calls.
//!
//! Every step was checked against BOTH llama.cpp and transformers
//! (`Qwen4ExpTextNGramEmbedding` / `Qwen4ExpTextPLELayer`); they agree on the
//! math. The row ids are additionally pinned to a numpy transcription of the
//! transformers code (`tools/qwen4exp_ple_rows.py`, fixture
//! `tests/fixtures/qwen4exp_ple_rows.json`), which also re-derives the
//! header's multipliers, prime head vocabularies and offsets from the
//! transformers seed and asserts they equal the shipped GGUF's.
//!
//! The window-reset token is `qwen4exp.ple.eos_token_id` (248044 in the
//! shipped file), NOT the tokenizer EOS (`tokenizer.ggml.eos_token_id`,
//! 248046): the two differ, and hashing with the wrong one silently selects
//! wrong rows after every document boundary.

use std::path::PathBuf;
use std::sync::Arc;

use burn::module::{Module, Param};
use burn::nn::Linear;
use burn::tensor::{Device, Tensor, TensorData, activation};

use super::hc::{grouped_rms, linear_from, linear3};
use crate::gguf::{GgmlType, GgufFile, GgufValue};

/// The name of the PLE table tensor in a qwen4exp GGUF.
pub const PLE_TABLE_TENSOR: &str = "per_layer_token_embd.weight";

/// Largest n-gram the hash supports. The shipped model uses 3; the bound only
/// lets the per-token context live in a fixed stack array instead of a heap
/// allocation per token.
pub const MAX_NGRAM: usize = 16;

/// Upper bound on header hash arrays: a corrupt header must not drive an
/// unbounded allocation (the shipped arrays hold 3 and 16 entries).
const MAX_PLE_HEADS: usize = 1024;

// ---- 1. Row ids ------------------------------------------------------------

/// The PLE table rows for ONE token, written to `out`
/// (`(n_gram-1)·per_gram` entries, head order).
///
/// * `prev[s-1]` is the token `s` positions back in the SEQUENCE (earlier
///   calls count), `None` or absent when that position is before the
///   sequence start. Entries beyond `n_gram-1` are ignored.
/// * `multipliers` has one entry per n-gram position, so `n_gram` is its
///   length.
///
/// Exactly llama.cpp's `set_input`:
///
/// ```text
/// ctx[0] = token; cut = false
/// for s in 1..n_gram:  t = cut ? NONE : prev[s-1]
///                      cut |= t is NONE || t == eos
///                      ctx[s] = cut ? eos : t
/// for n in 2..=n_gram: mixed = ctx[0]*m[0] ^ ... ^ ctx[n-1]*m[n-1]   (u64, wrapping)
///                      row[(n-2)*per_gram + g] = mixed % vocab[..] + offset[..]
/// ```
///
/// An EOS anywhere in the window replaces itself and everything older with
/// EOS; a missing predecessor reads as EOS; the current token being EOS does
/// NOT cut its own context (transformers' `_shift_right_ignore_eos` looks
/// only at EOS strictly before the position — same rule).
///
/// # Panics
///
/// When the slice lengths are inconsistent (`n_gram < 2`, `n_gram >`
/// [`MAX_NGRAM`], or `head_vocab_sizes`/`head_offsets`/`out` not
/// `(n_gram-1)·per_gram` long) or a head vocabulary is zero. [`PleHash::new`]
/// validates all of these once for the hot path.
#[allow(clippy::too_many_arguments)] // the hash's parameters, one per header array
pub fn ple_row_ids(
    token: u32,
    prev: &[Option<u32>],
    multipliers: &[u64],
    head_vocab_sizes: &[u64],
    head_offsets: &[u64],
    per_gram: usize,
    eos: u32,
    out: &mut [u64],
) {
    let n_gram = multipliers.len();
    assert!(
        (2..=MAX_NGRAM).contains(&n_gram),
        "n_gram {n_gram} outside 2..={MAX_NGRAM}"
    );
    let n_heads = (n_gram - 1) * per_gram;
    assert_eq!(head_vocab_sizes.len(), n_heads, "one vocab per head");
    assert_eq!(head_offsets.len(), n_heads, "one offset per head");
    assert_eq!(out.len(), n_heads, "one row per head");

    let mut ctx = [0u64; MAX_NGRAM];
    ctx[0] = u64::from(token);
    let mut cut = false;
    for s in 1..n_gram {
        let t = if cut {
            None
        } else {
            prev.get(s - 1).copied().flatten()
        };
        cut = cut || t.is_none() || t == Some(eos);
        ctx[s] = u64::from(match t {
            Some(v) if !cut => v,
            _ => eos,
        });
    }

    for n in 2..=n_gram {
        // u64 wrapping multiply + xor, as llama.cpp's uint64_t arithmetic.
        // (transformers uses int64 but its multipliers are drawn below
        // (2^63-1)/vocab, so no in-vocab product overflows either way.)
        let mut mixed = ctx[0].wrapping_mul(multipliers[0]);
        for j in 1..n {
            mixed ^= ctx[j].wrapping_mul(multipliers[j]);
        }
        let base = (n - 2) * per_gram;
        for g in 0..per_gram {
            let h = base + g;
            let vocab = head_vocab_sizes[h];
            assert!(vocab > 0, "PLE head {h} has an empty vocabulary");
            out[h] = mixed % vocab + head_offsets[h];
        }
    }
}

/// The PLE hash parameters, validated once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PleHash {
    /// `n_gram` (3): the largest n-gram hashed.
    pub ngram_size: usize,
    /// Heads per n-gram order (8).
    pub heads_per_ngram: usize,
    /// One multiplier per n-gram position.
    pub multipliers: Vec<u64>,
    /// Vocabulary (a distinct prime) per head, head order.
    pub head_vocab_sizes: Vec<u64>,
    /// Start of each head's slice of the table.
    pub head_offsets: Vec<u64>,
    /// The window-reset token (`qwen4exp.ple.eos_token_id`).
    pub eos: u32,
}

impl PleHash {
    /// Validate and build.
    pub fn new(
        ngram_size: usize,
        heads_per_ngram: usize,
        multipliers: Vec<u64>,
        head_vocab_sizes: Vec<u64>,
        head_offsets: Vec<u64>,
        eos: u32,
    ) -> Result<Self, String> {
        if !(2..=MAX_NGRAM).contains(&ngram_size) {
            return Err(format!(
                "PLE ngram_size {ngram_size} outside 2..={MAX_NGRAM}"
            ));
        }
        if multipliers.len() != ngram_size {
            return Err(format!(
                "PLE has {} multipliers for ngram_size {ngram_size}",
                multipliers.len()
            ));
        }
        let n_heads = (ngram_size - 1)
            .checked_mul(heads_per_ngram)
            .filter(|&n| n > 0 && n <= MAX_PLE_HEADS)
            .ok_or_else(|| {
                format!("PLE head count ({ngram_size}-1)*{heads_per_ngram} is out of range")
            })?;
        if head_vocab_sizes.len() != n_heads || head_offsets.len() != n_heads {
            return Err(format!(
                "PLE expects {n_heads} heads, header has {} vocab sizes and {} offsets",
                head_vocab_sizes.len(),
                head_offsets.len()
            ));
        }
        if let Some(h) = head_vocab_sizes.iter().position(|&v| v == 0) {
            return Err(format!("PLE head {h} has an empty vocabulary"));
        }
        let total = head_vocab_sizes
            .iter()
            .try_fold(0u64, |acc, &v| acc.checked_add(v))
            .ok_or("PLE head vocabularies overflow u64")?;
        // The table bound is the SUMMED vocabulary; every head's slice must
        // fit under it or a valid hash could address a padding row.
        for (h, (&off, &v)) in head_offsets.iter().zip(&head_vocab_sizes).enumerate() {
            if off.checked_add(v).is_none_or(|end| end > total) {
                return Err(format!(
                    "PLE head {h} slice {off}+{v} runs past the {total}-row table"
                ));
            }
        }
        Ok(Self {
            ngram_size,
            heads_per_ngram,
            multipliers,
            head_vocab_sizes,
            head_offsets,
            eos,
        })
    }

    /// The hash parameters from a qwen4exp header's `qwen4exp.ple.*` keys.
    ///
    /// The reset token is read from `qwen4exp.ple.eos_token_id` and a missing
    /// key is an error — falling back to the tokenizer EOS would pick a
    /// different id (248046 vs 248044 in the shipped file) and silently hash
    /// wrong rows after every EOS.
    pub fn from_gguf(f: &GgufFile) -> Result<Self, String> {
        let uint = |key: &str| -> Result<u64, String> {
            let v = f
                .get(key)
                .ok_or_else(|| format!("GGUF metadata missing {key}"))?;
            v.as_u64()
                .or_else(|| v.as_i64().and_then(|x| u64::try_from(x).ok()))
                .ok_or_else(|| format!("{key} is not a non-negative integer"))
        };
        let usize_at = |key: &str| -> Result<usize, String> {
            usize::try_from(uint(key)?).map_err(|_| format!("{key} does not fit usize"))
        };
        let array = |key: &str| -> Result<Vec<u64>, String> {
            let vals = f
                .get(key)
                .and_then(GgufValue::as_array)
                .ok_or_else(|| format!("GGUF metadata missing array {key}"))?;
            if vals.len() > MAX_PLE_HEADS {
                return Err(format!(
                    "{key} has {} entries, past the {MAX_PLE_HEADS} bound",
                    vals.len()
                ));
            }
            vals.iter()
                .map(|v| {
                    v.as_u64()
                        .or_else(|| v.as_i64().and_then(|x| u64::try_from(x).ok()))
                        .ok_or_else(|| format!("{key} holds a non-integer entry"))
                })
                .collect()
        };
        let eos = u32::try_from(uint("qwen4exp.ple.eos_token_id")?)
            .map_err(|_| "qwen4exp.ple.eos_token_id does not fit u32".to_string())?;
        Self::new(
            usize_at("qwen4exp.ple.ngram_size")?,
            usize_at("qwen4exp.ple.heads_per_ngram")?,
            array("qwen4exp.ple.layer_multipliers")?,
            array("qwen4exp.ple.head_vocab_sizes")?,
            array("qwen4exp.ple.head_offsets")?,
            eos,
        )
    }

    /// Rows gathered per token (16).
    #[must_use]
    pub fn n_heads(&self) -> usize {
        self.head_vocab_sizes.len()
    }

    /// Predecessor tokens the hash reads (`n_gram - 1` = 2) — what a cache
    /// must keep between calls.
    #[must_use]
    pub fn context_len(&self) -> usize {
        self.ngram_size - 1
    }

    /// Rows a lookup may address: the summed head vocabularies
    /// (320,001,446). The stored tensor is padded past this.
    #[must_use]
    pub fn total_rows(&self) -> u64 {
        self.head_vocab_sizes.iter().sum()
    }

    /// [`ple_row_ids`] with these parameters.
    pub fn row_ids(&self, token: u32, prev: &[Option<u32>], out: &mut [u64]) {
        ple_row_ids(
            token,
            prev,
            &self.multipliers,
            &self.head_vocab_sizes,
            &self.head_offsets,
            self.heads_per_ngram,
            self.eos,
            out,
        );
    }

    /// Rows for a span of `tokens` that follows `history` (the sequence's
    /// earlier tokens, oldest first; only the last [`Self::context_len`] are
    /// read). Returns `tokens.len() · n_heads` rows, token-major.
    ///
    /// One-shot and chunked calls agree by construction: position `i` sees
    /// the same predecessors whether they sit in `tokens` or in `history`.
    #[must_use]
    pub fn rows_for_span(&self, history: &[u32], tokens: &[u32]) -> Vec<u64> {
        let n_heads = self.n_heads();
        let ctx = self.context_len();
        let mut out = vec![0u64; tokens.len() * n_heads];
        let mut prev = [None; MAX_NGRAM];
        for (i, &tok) in tokens.iter().enumerate() {
            for s in 1..=ctx {
                prev[s - 1] = if s <= i {
                    Some(tokens[i - s])
                } else {
                    history.len().checked_sub(s - i).map(|k| history[k])
                };
            }
            self.row_ids(tok, &prev[..ctx], &mut out[i * n_heads..(i + 1) * n_heads]);
        }
        out
    }

    /// Append `tokens` to a rolling `history`, keeping only the last
    /// [`Self::context_len`] ids the hash can ever read.
    pub fn advance_history(&self, history: &mut Vec<u32>, tokens: &[u32]) {
        history.extend_from_slice(tokens);
        let keep = self.context_len();
        if history.len() > keep {
            history.drain(..history.len() - keep);
        }
    }
}

// ---- 2. The table ------------------------------------------------------------

/// One positional read, looping over short reads — `pread` on unix and its
/// `seek_read` twin on Windows (the same shape as `pack.rs`'s reader).
fn read_exact_at(file: &std::fs::File, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    fn read_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
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
                    "GGUF payload ended before the requested range",
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

/// One open read handle per payload file of a (possibly split) GGUF.
///
/// Positional reads on a shared `&File` are thread-safe (no seek cursor), so
/// the handles can be shared — e.g. with the routed-expert reader, whose
/// banks span every shard — through an `Arc`.
#[derive(Debug)]
pub struct GgufPayloadFiles {
    shards: Vec<PayloadShard>,
}

#[derive(Debug)]
struct PayloadShard {
    path: PathBuf,
    file: std::fs::File,
    data_offset: u64,
    tensors: std::ops::Range<usize>,
}

impl GgufPayloadFiles {
    /// Open every payload file of `f` once.
    pub fn open(f: &GgufFile) -> Result<Self, String> {
        let open = |path: &std::path::Path| {
            std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))
        };
        let shards = if f.shards.is_empty() {
            vec![PayloadShard {
                file: open(&f.path)?,
                path: f.path.clone(),
                data_offset: f.data_offset,
                tensors: 0..f.tensors.len(),
            }]
        } else {
            f.shards
                .iter()
                .map(|s| {
                    Ok(PayloadShard {
                        file: open(&s.path)?,
                        path: s.path.clone(),
                        data_offset: s.data_offset,
                        tensors: s.tensors.clone(),
                    })
                })
                .collect::<Result<_, String>>()?
        };
        Ok(Self { shards })
    }

    /// `(shard slot, absolute payload start)` of tensor `index` of `f`, after
    /// checking the owning file really holds the whole payload — a truncated
    /// shard would otherwise surface as a short read mid-generation.
    fn locate(&self, f: &GgufFile, index: usize) -> Result<(usize, u64), String> {
        let info = f
            .tensors
            .get(index)
            .ok_or_else(|| format!("tensor index {index} out of range"))?;
        let slot = self
            .shards
            .iter()
            .position(|s| s.tensors.contains(&index))
            .ok_or_else(|| format!("tensor {} belongs to no open shard", info.name))?;
        let shard = &self.shards[slot];
        let start = shard
            .data_offset
            .checked_add(info.offset)
            .ok_or("payload offset overflows u64")?;
        let end = start
            .checked_add(info.byte_len())
            .ok_or("payload end overflows u64")?;
        let len = shard
            .file
            .metadata()
            .map_err(|e| format!("stat {}: {e}", shard.path.display()))?
            .len();
        if len < end {
            return Err(format!(
                "{} is {len} bytes but {} needs bytes up to {end}",
                shard.path.display(),
                info.name
            ));
        }
        Ok((slot, start))
    }

    /// Read `buf.len()` bytes at absolute `offset` of shard `slot`.
    fn read_exact_at(&self, slot: usize, offset: u64, buf: &mut [u8]) -> Result<(), String> {
        let shard = &self.shards[slot];
        read_exact_at(&shard.file, buf, offset).map_err(|e| {
            format!(
                "read {} bytes at {offset} of {}: {e}",
                buf.len(),
                shard.path.display()
            )
        })
    }
}

#[derive(Debug)]
enum TableSource {
    /// Rows read on demand from the GGUF payload.
    File {
        files: Arc<GgufPayloadFiles>,
        slot: usize,
        start: u64,
    },
    /// The table's raw bytes held in memory (tests, small tables).
    Memory(Vec<u8>),
}

/// The PLE table: fixed-width rows of one GGML dtype, gathered row by row.
#[derive(Debug)]
pub struct PleTable {
    source: TableSource,
    dtype: GgmlType,
    /// Elements per row (160).
    row_width: usize,
    /// Bytes per row (90 at IQ4_NL: 5 blocks of 18).
    row_bytes: usize,
    /// Rows the table stores (320,001,536 — padded).
    rows: u64,
    /// Rows a lookup may address (the summed head vocab, 320,001,446).
    row_bound: u64,
}

impl PleTable {
    /// Locate [`PLE_TABLE_TENSOR`] in `f` and open its payload files.
    /// `row_bound` is the summed head vocabulary ([`PleHash::total_rows`]).
    pub fn open(f: &GgufFile, row_bound: u64) -> Result<Self, String> {
        Self::open_with(f, Arc::new(GgufPayloadFiles::open(f)?), row_bound)
    }

    /// [`Self::open`] on handles shared with another reader.
    pub fn open_with(
        f: &GgufFile,
        files: Arc<GgufPayloadFiles>,
        row_bound: u64,
    ) -> Result<Self, String> {
        let index = f
            .tensor_index(PLE_TABLE_TENSOR)
            .ok_or_else(|| format!("GGUF has no {PLE_TABLE_TENSOR}"))?;
        let info = &f.tensors[index];
        // ggml dims are fastest-varying first: [row_width, rows].
        let &[width, rows] = info.dims.as_slice() else {
            return Err(format!(
                "{PLE_TABLE_TENSOR} must be 2-D, got dims {:?}",
                info.dims
            ));
        };
        let width = usize::try_from(width).map_err(|_| "PLE row width does not fit usize")?;
        let (row_width, row_bytes) = row_layout(info.dtype, width)?;
        let (slot, start) = files.locate(f, index)?;
        Self {
            source: TableSource::File { files, slot, start },
            dtype: info.dtype,
            row_width,
            row_bytes,
            rows,
            row_bound: rows,
        }
        .with_row_bound(row_bound)
    }

    /// A table over in-memory bytes: `rows` rows of `dtype`, row width
    /// inferred from the byte count.
    pub fn from_bytes(bytes: Vec<u8>, dtype: GgmlType, rows: u64) -> Result<Self, String> {
        let rows_usize = usize::try_from(rows).map_err(|_| "row count does not fit usize")?;
        if rows == 0 || bytes.is_empty() || !bytes.len().is_multiple_of(rows_usize) {
            return Err(format!("{} bytes is not {rows} whole rows", bytes.len()));
        }
        let row_bytes = bytes.len() / rows_usize;
        let bpb = usize::try_from(dtype.bytes_per_block()).map_err(|_| "block too wide")?;
        if !row_bytes.is_multiple_of(bpb) {
            return Err(format!(
                "a {row_bytes}-byte row is not whole {dtype:?} blocks of {bpb}"
            ));
        }
        let width =
            row_bytes / bpb * usize::try_from(dtype.block_size()).map_err(|_| "block too wide")?;
        let (row_width, row_bytes_check) = row_layout(dtype, width)?;
        debug_assert_eq!(row_bytes_check, row_bytes, "layout round-trips");
        Ok(Self {
            source: TableSource::Memory(bytes),
            dtype,
            row_width,
            row_bytes,
            rows,
            row_bound: rows,
        })
    }

    /// Restrict lookups to rows below `bound` (must not exceed the stored
    /// rows). The shipped table is padded to a multiple of 128 rows; the pad
    /// rows are never valid hash outputs, so addressing one is a bug upstream.
    pub fn with_row_bound(mut self, bound: u64) -> Result<Self, String> {
        if bound == 0 || bound > self.rows {
            return Err(format!(
                "PLE row bound {bound} must be in 1..={} (the stored rows)",
                self.rows
            ));
        }
        self.row_bound = bound;
        Ok(self)
    }

    /// Elements per row.
    #[must_use]
    pub fn row_width(&self) -> usize {
        self.row_width
    }

    /// Rows a lookup may address.
    #[must_use]
    pub fn row_bound(&self) -> u64 {
        self.row_bound
    }

    /// The stored dtype.
    #[must_use]
    pub fn dtype(&self) -> GgmlType {
        self.dtype
    }

    /// Dequantized rows, concatenated in `rows` order into `out`
    /// (`rows.len() · row_width`). A row at or past the bound is an error.
    pub fn gather(&self, rows: &[u64], out: &mut [f32]) -> Result<(), String> {
        if out.len() != rows.len() * self.row_width {
            return Err(format!(
                "gather of {} rows needs {} outputs, got {}",
                rows.len(),
                rows.len() * self.row_width,
                out.len()
            ));
        }
        if rows.is_empty() {
            return Ok(());
        }
        let rb = self.row_bytes;
        let mut bytes = vec![0u8; rows.len() * rb];
        for (dst, &row) in bytes.chunks_exact_mut(rb).zip(rows) {
            if row >= self.row_bound {
                return Err(format!(
                    "PLE row {row} is past the {}-row hash range",
                    self.row_bound
                ));
            }
            // row < bound <= rows, and rows·row_bytes was a real payload size.
            let at = row * rb as u64;
            match &self.source {
                TableSource::File { files, slot, start } => {
                    files.read_exact_at(*slot, start + at, dst)?;
                }
                TableSource::Memory(b) => {
                    let at = usize::try_from(at).map_err(|_| "row offset does not fit usize")?;
                    dst.copy_from_slice(&b[at..at + rb]);
                }
            }
        }
        // Rows are whole blocks, so dequantizing the concatenation equals
        // dequantizing each row.
        let vals = crate::gguf::dequantize(self.dtype, &bytes)?;
        out.copy_from_slice(&vals);
        Ok(())
    }

    /// The concatenated PLE embedding of each token of a span following
    /// `history` (see [`PleHash::rows_for_span`]): `tokens.len() · n_heads ·
    /// row_width` values, token-major, head-major within a token (head 0's
    /// row first — ggml `get_rows` + reshape and transformers' `flatten(-2)`
    /// agree on that order).
    pub fn embed(
        &self,
        hash: &PleHash,
        history: &[u32],
        tokens: &[u32],
    ) -> Result<Vec<f32>, String> {
        if hash.total_rows() > self.row_bound {
            return Err(format!(
                "PLE hash addresses {} rows but the table allows {}",
                hash.total_rows(),
                self.row_bound
            ));
        }
        let rows = hash.rows_for_span(history, tokens);
        let mut out = vec![0f32; rows.len() * self.row_width];
        self.gather(&rows, &mut out)?;
        Ok(out)
    }

    /// [`Self::embed`] as a `[1, T, n_heads·row_width]` tensor in the
    /// device's float dtype.
    pub fn embed_tensor(
        &self,
        hash: &PleHash,
        history: &[u32],
        tokens: &[u32],
        device: &Device,
    ) -> Result<Tensor<3>, String> {
        let vals = self.embed(hash, history, tokens)?;
        let width = hash.n_heads() * self.row_width;
        let dtype = crate::backend::float_dtype(device);
        Ok(Tensor::from_data(
            TensorData::new(vals, [1, tokens.len(), width]),
            (device, dtype),
        ))
    }
}

/// `(elements, bytes)` of one row of `width` elements of `dtype`, refusing a
/// row that is not whole blocks (it could not be read independently).
fn row_layout(dtype: GgmlType, width: usize) -> Result<(usize, usize), String> {
    let block = usize::try_from(dtype.block_size()).map_err(|_| "block too wide")?;
    let bpb = usize::try_from(dtype.bytes_per_block()).map_err(|_| "block too wide")?;
    if width == 0 || !width.is_multiple_of(block) {
        return Err(format!(
            "a {width}-element row is not whole {dtype:?} blocks of {block}"
        ));
    }
    Ok((width, width / block * bpb))
}

// ---- 3. The block ------------------------------------------------------------

/// The PLE conv's carried history for one generation: the last
/// `(kernel-1)·dilation = 9` columns of the normalized gated value per
/// channel, oldest column first. Zeros at sequence start.
///
/// Layout `[batch][channel][column]` — flat `col + cols·(channel +
/// channels·batch)`, the same order as llama.cpp's recurrent conv row
/// (`ne = [cols, channels]`).
#[derive(Debug, Clone, PartialEq)]
pub struct PleConvState {
    hist: Vec<f32>,
    batch: usize,
    channels: usize,
    cols: usize,
}

impl PleConvState {
    /// A fresh (all-zero) history.
    #[must_use]
    pub fn zeros(batch: usize, channels: usize, cols: usize) -> Self {
        Self {
            hist: vec![0.0; batch * channels * cols],
            batch,
            channels,
            cols,
        }
    }

    /// The raw history (see the type docs for the layout).
    #[must_use]
    pub fn history(&self) -> &[f32] {
        &self.hist
    }

    /// Back to the sequence-start state.
    pub fn reset(&mut self) {
        self.hist.fill(0.0);
    }
}

/// The PLE block (GGUF `blk.L.ple_*`).
#[derive(Module, Debug)]
pub struct PleBlock {
    /// `E_ple → H·E` (GGUF `ple_key`, ne `[E_ple, H·E]`).
    pub key: Linear,
    /// `E_ple → E` (GGUF `ple_value`, ne `[E_ple, E]`).
    pub value: Linear,
    /// Grouped-norm gammas `[H·E]`, stream-major, already `1 + w`.
    pub norm_key: Param<Tensor<1>>,
    pub norm_query: Param<Tensor<1>>,
    pub norm_conv: Param<Tensor<1>>,
    /// Depthwise conv kernel `[kernel·H·E]` exactly as the GGUF stores it
    /// (ne `[kernel, H·E]`): tap `k` of channel `c` is flat `k + kernel·c`.
    /// Float storage only — it is read to the host every call.
    pub conv: Param<Tensor<1>>,
    /// `E`.
    pub hidden: usize,
    /// `H`.
    pub streams: usize,
    /// Conv taps (`ple.conv_kernel` = 4).
    pub kernel: usize,
    /// Conv dilation — the n-gram size (3) in both references.
    pub dilation: usize,
    /// RMSNorm epsilon.
    pub eps: f64,
}

impl PleBlock {
    /// Assemble from ready weights; linear weights are burn `[in, out]` and
    /// may be packed.
    ///
    /// # Panics
    ///
    /// On any shape that disagrees with `hidden`/`streams`/`kernel`.
    #[must_use]
    #[allow(clippy::too_many_arguments)] // one argument per GGUF tensor + the scalars
    pub fn from_weights(
        key: Tensor<2>,
        value: Tensor<2>,
        norm_key: Tensor<1>,
        norm_query: Tensor<1>,
        norm_conv: Tensor<1>,
        conv: Tensor<1>,
        hidden: usize,
        streams: usize,
        kernel: usize,
        dilation: usize,
        eps: f64,
    ) -> Self {
        let width = hidden * streams;
        let [e_ple, key_out] = key.dims();
        assert_eq!(key_out, width, "ple key maps E_ple -> H*E");
        assert_eq!(value.dims(), [e_ple, hidden], "ple value maps E_ple -> E");
        for (name, n) in [
            ("norm_key", &norm_key),
            ("norm_query", &norm_query),
            ("norm_conv", &norm_conv),
        ] {
            assert_eq!(n.dims(), [width], "ple {name} is [H*E]");
        }
        assert!(kernel >= 1 && dilation >= 1, "degenerate conv");
        assert_eq!(conv.dims(), [kernel * width], "ple conv is [kernel*H*E]");
        Self {
            key: linear_from(key),
            value: linear_from(value),
            norm_key: Param::from_tensor(norm_key),
            norm_query: Param::from_tensor(norm_query),
            norm_conv: Param::from_tensor(norm_conv),
            conv: Param::from_tensor(conv),
            hidden,
            streams,
            kernel,
            dilation,
            eps,
        }
    }

    /// Columns of history the conv carries: `(kernel-1)·dilation` (9).
    #[must_use]
    pub fn history_len(&self) -> usize {
        (self.kernel - 1) * self.dilation
    }

    /// A zero history for `batch` sequences.
    #[must_use]
    pub fn new_state(&self, batch: usize) -> PleConvState {
        PleConvState::zeros(batch, self.hidden * self.streams, self.history_len())
    }

    /// `res [b, t, H·E]`, `emb [b, t, E_ple]` → `res + gated + conv`, advancing
    /// `state` by `t` columns.
    ///
    /// ```text
    /// key, value = W_key·emb [H·E], W_value·emb [E]
    /// score[s]   = Σ_e (grouped_rms(key)·γk)[s] · (grouped_rms(res)·γq)[s] / sqrt(E)
    /// gate[s]    = sigmoid(sign(score) · sqrt(max(|score|, 1e-6)))       # sign(0) = 0
    /// gated[s]   = value · gate[s]
    /// conv[c, t] = Σ_k w[k + kernel·c] · x[c, t - (kernel-1-k)·dilation]  # x = grouped_rms(gated)·γc, history-prefixed
    /// out        = res + gated + silu(conv)
    /// ```
    ///
    /// # Panics
    ///
    /// When shapes disagree with the block or `state`.
    #[must_use]
    pub fn forward(&self, res: Tensor<3>, emb: Tensor<3>, state: &mut PleConvState) -> Tensor<3> {
        let [b, t, width] = res.dims();
        let (h, e) = (self.streams, self.hidden);
        assert_eq!(
            width,
            h * e,
            "residual width {width} is not {h} streams of {e}"
        );
        let [eb, et, _] = emb.dims();
        assert_eq!(
            (eb, et),
            (b, t),
            "PLE embedding batch/time must match the residual"
        );
        let device = res.device();

        let key = linear3(&self.key, emb.clone()); // [b, t, HE]
        let value = linear3(&self.value, emb); // [b, t, E]
        let key_n = grouped_rms(key, self.norm_key.val(), h, self.eps);
        let query_n = grouped_rms(res.clone(), self.norm_query.val(), h, self.eps);
        // Per-stream dot product, scaled by 1/sqrt(E) BEFORE the signed root
        // (both references).
        let score = key_n
            .mul(query_n)
            .reshape([b, t, h, e])
            .sum_dim(3) // [b, t, H, 1]
            .mul_scalar(1.0 / (e as f32).sqrt());
        let magnitude = score.clone().abs().clamp_min(1e-6_f32).sqrt();
        let gate = activation::sigmoid(score.sign().mul(magnitude));
        let gated = value
            .reshape([b, t, 1, e])
            .mul(gate) // [b, t, H, E]
            .reshape([b, t, width]);
        let normed = grouped_rms(gated.clone(), self.norm_conv.val(), h, self.eps);
        let conv = self.conv_host(normed, state, &device);
        res.add(gated.add(activation::silu(conv)))
    }

    /// The dilated depthwise causal conv over `[history | x]` on the host.
    /// `t` is a prompt chunk or one decode token, and the kernel is 40 K
    /// floats, so a host loop is cheaper than assembling the equivalent
    /// cat/narrow graph — and it keeps the carried state in plain memory.
    fn conv_host(&self, x: Tensor<3>, state: &mut PleConvState, device: &Device) -> Tensor<3> {
        let [b, t, c] = x.dims();
        let (kern, dil, cols) = (self.kernel, self.dilation, self.history_len());
        assert_eq!(
            (state.batch, state.channels, state.cols),
            (b, c, cols),
            "PLE conv state is [batch, channels, cols]"
        );
        let dtype = x.dtype();
        let xs = x
            .into_data()
            .convert::<f32>()
            .try_into_vec::<f32>()
            .expect("converted to f32");
        let w = self
            .conv
            .val()
            .into_data()
            .convert::<f32>()
            .try_into_vec::<f32>()
            .expect("PLE conv kernel is float storage");
        debug_assert_eq!(w.len(), kern * c);

        // padded(bi, ch, j): column j of [history (cols) | x (t)].
        let padded = |hist: &[f32], bi: usize, ch: usize, j: usize| -> f32 {
            if j < cols {
                hist[(bi * c + ch) * cols + j]
            } else {
                xs[(bi * t + (j - cols)) * c + ch]
            }
        };
        let mut out = vec![0f32; b * t * c];
        for bi in 0..b {
            for ti in 0..t {
                let row = &mut out[(bi * t + ti) * c..(bi * t + ti + 1) * c];
                // Tap k reads (kern-1-k)·dil columns back; taps are summed in
                // k order as llama.cpp's graph does.
                for k in 0..kern {
                    let j = cols + ti - (kern - 1 - k) * dil;
                    for (ch, acc) in row.iter_mut().enumerate() {
                        *acc += w[k + kern * ch] * padded(&state.hist, bi, ch, j);
                    }
                }
            }
        }
        // The new history is the last `cols` columns of the padded input.
        let mut next = vec![0f32; b * c * cols];
        for bi in 0..b {
            for ch in 0..c {
                for j in 0..cols {
                    next[(bi * c + ch) * cols + j] = padded(&state.hist, bi, ch, t + j);
                }
            }
        }
        state.hist = next;
        Tensor::from_data(TensorData::new(out, [b, t, c]), (device, dtype))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- row ids ---------------------------------------------------------

    /// Toy parameters small enough to hash by hand: n_gram 3, 2 heads per
    /// order, EOS = 9.
    fn toy_hash() -> PleHash {
        PleHash::new(
            3,
            2,
            vec![3, 5, 7],
            vec![7, 11, 13, 17],
            vec![0, 7, 18, 31],
            9,
        )
        .expect("toy hash validates")
    }

    /// Hand-computed rows for `[2, 4, 9, 6, 1]` (EOS = 9 mid-stream):
    ///
    /// ```text
    /// pos ctx         bigram (c0*3 ^ c1*5)        trigram (^ c2*7)
    /// 0   [2, 9, 9]   6^45=43  -> 43%7, 43%11+7   43^63=20 -> 20%13+18, 20%17+31
    /// 1   [4, 2, 9]   12^10=6  -> 6, 6+7          6^63=57  -> 5+18, 6+31
    /// 2   [9, 4, 2]   27^20=15 -> 1, 4+7          15^14=1  -> 1+18, 1+31   (own EOS does not cut)
    /// 3   [6, 9, 9]   18^45=63 -> 0, 8+7          63^63=0  -> 18, 31       (EOS one back cuts both)
    /// 4   [1, 6, 9]   3^30=29  -> 1, 7+7          29^63=34 -> 8+18, 0+31   (EOS two back)
    /// ```
    /// Positions 0 and 1 read their missing predecessors as EOS.
    #[test]
    fn row_ids_match_a_hand_checked_sequence() {
        let hash = toy_hash();
        let rows = hash.rows_for_span(&[], &[2, 4, 9, 6, 1]);
        let want: [[u64; 4]; 5] = [
            [1, 17, 25, 34],
            [6, 13, 23, 37],
            [1, 11, 19, 32],
            [0, 15, 18, 31],
            [1, 14, 26, 31],
        ];
        for (i, w) in want.iter().enumerate() {
            assert_eq!(&rows[i * 4..(i + 1) * 4], w, "position {i}");
        }
        // The free function agrees, with an explicit missing predecessor.
        let mut out = [0u64; 4];
        ple_row_ids(
            4,
            &[Some(2), None],
            &[3, 5, 7],
            &[7, 11, 13, 17],
            &[0, 7, 18, 31],
            2,
            9,
            &mut out,
        );
        assert_eq!(out, [6, 13, 23, 37]);
        // A shorter prev slice is the same as trailing Nones.
        ple_row_ids(
            4,
            &[Some(2)],
            &[3, 5, 7],
            &[7, 11, 13, 17],
            &[0, 7, 18, 31],
            2,
            9,
            &mut out,
        );
        assert_eq!(out, [6, 13, 23, 37]);
    }

    /// llama.cpp hashes in `uint64_t`: the multiply must wrap, not panic in a
    /// debug build or saturate.
    #[test]
    fn the_hash_multiply_wraps_like_uint64() {
        let m = [u64::MAX, 3, 1];
        let mut out = [0u64; 2];
        ple_row_ids(
            3,
            &[None, None],
            &m,
            &[1_000_003, 7],
            &[0, 1_000_003],
            1,
            0,
            &mut out,
        );
        // ctx = [3, 0, 0]: bigram = 3·(2^64-1) mod 2^64 = 2^64-3; trigram equal.
        let mixed = u64::MAX - 2;
        assert_eq!(out, [mixed % 1_000_003, mixed % 7 + 1_000_003]);
    }

    #[derive(serde::Deserialize)]
    struct RowFixture {
        ngram_size: usize,
        heads_per_ngram: usize,
        eos: u32,
        multipliers: Vec<u64>,
        head_vocab_sizes: Vec<u64>,
        head_offsets: Vec<u64>,
        eos_position: usize,
        tokens: Vec<u32>,
        rows: Vec<Vec<u64>>,
    }

    /// Row ids for 40 random tokens with one EOS equal the numpy
    /// transcription of the TRANSFORMERS reference
    /// (`tools/qwen4exp_ple_rows.py`), with the shipped header's parameters —
    /// an independent second source for the llama.cpp-derived hash. Chunked
    /// spans (history carried) must reproduce the same ids.
    #[test]
    fn row_ids_match_the_transformers_numpy_vectors() {
        let fx: RowFixture = serde_json::from_str(include_str!(
            "../../../tests/fixtures/qwen4exp_ple_rows.json"
        ))
        .expect("fixture parses");
        let hash = PleHash::new(
            fx.ngram_size,
            fx.heads_per_ngram,
            fx.multipliers,
            fx.head_vocab_sizes,
            fx.head_offsets,
            fx.eos,
        )
        .expect("shipped parameters validate");
        assert_eq!(hash.n_heads(), 16);
        assert_eq!(hash.total_rows(), 320_001_446);
        assert_eq!(fx.tokens[fx.eos_position], hash.eos, "fixture has its EOS");
        let n = hash.n_heads();

        let one_shot = hash.rows_for_span(&[], &fx.tokens);
        for (i, want) in fx.rows.iter().enumerate() {
            assert_eq!(&one_shot[i * n..(i + 1) * n], want.as_slice(), "token {i}");
        }

        // Spans of 5, 7 and 6 (the EOS at 17 closes the third), then single
        // steps whose window reaches back across the span boundaries.
        let mut history = Vec::new();
        let mut chunked = Vec::new();
        let mut at = 0;
        for len in [5, 7, 6].into_iter().chain(std::iter::repeat(1)) {
            if at >= fx.tokens.len() {
                break;
            }
            let span = &fx.tokens[at..(at + len).min(fx.tokens.len())];
            chunked.extend(hash.rows_for_span(&history, span));
            hash.advance_history(&mut history, span);
            assert!(history.len() <= hash.context_len());
            at += span.len();
        }
        assert_eq!(
            chunked, one_shot,
            "chunked spans reproduce the one-shot ids"
        );
    }

    #[test]
    fn inconsistent_hash_parameters_are_refused() {
        assert!(PleHash::new(3, 2, vec![1, 2], vec![7; 4], vec![0, 7, 14, 21], 0).is_err());
        assert!(PleHash::new(3, 2, vec![1, 2, 3], vec![7; 3], vec![0, 7, 14], 0).is_err());
        assert!(PleHash::new(3, 2, vec![1, 2, 3], vec![7, 0, 7, 7], vec![0, 7, 7, 14], 0).is_err());
        // An offset past the summed vocabulary would address padding rows.
        assert!(PleHash::new(3, 2, vec![1, 2, 3], vec![7; 4], vec![0, 7, 14, 22], 0).is_err());
        assert!(PleHash::new(3, 2, vec![1, 2, 3], vec![7; 4], vec![0, 7, 14, 21], 0).is_ok());
    }

    // ---- table -------------------------------------------------------------

    fn f32_table_bytes(rows: usize, width: usize) -> Vec<u8> {
        (0..rows * width)
            .flat_map(|i| ((i / width) as f32 * 1000.0 + (i % width) as f32).to_le_bytes())
            .collect()
    }

    /// Rows come back in request order, head-major, and the bound is the
    /// row bound, not the stored row count.
    #[test]
    fn gather_returns_rows_in_order_and_enforces_the_bound() {
        let table = PleTable::from_bytes(f32_table_bytes(4, 3), GgmlType::F32, 4).expect("table");
        assert_eq!(table.row_width(), 3);
        let mut out = vec![0f32; 9];
        table.gather(&[3, 0, 2], &mut out).expect("in range");
        assert_eq!(
            out,
            [
                3000.0, 3001.0, 3002.0, 0.0, 1.0, 2.0, 2000.0, 2001.0, 2002.0
            ]
        );

        let err = table
            .gather(&[4, 0, 0], &mut out)
            .expect_err("past the stored rows");
        assert!(err.contains("past the 4-row"), "{err}");
        let padded = table.with_row_bound(3).expect("bound within rows");
        let err = padded
            .gather(&[0, 3, 1], &mut out)
            .expect_err("a padding row");
        assert!(err.contains("row 3"), "{err}");
        assert!(
            PleTable::from_bytes(f32_table_bytes(4, 3), GgmlType::F32, 4)
                .expect("table")
                .with_row_bound(5)
                .is_err()
        );
    }

    /// Deterministic IQ4_NL rows (5 blocks = 90 bytes = 160 elements each)
    /// with a sane f16 scale in every block.
    fn iq4nl_table_bytes(rows: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut bytes = Vec::with_capacity(rows * 90);
        for _ in 0..rows * 5 {
            // f16 scale in [0.01, ~0.3): exponent bits well below inf/NaN.
            let d = half::f16::from_f32(0.01 + (next() % 1000) as f32 / 3500.0);
            bytes.extend_from_slice(&d.to_le_bytes());
            for _ in 0..16 {
                bytes.push((next() & 0xFF) as u8);
            }
        }
        bytes
    }

    /// 90-byte IQ4_NL rows are addressed at `row · 90` and decode exactly as
    /// `gguf::dequantize` decodes that slice.
    #[test]
    fn iq4nl_rows_decode_as_their_byte_slices() {
        let bytes = iq4nl_table_bytes(6, 42);
        let table = PleTable::from_bytes(bytes.clone(), GgmlType::IQ4_NL, 6).expect("table");
        assert_eq!(table.row_width(), 160);
        let mut out = vec![0f32; 2 * 160];
        table.gather(&[4, 1], &mut out).expect("gather");
        let want4 = crate::gguf::dequantize(GgmlType::IQ4_NL, &bytes[4 * 90..5 * 90]).expect("dq");
        let want1 = crate::gguf::dequantize(GgmlType::IQ4_NL, &bytes[90..180]).expect("dq");
        assert_eq!(&out[..160], want4.as_slice());
        assert_eq!(&out[160..], want1.as_slice());
    }

    /// The file-backed table finds the tensor in the RIGHT shard of a split
    /// set, at its payload base plus tensor offset, and returns what the
    /// in-memory table over the same bytes returns.
    #[test]
    fn the_file_table_reads_from_the_owning_shard() {
        use crate::gguf::{GgufFile, GgufShard, GgufTensorInfo};
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "mummu-ple-table-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let rows = 7usize;
        let table_bytes = iq4nl_table_bytes(rows, 7);

        // Shard 0: 64 junk bytes of header, one unrelated 32-byte tensor.
        let s0 = dir.join("s0.gguf");
        std::fs::write(&s0, vec![0xAAu8; 64 + 32]).expect("write shard 0");
        // Shard 1: a 100-byte header, a 34-byte tensor, then 30 bytes of
        // alignment, then the table at payload offset 64.
        let s1 = dir.join("s1.gguf");
        let mut b1 = vec![0x55u8; 100 + 64];
        b1.extend_from_slice(&table_bytes);
        std::fs::write(&s1, &b1).expect("write shard 1");

        let info = |name: &str, dims: Vec<u64>, dtype, offset| GgufTensorInfo {
            name: name.into(),
            dims,
            dtype,
            offset,
        };
        let f = GgufFile {
            path: s0.clone(),
            version: 3,
            metadata: Vec::new(),
            tensors: vec![
                info("junk.weight", vec![8], GgmlType::F32, 0),
                info("other.weight", vec![32], GgmlType::Q8_0, 0),
                info(
                    PLE_TABLE_TENSOR,
                    vec![160, rows as u64],
                    GgmlType::IQ4_NL,
                    64,
                ),
            ],
            alignment: 32,
            data_offset: 64,
            shards: vec![
                GgufShard {
                    path: s0,
                    data_offset: 64,
                    tensors: 0..1,
                },
                GgufShard {
                    path: s1,
                    data_offset: 100,
                    tensors: 1..3,
                },
            ],
        };
        let file_table = PleTable::open(&f, rows as u64 - 1).expect("table opens");
        let mem_table =
            PleTable::from_bytes(table_bytes, GgmlType::IQ4_NL, rows as u64).expect("mem");
        let ask = [5u64, 0, 3];
        let mut a = vec![0f32; ask.len() * 160];
        let mut b = vec![0f32; ask.len() * 160];
        file_table.gather(&ask, &mut a).expect("file gather");
        mem_table.gather(&ask, &mut b).expect("memory gather");
        assert_eq!(a, b, "file and memory tables agree");
        assert!(
            file_table.gather(&[6], &mut a[..160]).is_err(),
            "bound applies to files too"
        );

        // A header whose shard is shorter than the tensor claims is refused
        // at open, not at the first unlucky read.
        let mut short = f;
        short.tensors[2].dims[1] = rows as u64 + 10;
        assert!(
            PleTable::open(&short, 1).is_err(),
            "truncated shard refused"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    // ---- block ---------------------------------------------------------------

    fn rand_vec(seed: &mut u64, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed ^= *seed << 13;
                *seed ^= *seed >> 7;
                *seed ^= *seed << 17;
                ((*seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn silu(x: f32) -> f32 {
        x * sigmoid(x)
    }

    /// Toy block dims: E = 4, H = 3, E_ple = 8 (distinct from E so a swapped
    /// projection cannot pass), kernel 4, dilation 3 (history 9).
    struct Toy {
        e: usize,
        h: usize,
        ep: usize,
        kern: usize,
        dil: usize,
        key: Vec<f32>,
        value: Vec<f32>,
        nk: Vec<f32>,
        nq: Vec<f32>,
        nc: Vec<f32>,
        conv: Vec<f32>,
    }

    fn toy(seed: u64) -> Toy {
        let (e, h, ep, kern, dil) = (4, 3, 8, 4, 3);
        let mut s = seed;
        let gamma = |s: &mut u64| -> Vec<f32> {
            rand_vec(s, h * e, 0.4)
                .into_iter()
                .map(|x| 1.0 + x)
                .collect()
        };
        Toy {
            e,
            h,
            ep,
            kern,
            dil,
            key: rand_vec(&mut s, h * e * ep, 0.7),
            value: rand_vec(&mut s, e * ep, 0.7),
            nk: gamma(&mut s),
            nq: gamma(&mut s),
            nc: gamma(&mut s),
            conv: rand_vec(&mut s, kern * h * e, 0.6),
        }
    }

    fn build(t: &Toy, device: &Device) -> PleBlock {
        // GGUF row-major [out, in] → burn [in, out].
        let w = |v: &[f32], out: usize, inp: usize| {
            Tensor::<2>::from_data(TensorData::new(v.to_vec(), [out, inp]), device).swap_dims(0, 1)
        };
        let v1 = |v: &[f32]| Tensor::<1>::from_data(TensorData::new(v.to_vec(), [v.len()]), device);
        PleBlock::from_weights(
            w(&t.key, t.h * t.e, t.ep),
            w(&t.value, t.e, t.ep),
            v1(&t.nk),
            v1(&t.nq),
            v1(&t.nc),
            v1(&t.conv),
            t.e,
            t.h,
            t.kern,
            t.dil,
            1e-6,
        )
    }

    fn host(x: Tensor<3>) -> Vec<f32> {
        x.into_data()
            .convert::<f32>()
            .try_into_vec::<f32>()
            .expect("f32")
    }

    fn grouped(v: &[f32], gamma: &[f32], e: usize, h: usize) -> Vec<f32> {
        let mut out = vec![0f32; h * e];
        for s in 0..h {
            let g = &v[s * e..(s + 1) * e];
            let inv = 1.0 / (g.iter().map(|x| x * x).sum::<f32>() / e as f32 + 1e-6).sqrt();
            for i in 0..e {
                out[s * e + i] = g[i] * inv * gamma[s * e + i];
            }
        }
        out
    }

    /// Scalar reference over a whole sequence (batch 1, zero initial
    /// history), transcribed from the spec (section 4).
    fn ref_forward(t: &Toy, res: &[f32], emb: &[f32], n: usize) -> Vec<f32> {
        let (e, h, ep) = (t.e, t.h, t.ep);
        let hd = h * e;
        let matvec = |w: &[f32], x: &[f32], out: usize| -> Vec<f32> {
            (0..out)
                .map(|o| (0..x.len()).map(|i| w[o * x.len() + i] * x[i]).sum())
                .collect()
        };
        let mut gated_all = vec![0f32; n * hd];
        let mut normed_all = vec![0f32; n * hd];
        for tok in 0..n {
            let em = &emb[tok * ep..(tok + 1) * ep];
            let r = &res[tok * hd..(tok + 1) * hd];
            let keyn = grouped(&matvec(&t.key, em, hd), &t.nk, e, h);
            let value = matvec(&t.value, em, e);
            let qn = grouped(r, &t.nq, e, h);
            for s in 0..h {
                let score: f32 = (0..e).map(|i| keyn[s * e + i] * qn[s * e + i]).sum::<f32>()
                    / (e as f32).sqrt();
                let sign = if score > 0.0 {
                    1.0
                } else if score < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                let gate = sigmoid(sign * score.abs().max(1e-6).sqrt());
                for i in 0..e {
                    gated_all[tok * hd + s * e + i] = value[i] * gate;
                }
            }
            let nrm = grouped(&gated_all[tok * hd..(tok + 1) * hd], &t.nc, e, h);
            normed_all[tok * hd..(tok + 1) * hd].copy_from_slice(&nrm);
        }
        let mut out = vec![0f32; n * hd];
        for tok in 0..n {
            for c in 0..hd {
                let mut acc = 0f32;
                for k in 0..t.kern {
                    // Tap k reads (kern-1-k)·dil tokens back; before the
                    // sequence start that is the zero history.
                    let back = (t.kern - 1 - k) * t.dil;
                    let x = if tok >= back {
                        normed_all[(tok - back) * hd + c]
                    } else {
                        0.0
                    };
                    acc += t.conv[k + t.kern * c] * x;
                }
                out[tok * hd + c] = res[tok * hd + c] + gated_all[tok * hd + c] + silu(acc);
            }
        }
        out
    }

    /// The tensor block equals the scalar spec transcription on random
    /// weights over 15 tokens — gate, grouped norms, the conv's tap order and
    /// dilation, and the `[kernel, H·E]` kernel layout all have to agree.
    #[test]
    fn ple_block_matches_the_scalar_reference() {
        let device = crate::backend::cpu_device();
        let t = toy(0xC0FFEE);
        let block = build(&t, &device);
        assert_eq!(block.history_len(), 9);
        let n = 15;
        let hd = t.h * t.e;
        let mut s = 17;
        let res = rand_vec(&mut s, n * hd, 2.0);
        let emb = rand_vec(&mut s, n * t.ep, 1.0);
        let mut state = block.new_state(1);
        let got = host(block.forward(
            Tensor::from_data(TensorData::new(res.clone(), [1, n, hd]), &device),
            Tensor::from_data(TensorData::new(emb.clone(), [1, n, t.ep]), &device),
            &mut state,
        ));
        let want = ref_forward(&t, &res, &emb, n);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() <= 1e-5 * (1.0 + w.abs()), "[{i}] {g} vs {w}");
        }
    }

    /// A score of exactly zero gates at sigmoid(0) = 1/2: `sign(0) = 0` in
    /// ggml, torch and burn. Were it +1 the gate would be sigmoid(1e-3).
    #[test]
    fn a_zero_score_gates_at_one_half() {
        let device = crate::backend::cpu_device();
        let mut t = toy(5);
        t.nk = vec![0.0; t.h * t.e]; // normalized key is zero -> score 0
        t.nc = vec![0.0; t.h * t.e]; // conv input zero -> conv term silu(0) = 0
        let block = build(&t, &device);
        let hd = t.h * t.e;
        let mut s = 3;
        let emb = rand_vec(&mut s, t.ep, 1.0);
        let res = vec![0.0; hd];
        let got = host(block.forward(
            Tensor::from_data(TensorData::new(res, [1, 1, hd]), &device),
            Tensor::from_data(TensorData::new(emb.clone(), [1, 1, t.ep]), &device),
            &mut block.new_state(1),
        ));
        let value: Vec<f32> = (0..t.e)
            .map(|o| (0..t.ep).map(|i| t.value[o * t.ep + i] * emb[i]).sum())
            .collect();
        for st in 0..t.h {
            for i in 0..t.e {
                let (g, w) = (got[st * t.e + i], 0.5 * value[i]);
                assert!(
                    (g - w).abs() <= 1e-6 * (1.0 + w.abs()),
                    "stream {st} elem {i}: {g} vs {w}"
                );
            }
        }
    }

    /// Chunked prefill (5 + 7) then three single-token steps equals one
    /// 15-token call, including the carried history — the conv-cache bug
    /// class PR #42 found: a chunk boundary inside the 9-column reach.
    #[test]
    fn chunked_conv_equals_one_shot() {
        let device = crate::backend::cpu_device();
        let t = toy(0xBADC0DE);
        let block = build(&t, &device);
        let n = 15;
        let hd = t.h * t.e;
        let mut s = 99;
        let res = rand_vec(&mut s, n * hd, 2.0);
        let emb = rand_vec(&mut s, n * t.ep, 1.0);
        let run = |state: &mut PleConvState, from: usize, len: usize| {
            host(block.forward(
                Tensor::from_data(
                    TensorData::new(res[from * hd..(from + len) * hd].to_vec(), [1, len, hd]),
                    &device,
                ),
                Tensor::from_data(
                    TensorData::new(
                        emb[from * t.ep..(from + len) * t.ep].to_vec(),
                        [1, len, t.ep],
                    ),
                    &device,
                ),
                state,
            ))
        };
        let mut one = block.new_state(1);
        let whole = run(&mut one, 0, n);

        let mut st = block.new_state(1);
        let mut pieces = Vec::new();
        let mut at = 0;
        for len in [5, 7, 1, 1, 1] {
            pieces.extend(run(&mut st, at, len));
            at += len;
        }
        assert_eq!(at, n);
        for (i, (a, b)) in pieces.iter().zip(&whole).enumerate() {
            assert!(
                (a - b).abs() <= 1e-5 * (1.0 + b.abs()),
                "[{i}] chunked {a} vs one-shot {b}"
            );
        }
        for (a, b) in st.history().iter().zip(one.history()) {
            assert!(
                (a - b).abs() <= 1e-6 * (1.0 + b.abs()),
                "history {a} vs {b}"
            );
        }
    }

    // ---- the shipped table ---------------------------------------------------

    /// Gather PLE embeddings from the REAL shards (NVMe copy!) and time
    /// random-token gathers.
    ///
    /// ```text
    /// MUMMU_QWEN4EXP_DIR=/home/physics515/.cache/mummu-models/qwen3.8-flash-next \
    ///   cargo test -p mummu --lib qwen4exp::ple -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs the Flash-Next split set (MUMMU_QWEN4EXP_DIR)"]
    fn the_shipped_table_gathers_finite_rows() {
        let Some(dir) = std::env::var_os("MUMMU_QWEN4EXP_DIR") else {
            eprintln!("skipped: set MUMMU_QWEN4EXP_DIR to the shard directory");
            return;
        };
        let first =
            std::path::Path::new(&dir).join("Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf");
        if !first.is_file() {
            eprintln!("skipped: {} not found", first.display());
            return;
        }
        let f = GgufFile::open_sharded(&first).expect("split set opens");
        let hash = PleHash::from_gguf(&f).expect("PLE hash parameters");
        assert_eq!(
            hash.eos, 248_044,
            "the PLE reset token, not the tokenizer EOS"
        );
        assert_eq!(hash.n_heads(), 16);
        assert_eq!(hash.total_rows(), 320_001_446);
        let table = PleTable::open(&f, hash.total_rows()).expect("table opens");
        assert_eq!(table.dtype(), GgmlType::IQ4_NL);
        assert_eq!(table.row_width(), 160);

        // Three tokens, the middle one the PLE EOS.
        let tokens = [9707u32, 248_044, 1_234];
        let emb = table.embed(&hash, &[], &tokens).expect("embed");
        assert_eq!(emb.len(), 3 * 2560, "16 x 160 per token");
        assert!(emb.iter().all(|v| v.is_finite()), "finite values");
        let mean_abs = emb.iter().map(|v| v.abs()).sum::<f32>() / emb.len() as f32;
        println!("3-token PLE embedding: mean |x| = {mean_abs:.5}");
        assert!(mean_abs > 0.0, "not an all-zero read");

        // 1000 random single-token gathers (16 preads + dequant each), with
        // random predecessors so the rows spread over the whole table.
        let mut s = 0x5EED_u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let toks: Vec<[u32; 3]> = (0..1000)
            .map(|_| std::array::from_fn(|_| (next() % 248_000) as u32))
            .collect();
        let mut out = vec![0f32; 2560];
        for pass in ["cold", "warm"] {
            let start = std::time::Instant::now();
            for t in &toks {
                let rows = hash.rows_for_span(&t[..2], &t[2..]);
                table.gather(&rows, &mut out).expect("gather");
            }
            let ms = start.elapsed().as_secs_f64() * 1e3 / toks.len() as f64;
            assert!(out.iter().all(|v| v.is_finite()));
            println!(
                "PLE gather ({pass} page cache): {ms:.4} ms/token over {} tokens",
                toks.len()
            );
        }
    }
}
