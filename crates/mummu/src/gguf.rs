//! GGUF container reader — the first slice of P3's "run what the ecosystem
//! ships" import path. GGUF (llama.cpp's format) is one file: a small header
//! of typed metadata key-values, a tensor table (name, shape, quantized
//! dtype, offset), then an aligned blob of tensor payloads.
//!
//! This module reads the *header*: every metadata value typed and bounded,
//! every tensor located and size-checked — and fails loudly on anything
//! malformed, oversized, or unknown. Tensor payloads are *located*, never
//! loaded here; dequantizing them into Burn tensors is the next slice.

use std::fs::File;
use std::io::{BufReader, Read, Seek, Write};
use std::path::Path;

/// GGUF file magic, little-endian `"GGUF"`.
const MAGIC: [u8; 4] = *b"GGUF";

/// Versions this reader understands (v2 moved counts to u64; v3 is v2 plus a
/// big-endian variant this reader rejects by magic).
const SUPPORTED_VERSIONS: [u32; 2] = [2, 3];

/// Default payload alignment when `general.alignment` is absent.
const DEFAULT_ALIGNMENT: u64 = 32;

/// Most metadata key-values a sane model file carries (real models: ~20-40).
const MAX_KVS: u64 = 4096;

/// Most tensors a supported model carries (Qwen2.5-1.5B: 339).
const MAX_TENSORS: u64 = 65_536;

/// Longest metadata string (chat templates run ~10 KiB; 1 MiB is generous).
const MAX_STRING_BYTES: u64 = 1 << 20;

/// Longest metadata array (tokenizer vocab/merges run ~152k entries).
const MAX_ARRAY_LEN: u64 = 1 << 22;

/// GGML allows at most 4 tensor dimensions.
const MAX_DIMS: u32 = 4;

/// Largest dequantized-to-f32 payload either dequant will produce (a ~12B-param
/// model; the reference machine has 128 GB). Only
/// [`GgufFile::dequant_to_safetensors`] holds this much at once;
/// [`GgufFile::dequant_to_safetensors_file`] streams it and never buffers more
/// than one tensor.
const MAX_DEQUANT_BYTES: u64 = 48 << 30;

/// Largest SINGLE dequantized tensor the streaming dequant will hold. The
/// widest real one is a vocabulary embedding — Qwen2.5-1.5B's is 933 MB at
/// f32, a 35B's would be ~3.1 GB — so 8 GiB is a corrupt header claiming an
/// absurd tensor. It is twice `safetensors::MAX_PART_BYTES` on purpose: that
/// path keeps the source dtype, this one widens everything to f32.
const MAX_TENSOR_F32_BYTES: u64 = 8 << 30;

/// Bytes staged between the f32 values and the sink. Fixed and small: the
/// point of the streaming dequant is that peak allocation tracks the widest
/// TENSOR, never the payload, and this buffer must not reintroduce a second
/// copy of either.
const DEQUANT_STAGE_BYTES: usize = 1 << 20;

/// What went wrong reading a GGUF header.
#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("gguf {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("gguf {path}: bad magic {found:02x?} (big-endian GGUF is unsupported)")]
    BadMagic { path: String, found: [u8; 4] },
    #[error("gguf {path}: unsupported version {version} (supported: {SUPPORTED_VERSIONS:?})")]
    UnsupportedVersion { path: String, version: u32 },
    #[error("gguf {path}: {what} count {count} exceeds the {bound} bound")]
    OverBound {
        path: String,
        what: &'static str,
        count: u64,
        bound: u64,
    },
    #[error("gguf {path}: metadata '{key}': {reason}")]
    BadValue {
        path: String,
        key: String,
        reason: String,
    },
    #[error("gguf {path}: tensor {index}: {reason}")]
    BadTensor {
        path: String,
        index: usize,
        reason: String,
    },
}

/// `u64` -> `usize` as an error rather than a panic.
///
/// Every caller has already bounded `value` against one of the `MAX_*`
/// ceilings or a tensor's own declared size, so on a 64-bit target this
/// cannot fail — but an import path is exactly where "cannot fail" should
/// still be an `Err` instead of an `.expect()`, so a 32-bit build degrades to
/// a clean error instead of aborting mid-load. (Same helper, same reasoning,
/// as `safetensors::to_usize`.)
fn to_usize(value: u64, path: &str, what: &'static str) -> Result<usize, GgufError> {
    debug_assert!(
        value <= usize::MAX as u64,
        "{what} fits usize on this target"
    );
    debug_assert!(!path.is_empty(), "an error needs a file to name");
    usize::try_from(value).map_err(|_| GgufError::OverBound {
        path: path.to_string(),
        what,
        count: value,
        bound: usize::MAX as u64,
    })
}

/// One typed metadata value. Arrays are homogeneous per the spec; nested
/// arrays are legal but bounded to one level of nesting in practice.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    /// The value as a string, if it is one.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }

    /// The value widened to u64, if it is any unsigned integer.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U8(v) => Some(u64::from(v)),
            Self::U16(v) => Some(u64::from(v)),
            Self::U32(v) => Some(u64::from(v)),
            Self::U64(v) => Some(v),
            _ => None,
        }
    }

    /// The value widened to i64, if it is any integer (signed or unsigned)
    /// that fits.
    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            Self::I8(v) => Some(i64::from(v)),
            Self::I16(v) => Some(i64::from(v)),
            Self::I32(v) => Some(i64::from(v)),
            Self::I64(v) => Some(v),
            _ => self.as_u64().and_then(|v| i64::try_from(v).ok()),
        }
    }

    /// The value as f32, if it is one.
    #[must_use]
    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Self::F32(v) => Some(v),
            _ => None,
        }
    }

    /// The value as a bool, if it is one.
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Self::Bool(v) => Some(v),
            _ => None,
        }
    }

    /// The value as an array slice, if it is one.
    #[must_use]
    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            Self::Array(items) => Some(items),
            _ => None,
        }
    }
}

/// A GGML tensor dtype, as stored on disk. Quantized types pack fixed-size
/// blocks; `block_size`/`bytes_per_block` give the layout the dequant slice
/// (and size validation here) needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)] // the ecosystem's canonical spellings
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    /// IQ4_XS: 4-bit non-linear (a 16-entry value table) with 6-bit
    /// sub-scales — the workhorse of unsloth's UD dynamic quants.
    IQ4_XS,
    /// IQ4_NL: IQ4_XS's 16-entry table in a simple 32-element block.
    IQ4_NL,
    /// IQ2_XS: 2.3125 bpw — 8-value rows from a 512-entry codebook grid.
    IQ2_XS,
    /// IQ2_S: 2.5625 bpw — the 1024-entry grid with separate sign bytes.
    IQ2_S,
    /// IQ3_XXS: 3.0625 bpw — 4-value rows from a 256-entry grid.
    IQ3_XXS,
    /// IQ3_S: 3.4375 bpw — the 512-entry grid, high index bits in `qh`.
    IQ3_S,
    BF16,
    /// Q1_0: binary `{−d, +d}`, one bit per weight (LSB first) with an f16
    /// scale per 128 (upstream ggml id 41 — Prism's Bonsai 1-bit line).
    Q1_0,
    /// Q2_0: 2-bit code slots, four per byte low bits first, decoding as
    /// `(code − 1)·d` — `{−d, 0, +d, +2d}`, the last reserved — with an f16
    /// scale per 64 (upstream ggml id 42, the group-64 ternary format).
    Q2_0,
    /// PQ2_0: Q2_0's codec with the scale per 128 (Prism-private id 142 —
    /// what Ternary-Bonsai ships as `*-PQ2_0.gguf`).
    PQ2_0,
    /// PTQ1_0: five base-3 trits per byte, scale per 128 (Prism-private id
    /// 143, 1.75 bits/weight — TQ1_0's packing at group 128, not 256).
    PTQ1_0,
}

impl GgmlType {
    /// Decode the on-disk type id; unknown ids are a loud error, never a guess.
    fn from_id(id: u32) -> Option<Self> {
        let ty = match id {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS,
            20 => Self::IQ4_NL,
            21 => Self::IQ3_S,
            22 => Self::IQ2_S,
            23 => Self::IQ4_XS,
            30 => Self::BF16,
            41 => Self::Q1_0,
            42 => Self::Q2_0,
            142 => Self::PQ2_0,
            143 => Self::PTQ1_0,
            _ => return None,
        };
        Some(ty)
    }

    /// Elements per quantization block (1 for plain float types).
    #[must_use]
    pub fn block_size(self) -> u64 {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_1
            | Self::IQ4_NL => 32,
            Self::Q2_0 => 64,
            Self::Q1_0 | Self::PQ2_0 | Self::PTQ1_0 => 128,
            Self::Q2_K
            | Self::Q3_K
            | Self::Q4_K
            | Self::Q5_K
            | Self::Q6_K
            | Self::Q8_K
            | Self::IQ4_XS
            | Self::IQ2_XS
            | Self::IQ2_S
            | Self::IQ3_XXS
            | Self::IQ3_S => 256,
        }
    }

    /// Bytes one block occupies on disk (ggml's type sizes).
    #[must_use]
    pub fn bytes_per_block(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18,    // f16 d + 16 B qs
            Self::Q4_1 => 20,    // f16 d + f16 m + 16 B qs
            Self::Q5_0 => 22,    // f16 d + 4 B qh + 16 B qs
            Self::Q5_1 => 24,    // f16 d + f16 m + 4 B qh + 16 B qs
            Self::Q8_0 => 34,    // f16 d + 32 i8
            Self::Q8_1 => 36,    // f16 d + f16 s + 32 i8
            Self::Q2_K => 84,    // 16 B scales + 64 B qs + f16 d + f16 dmin
            Self::Q3_K => 110,   // 32 B hmask + 64 B qs + 12 B scales + f16 d
            Self::Q4_K => 144,   // f16 d + f16 dmin + 12 B scales + 128 B qs
            Self::Q5_K => 176,   // Q4_K + 32 B qh
            Self::Q6_K => 210,   // 128 B ql + 64 B qh + 16 i8 scales + f16 d
            Self::Q8_K => 292,   // f32 d + 256 i8 + 16 i16 bsums
            Self::IQ4_XS => 136, // f16 d + u16 scales_h + 4 B scales_l + 128 B qs
            Self::IQ4_NL => 18,  // f16 d + 16 B qs
            Self::IQ2_XS => 74,  // f16 d + 32 u16 (grid|signs) + 8 B scales
            Self::IQ2_S => 82,   // f16 d + 64 B qs(idx lo + signs) + 8 B qh + 8 B scales
            Self::IQ3_XXS => 98, // f16 d + 64 B idx + 32 B (scale|signs) words
            Self::IQ3_S => 110,  // f16 d + 64 B qs + 8 B qh + 32 B signs + 4 B scales
            Self::Q1_0 => 18,    // f16 d + 16 B of bits
            Self::Q2_0 => 18,    // f16 d + 16 B of 2-bit slots
            Self::PQ2_0 => 34,   // f16 d + 32 B of 2-bit slots
            Self::PTQ1_0 => 28,  // 24 B qs (5 trits/byte) + 2 B qh (4 trits/byte) + f16 d
        }
    }

    /// Does every value of this dtype decode to a scale times one of
    /// `{−1, 0, +1}` — the ternary (and binary) family Prism's Bonsai models
    /// ship in? Two bits of information per value: a pack storing f16/f32
    /// copies of such a tensor holds bytes without information, and its
    /// integer levels are as good as their block layout allows (exact with
    /// scale blocks along the source's own axis, 5% mean at Q4 / 0.3% at Q8
    /// with the pack's blocks along the output axis — measured on the real
    /// 27B). Q2_0's reserved `+2` code is never emitted by a shipped
    /// checkpoint (histogrammed: zero occurrences).
    #[must_use]
    pub fn is_ternary_family(self) -> bool {
        matches!(self, Self::Q1_0 | Self::Q2_0 | Self::PQ2_0 | Self::PTQ1_0)
    }
}

/// One entry of the tensor table: where a tensor lives and what shape/dtype
/// it has. `offset` is relative to [`GgufFile::data_offset`] and is a
/// multiple of the file's alignment (validated on read).
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    /// Dimensions in ggml order (fastest-varying first — the *reverse* of
    /// the row-major order safetensors/PyTorch shapes use).
    pub dims: Vec<u64>,
    pub dtype: GgmlType,
    pub offset: u64,
}

impl GgufTensorInfo {
    /// Total element count.
    #[must_use]
    pub fn element_count(&self) -> u64 {
        self.dims.iter().product()
    }

    /// Exact on-disk payload size. Element counts are validated to be whole
    /// blocks at parse time, so this is always exact for parsed tensors.
    #[must_use]
    pub fn byte_len(&self) -> u64 {
        let elements = self.element_count();
        debug_assert!(
            elements.is_multiple_of(self.dtype.block_size()),
            "parse validated whole blocks"
        );
        elements / self.dtype.block_size() * self.dtype.bytes_per_block()
    }
}

/// How one GGUF tensor lands in the safetensors blob
/// ([`GgufFile::dequant_to_safetensors`]).
#[derive(Debug, Clone)]
pub enum GgufMap {
    /// Target name; shape = the GGUF dims reversed (the row-major twin of
    /// the ggml layout — right for everything but squeezed kernels).
    Rename(String),
    /// Target name + explicit row-major shape (same element count, same
    /// bytes — e.g. un-squeezing a depthwise conv kernel back to
    /// `[channels, 1, k]`).
    Reshape(String, Vec<u64>),
    /// Deliberately drop this tensor (e.g. qwen35's unused NextN/MTP block).
    /// Distinct from an unmapped name, which stays a loud error.
    Skip,
}

/// Where one tensor lands in the output blob, decided before any payload
/// byte is read ([`GgufFile::plan_dequant`]).
#[derive(Debug)]
struct PlannedTensor {
    /// Index into [`GgufFile::tensors`] — the payload to dequantize.
    source: usize,
    name: String,
    shape: Vec<u64>,
    /// Byte offset within the payload; contiguous and ascending.
    start: u64,
    /// Dequantized f32 byte length.
    len: u64,
}

/// Upper bound on shards in one split GGUF. `gguf-split` sets are a handful
/// of files (the 111 GB Flash-Next release is 4); this only exists so a
/// corrupt `split.count` cannot drive an unbounded open loop.
const MAX_GGUF_SHARDS: u64 = 64;

/// Every shard path of a split set, derived from one member's filename.
///
/// `gguf-split` names shards `<stem>-%05d-of-%05d.gguf`, so the set is found
/// by rewriting the index in place rather than by scanning the directory —
/// which would also sweep up unrelated models sharing it. Returns paths in
/// shard order, 1..=count.
fn shard_paths(member: &Path, count: usize) -> Result<Vec<std::path::PathBuf>, GgufError> {
    let bad = |reason: String| GgufError::BadValue {
        path: member.display().to_string(),
        key: "split filename".into(),
        reason,
    };
    let dir = member.parent().unwrap_or_else(|| Path::new("."));
    let name = member
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| bad("shard filename is not valid UTF-8".into()))?;
    // `<stem>-00002-of-00004.gguf` -> stem `<stem>`. Split from the RIGHT:
    // a model name may itself contain "-of-".
    let stem = name
        .strip_suffix(".gguf")
        .ok_or_else(|| bad("shard filename does not end in .gguf".into()))?;
    let (head, _tail) = stem.rsplit_once("-of-").ok_or_else(|| {
        bad(format!(
            "{name}: split.count > 1 but the name has no -of- index"
        ))
    })?;
    let (base, _index) = head
        .rsplit_once('-')
        .ok_or_else(|| bad(format!("{name}: no -NNNNN- shard index before -of-")))?;
    let mut out = Vec::with_capacity(count);
    for i in 1..=count {
        out.push(dir.join(format!("{base}-{i:05}-of-{count:05}.gguf")));
    }
    // Positive space: every shard the set claims must actually be present,
    // and saying so here beats a confusing parse error on a missing file.
    for p in &out {
        if !p.is_file() {
            return Err(bad(format!("missing shard {}", p.display())));
        }
    }
    assert_eq!(out.len(), count, "one path per shard");
    out.first()
        .map(|_| ())
        .ok_or_else(|| bad("a split set needs at least one shard".into()))?;
    Ok(out)
}

/// One shard's payload, for a model split across several `.gguf` files
/// (llama.cpp's `gguf-split` layout: `<stem>-00001-of-0000N.gguf`, with
/// `split.no` / `split.count` / `split.tensors.count` in each shard's KV).
///
/// A tensor's `offset` is relative to the payload base of **its own shard**,
/// not the first one, so the file and the base have to travel together.
/// [`GgufFile::shards`] is EMPTY for an ordinary single-file GGUF, where
/// `GgufFile::{path, data_offset}` already locate every tensor — that keeps
/// every existing caller and every hand-built test fixture correct by
/// construction rather than by remembering to fill a field in.
#[derive(Debug, Clone)]
pub struct GgufShard {
    /// The `.gguf` file holding this shard's payload.
    pub path: std::path::PathBuf,
    /// Absolute offset in `path` where this shard's tensor payload begins.
    pub data_offset: u64,
    /// Half-open range into [`GgufFile::tensors`] owned by this shard.
    /// Ranges partition the table in shard order and never overlap.
    pub tensors: std::ops::Range<usize>,
}

/// A parsed GGUF header: typed metadata + the located tensor table.
#[derive(Debug)]
pub struct GgufFile {
    /// Where this header was read from (payload reads re-open it).
    pub path: std::path::PathBuf,
    pub version: u32,
    /// Metadata in file order (keys are unique per the spec).
    pub metadata: Vec<(String, GgufValue)>,
    pub tensors: Vec<GgufTensorInfo>,
    /// Payload alignment (`general.alignment`, default 32).
    pub alignment: u64,
    /// Absolute file offset where the aligned tensor payload blob begins.
    pub data_offset: u64,
    /// Payload location per shard when this header came from a SPLIT model
    /// (see [`GgufShard`]). Empty for a single-file GGUF.
    pub shards: Vec<GgufShard>,
}

impl GgufFile {
    /// Read and validate a GGUF header (metadata + tensor table only — no
    /// tensor payloads are loaded).
    pub fn open(path: &Path) -> Result<Self, GgufError> {
        let file = File::open(path).map_err(|source| GgufError::Io {
            path: path.display().to_string(),
            source,
        })?;
        let mut r = Reader {
            inner: BufReader::new(file),
            path: path.display().to_string(),
        };
        let parsed = r.read_file(path)?;
        // Positive space: the header must end before the payload it locates.
        assert!(
            parsed.data_offset.is_multiple_of(parsed.alignment),
            "data offset is aligned by construction"
        );
        Ok(parsed)
    }

    /// Open a GGUF that may be SPLIT across several files, presenting the
    /// whole set as one logical tensor namespace.
    ///
    /// `path` may be any shard of the set (conventionally the first). When
    /// the header carries no `split.count`, or claims a count of 1, this is
    /// exactly [`Self::open`] and `shards` stays empty — so pointing this at
    /// an ordinary single-file GGUF costs one extra metadata lookup and
    /// changes nothing else.
    ///
    /// Shards are located by llama.cpp's `gguf-split` naming
    /// (`<stem>-00001-of-0000N.gguf`) rather than by trusting `split.no`
    /// ordering, then each is opened and its tensor table appended in shard
    /// order. Metadata is taken from the shard that HAS it: `gguf-split`
    /// writes the full KV into the first shard and a stub into the rest, and
    /// a real 4-shard model was observed with **0 tensors in shard 1**, so
    /// neither "metadata lives with tensors" nor "every shard has tensors"
    /// can be assumed.
    pub fn open_sharded(path: &Path) -> Result<Self, GgufError> {
        let first = Self::open(path)?;
        let Some(count) = first.get("split.count").and_then(GgufValue::as_u64) else {
            return Ok(first);
        };
        if count <= 1 {
            return Ok(first);
        }
        // Bounded: a split set is a handful of files. Refusing an absurd
        // count keeps a corrupt header from driving an unbounded open loop.
        if count > MAX_GGUF_SHARDS {
            return Err(GgufError::BadValue {
                path: path.display().to_string(),
                key: "split.count".into(),
                reason: format!("{count} shards exceeds the {MAX_GGUF_SHARDS} bound"),
            });
        }
        let count = usize::try_from(count).map_err(|_| GgufError::OverBound {
            path: path.display().to_string(),
            what: "split.count",
            count,
            bound: usize::MAX as u64,
        })?;
        let paths = shard_paths(path, count)?;
        assert_eq!(paths.len(), count, "one path per declared shard");

        let mut merged: Option<Self> = None;
        let mut tensors: Vec<GgufTensorInfo> = Vec::new();
        let mut shards: Vec<GgufShard> = Vec::with_capacity(count);
        for shard_path in &paths {
            let shard = Self::open(shard_path)?;
            let start = tensors.len();
            tensors.extend(shard.tensors.iter().cloned());
            shards.push(GgufShard {
                path: shard_path.clone(),
                data_offset: shard.data_offset,
                tensors: start..tensors.len(),
            });
            // Keep the header whose metadata is the real one: the richest KV
            // wins, which is the full-metadata shard regardless of position.
            let better = merged
                .as_ref()
                .is_none_or(|m| shard.metadata.len() > m.metadata.len());
            if better {
                merged = Some(shard);
            }
        }
        let mut out = merged.expect("at least one shard opened");

        // Negative space: a duplicate name would make lookup order decide
        // which payload is read, i.e. silently wrong weights.
        let mut seen: std::collections::HashSet<&str> =
            std::collections::HashSet::with_capacity(tensors.len());
        for t in &tensors {
            if !seen.insert(t.name.as_str()) {
                return Err(GgufError::BadValue {
                    path: path.display().to_string(),
                    key: t.name.clone(),
                    reason: "tensor name appears in more than one shard".into(),
                });
            }
        }
        // Cross-check against the count the writer recorded, when it did.
        if let Some(declared) = first
            .get("split.tensors.count")
            .and_then(GgufValue::as_u64)
            .or_else(|| {
                first
                    .get("split.tensors.count")
                    .and_then(|v| v.as_i64().and_then(|i| u64::try_from(i).ok()))
            })
            && declared != tensors.len() as u64
        {
            return Err(GgufError::BadValue {
                path: path.display().to_string(),
                key: "split.tensors.count".into(),
                reason: format!(
                    "header declares {declared} tensors, shards carry {}",
                    tensors.len()
                ),
            });
        }

        out.tensors = tensors;
        out.shards = shards;
        Ok(out)
    }

    /// Read one tensor's payload and dequantize it to f32, in the on-disk
    /// (ggml fastest-varying-first) element order.
    pub fn read_tensor_f32(&self, name: &str) -> Result<Vec<f32>, GgufError> {
        use std::io::SeekFrom;
        let index = self.tensor_index(name).ok_or_else(|| GgufError::BadValue {
            path: self.path.display().to_string(),
            key: name.to_string(),
            reason: "no such tensor".into(),
        })?;
        let info = &self.tensors[index];
        // Split models keep each tensor's offset relative to ITS OWN shard's
        // payload base, so the file and the base must be resolved together.
        let (payload_path, payload_base) = self.payload_location(index);
        let payload_path = payload_path.to_path_buf();
        let mut file = File::open(&payload_path).map_err(|source| GgufError::Io {
            path: payload_path.display().to_string(),
            source,
        })?;
        file.seek(SeekFrom::Start(payload_base + info.offset))
            .map_err(|source| GgufError::Io {
                path: payload_path.display().to_string(),
                source,
            })?;
        let byte_len = usize::try_from(info.byte_len()).map_err(|_| GgufError::OverBound {
            path: self.path.display().to_string(),
            what: "tensor payload bytes",
            count: info.byte_len(),
            bound: usize::MAX as u64,
        })?;
        let mut bytes = vec![0u8; byte_len];
        file.read_exact(&mut bytes)
            .map_err(|source| GgufError::Io {
                path: self.path.display().to_string(),
                source,
            })?;
        let out = dequantize(info.dtype, &bytes).map_err(|reason| GgufError::BadTensor {
            path: self.path.display().to_string(),
            index: 0,
            reason: format!("{name}: {reason}"),
        })?;
        assert_eq!(
            out.len() as u64,
            info.element_count(),
            "dequant must yield exactly the tensor's elements"
        );
        Ok(out)
    }

    /// Dequantize every tensor to f32 and serialize the result as an
    /// in-memory **safetensors** file — the bridge onto the exact store
    /// pipeline (adapters, key remaps, checked load) the safetensors path
    /// already trusts. `map` maps each GGUF tensor to the name (and
    /// optionally an explicit shape) the blob should carry (HF-checkpoint
    /// naming, so per-model remap tables apply unchanged); an unmapped
    /// tensor is a loud error, never a skip — a name this crate doesn't
    /// recognize means weights would silently vanish from the model.
    ///
    /// Default shapes are the GGUF dims **reversed**: ggml orders dims
    /// fastest-varying-first, so the raw payload bytes are exactly the
    /// row-major layout of the reversed shape — same bytes, HF convention.
    /// [`GgufMap::Reshape`] overrides that for tensors whose checkpoint
    /// shape differs by more than dim order (e.g. llama.cpp squeezes the
    /// middle 1 out of depthwise-conv kernels).
    ///
    /// This form needs the whole f32 payload resident. For a model whose
    /// dequantized size is a meaningful fraction of RAM, use
    /// [`Self::dequant_to_safetensors_file`] — the two produce byte-identical
    /// output.
    pub fn dequant_to_safetensors(
        &self,
        map: &dyn Fn(&GgufTensorInfo) -> Option<GgufMap>,
    ) -> Result<Vec<u8>, GgufError> {
        let mut blob = Vec::new();
        self.dequant_into(map, &mut blob)?;
        assert!(blob.len() > 8, "the header length prefix is always written");
        Ok(blob)
    }

    /// [`Self::dequant_to_safetensors`] straight to a file, never holding the
    /// payload in RAM. Returns the payload bytes written (header excluded).
    ///
    /// This is the variant a real quantized checkpoint wants. The in-memory
    /// form needs the whole f32 payload resident — ~28 GB for OLMoE-1B-7B —
    /// *on top of* the model the load then builds from it, and that sum is
    /// what a 128 GB box with other tenants actually fails to satisfy.
    /// Writing to disk trades the spike for temp space and lets
    /// `SafetensorsStore::from_file` page the weights in as it needs them.
    /// (`safetensors::fuse_checkpoint_to_file` is the same trade on the
    /// unquantized path.)
    pub fn dequant_to_safetensors_file(
        &self,
        map: &dyn Fn(&GgufTensorInfo) -> Option<GgufMap>,
        out: &Path,
    ) -> Result<u64, GgufError> {
        assert!(!out.as_os_str().is_empty(), "the sink file must be named");
        let io = |source: std::io::Error| GgufError::Io {
            path: out.display().to_string(),
            source,
        };
        let file = File::create(out).map_err(io)?;
        let mut sink = std::io::BufWriter::with_capacity(DEQUANT_STAGE_BYTES, file);
        let written = self.dequant_into(map, &mut sink)?;
        sink.flush().map_err(io)?;
        sink.into_inner()
            .map_err(|e| GgufError::Io {
                path: out.display().to_string(),
                source: e.into_error(),
            })?
            .sync_all()
            .map_err(io)?;
        Ok(written)
    }

    /// The shared dequant: plan, then stream header + payload into `sink` in
    /// output order, one tensor at a time.
    ///
    /// Planning first is not only what makes streaming possible — it moves
    /// every *claim* check (unmapped name, reshape that changes the element
    /// count, rename collision) ahead of the first payload byte, so a bad map
    /// fails in milliseconds instead of after N tensors have been
    /// dequantized.
    ///
    /// # Why the progress bar is fed from HERE
    ///
    /// This loop is the load, for every architecture that goes through
    /// [`crate::import::gguf_store`] — qwen2, qwen3, lfm2 and olmoe. It reads
    /// every tensor off the checkpoint, dequantizes it and writes it out;
    /// what follows is burn's store installing weights that are already on
    /// the fast path. Only qwen35 had a counted bar, so for those four
    /// families the whole release rendered as a spinner. The count is
    /// published here rather than in each of the four ports because here is
    /// the only place that has it, and four copies of the same three lines is
    /// how the four drift apart.
    ///
    /// The plan is the denominator on purpose: it excludes [`GgufMap::Skip`]
    /// tensors, so the bar counts the tensors that will actually be read
    /// rather than the ones the file happens to contain.
    fn dequant_into<W: std::io::Write>(
        &self,
        map: &dyn Fn(&GgufTensorInfo) -> Option<GgufMap>,
        sink: &mut W,
    ) -> Result<u64, GgufError> {
        let path = self.path.display().to_string();
        let plan = self.plan_dequant(map)?;
        let total: u64 = plan.iter().map(|p| p.len).sum();

        // Header first: names, dtypes, shapes, and the contiguous offsets the
        // copy pass will fill.
        let mut header = String::from("{");
        for (i, p) in plan.iter().enumerate() {
            if i > 0 {
                header.push(',');
            }
            let json_name = serde_json::to_string(&p.name).map_err(|e| GgufError::BadTensor {
                path: path.clone(),
                index: p.source,
                reason: format!("name is not encodable as JSON: {e}"),
            })?;
            header.push_str(&format!(
                "{json_name}:{{\"dtype\":\"F32\",\"shape\":{:?},\"data_offsets\":[{},{}]}}",
                p.shape,
                p.start,
                p.start + p.len,
            ));
        }
        header.push('}');

        let io = |source: std::io::Error| GgufError::Io {
            path: path.clone(),
            source,
        };
        sink.write_all(&(header.len() as u64).to_le_bytes())
            .map_err(io)?;
        sink.write_all(header.as_bytes()).map_err(io)?;

        // One small staging buffer, reused for every tensor: peak allocation
        // is the widest tensor's f32 values (from `read_tensor_f32`) plus this
        // 1 MiB, never a second copy of the payload.
        let mut stage: Vec<u8> = Vec::with_capacity(DEQUANT_STAGE_BYTES);
        let mut written = 0u64;
        crate::progress::begin(
            crate::progress::Phase::Loading,
            plan.len() as u64,
            crate::progress::Unit::Tensors,
        );
        // Bytes as they come OFF THE DISK, not as they land: a Q4 checkpoint
        // expands eightfold on the way through, and reporting the expansion
        // would claim a read rate the spinning array never delivered.
        let mut read_bytes = 0u64;
        for (done, p) in plan.iter().enumerate() {
            debug_assert_eq!(written, p.start, "tensors are written in output order");
            let source = &self.tensors[p.source];
            crate::progress::advance(done as u64, read_bytes);
            read_bytes += source.byte_len();
            let values = self.read_tensor_f32(&source.name)?;
            debug_assert_eq!(
                values.len() as u64 * 4,
                p.len,
                "the plan sized this tensor from the same element count"
            );
            for chunk in values.chunks(DEQUANT_STAGE_BYTES / 4) {
                stage.clear();
                stage.extend(chunk.iter().flat_map(|v| v.to_le_bytes()));
                sink.write_all(&stage).map_err(io)?;
                written += stage.len() as u64;
            }
        }
        // The advance above fires BEFORE each tensor, so the last one it
        // reported was n-1 of n. Close the count instead of leaving the bar
        // one tensor short of a pass that finished.
        crate::progress::advance(plan.len() as u64, read_bytes);
        assert_eq!(
            written, total,
            "every planned byte was written exactly once"
        );
        assert!(
            stage.capacity() <= DEQUANT_STAGE_BYTES,
            "the staging buffer never grew past its bound"
        );
        Ok(written)
    }

    /// First pass: decide what every tensor is called, what shape it claims,
    /// and where its bytes land — reading no payload at all.
    ///
    /// Offsets are contiguous and ascending in tensor-table order, which is
    /// what lets the copy pass be a single forward stream. A
    /// [`GgufMap::Skip`] tensor (e.g. qwen35's unused NextN block) is dropped
    /// here: it earns no plan entry and counts toward neither the size bound
    /// nor the blob.
    fn plan_dequant(
        &self,
        map: &dyn Fn(&GgufTensorInfo) -> Option<GgufMap>,
    ) -> Result<Vec<PlannedTensor>, GgufError> {
        let bad = |index: usize, reason: String| GgufError::BadTensor {
            path: self.path.display().to_string(),
            index,
            reason,
        };

        let mut plan: Vec<PlannedTensor> = Vec::with_capacity(self.tensors.len());
        let mut names = std::collections::HashSet::with_capacity(self.tensors.len());
        let mut start = 0u64;
        for (index, info) in self.tensors.iter().enumerate() {
            let (name, shape) = match map(info) {
                Some(GgufMap::Rename(name)) => {
                    (name, info.dims.iter().rev().copied().collect::<Vec<u64>>())
                }
                Some(GgufMap::Reshape(name, shape)) => {
                    if shape.iter().product::<u64>() != info.element_count() {
                        return Err(bad(
                            index,
                            format!(
                                "reshape of '{}' to {shape:?} changes the element count",
                                info.name
                            ),
                        ));
                    }
                    (name, shape)
                }
                Some(GgufMap::Skip) => continue,
                None => {
                    return Err(bad(index, format!("unmapped tensor name '{}'", info.name)));
                }
            };
            if !names.insert(name.clone()) {
                return Err(bad(index, format!("rename collision on '{name}'")));
            }
            let len = info.element_count() * 4;
            if len > MAX_TENSOR_F32_BYTES {
                return Err(GgufError::OverBound {
                    path: self.path.display().to_string(),
                    what: "single dequantized tensor bytes",
                    count: len,
                    bound: MAX_TENSOR_F32_BYTES,
                });
            }
            plan.push(PlannedTensor {
                source: index,
                name,
                shape,
                start,
                len,
            });
            start += len;
        }
        // The bound is on the payload actually planned (skips excluded); this
        // reads no payload, so checking after the plan loop still fails before
        // `dequant_into` copies a single byte.
        if start > MAX_DEQUANT_BYTES {
            return Err(GgufError::OverBound {
                path: self.path.display().to_string(),
                what: "dequantized f32 payload bytes",
                count: start,
                bound: MAX_DEQUANT_BYTES,
            });
        }
        Ok(plan)
    }

    /// Look up a metadata value by exact key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// The model architecture (`general.architecture`), when present.
    #[must_use]
    pub fn architecture(&self) -> Option<&str> {
        self.get("general.architecture").and_then(GgufValue::as_str)
    }

    /// Look up a tensor by exact name.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Index of a tensor by exact name — the shard-aware sibling of
    /// [`Self::tensor`], since locating a payload needs the position, not
    /// just the info.
    #[must_use]
    pub fn tensor_index(&self, name: &str) -> Option<usize> {
        self.tensors.iter().position(|t| t.name == name)
    }

    /// Which file holds tensor `index`'s payload, and that file's payload
    /// base. On a single-file GGUF this is always `(self.path,
    /// self.data_offset)`; on a split model it is the owning shard's.
    ///
    /// # Panics
    ///
    /// Never for an index produced by [`Self::tensor_index`] on a header
    /// this type built: shard ranges partition the tensor table by
    /// construction, which the debug assertion below restates.
    #[must_use]
    pub fn payload_location(&self, index: usize) -> (&Path, u64) {
        debug_assert!(index < self.tensors.len(), "tensor index in range");
        if self.shards.is_empty() {
            return (&self.path, self.data_offset);
        }
        let shard = self
            .shards
            .iter()
            .find(|s| s.tensors.contains(&index))
            .unwrap_or_else(|| {
                // Negative space: a split header whose ranges do not cover
                // every tensor is a parse bug, not a user error — reading
                // from the wrong shard would return silent garbage.
                panic!(
                    "tensor {index} belongs to no shard ({} shards)",
                    self.shards.len()
                )
            });
        (&shard.path, shard.data_offset)
    }

    /// Number of shards backing this header (1 for a single-file GGUF).
    #[must_use]
    pub fn shard_count(&self) -> usize {
        self.shards.len().max(1)
    }
}

/// Sequential little-endian reader over the header bytes.
struct Reader {
    inner: BufReader<File>,
    path: String,
}

impl Reader {
    fn io_err(&self, source: std::io::Error) -> GgufError {
        GgufError::Io {
            path: self.path.clone(),
            source,
        }
    }

    fn bytes<const N: usize>(&mut self) -> Result<[u8; N], GgufError> {
        let mut buf = [0u8; N];
        self.inner
            .read_exact(&mut buf)
            .map_err(|e| self.io_err(e))?;
        Ok(buf)
    }

    fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }

    fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }

    /// A length-prefixed UTF-8 string, bounded by [`MAX_STRING_BYTES`].
    fn string(&mut self, what: &'static str) -> Result<String, GgufError> {
        let len = self.u64()?;
        if len > MAX_STRING_BYTES {
            return Err(GgufError::OverBound {
                path: self.path.clone(),
                what,
                count: len,
                bound: MAX_STRING_BYTES,
            });
        }
        let mut buf = vec![0u8; to_usize(len, &self.path, what)?];
        self.inner
            .read_exact(&mut buf)
            .map_err(|e| self.io_err(e))?;
        String::from_utf8(buf).map_err(|e| GgufError::BadValue {
            path: self.path.clone(),
            key: what.to_string(),
            reason: format!("invalid UTF-8: {e}"),
        })
    }

    /// One typed metadata value. `depth` bounds array nesting.
    fn value(&mut self, key: &str, type_id: u32, depth: u32) -> Result<GgufValue, GgufError> {
        let bad = |reason: String, path: &str| GgufError::BadValue {
            path: path.to_string(),
            key: key.to_string(),
            reason,
        };
        let v = match type_id {
            0 => GgufValue::U8(self.bytes::<1>()?[0]),
            #[allow(clippy::cast_possible_wrap)] // bit-exact reinterpret is the format
            1 => GgufValue::I8(self.bytes::<1>()?[0] as i8),
            2 => GgufValue::U16(u16::from_le_bytes(self.bytes()?)),
            3 => GgufValue::I16(i16::from_le_bytes(self.bytes()?)),
            4 => GgufValue::U32(self.u32()?),
            5 => GgufValue::I32(i32::from_le_bytes(self.bytes()?)),
            6 => GgufValue::F32(f32::from_le_bytes(self.bytes()?)),
            7 => match self.bytes::<1>()?[0] {
                0 => GgufValue::Bool(false),
                1 => GgufValue::Bool(true),
                other => return Err(bad(format!("bool byte {other}"), &self.path)),
            },
            8 => GgufValue::Str(self.string("metadata string")?),
            9 => {
                if depth >= 2 {
                    return Err(bad("arrays nested deeper than 2".into(), &self.path));
                }
                let elem_type = self.u32()?;
                let len = self.u64()?;
                if len > MAX_ARRAY_LEN {
                    return Err(GgufError::OverBound {
                        path: self.path.clone(),
                        what: "metadata array",
                        count: len,
                        bound: MAX_ARRAY_LEN,
                    });
                }
                let mut items = Vec::with_capacity(to_usize(len, &self.path, "metadata array")?);
                for _ in 0..len {
                    items.push(self.value(key, elem_type, depth + 1)?);
                }
                GgufValue::Array(items)
            }
            10 => GgufValue::U64(self.u64()?),
            11 => GgufValue::I64(i64::from_le_bytes(self.bytes()?)),
            12 => GgufValue::F64(f64::from_le_bytes(self.bytes()?)),
            other => return Err(bad(format!("unknown value type {other}"), &self.path)),
        };
        Ok(v)
    }

    /// The whole header: magic, version, metadata, tensor table, alignment.
    fn read_file(&mut self, source_path: &Path) -> Result<GgufFile, GgufError> {
        let magic = self.bytes::<4>()?;
        if magic != MAGIC {
            return Err(GgufError::BadMagic {
                path: self.path.clone(),
                found: magic,
            });
        }
        let version = self.u32()?;
        if !SUPPORTED_VERSIONS.contains(&version) {
            return Err(GgufError::UnsupportedVersion {
                path: self.path.clone(),
                version,
            });
        }
        let tensor_count = self.u64()?;
        let kv_count = self.u64()?;
        for (what, count, bound) in [
            ("tensor", tensor_count, MAX_TENSORS),
            ("metadata kv", kv_count, MAX_KVS),
        ] {
            if count > bound {
                return Err(GgufError::OverBound {
                    path: self.path.clone(),
                    what,
                    count,
                    bound,
                });
            }
        }

        let mut metadata =
            Vec::with_capacity(to_usize(kv_count, &self.path, "metadata key-values")?);
        for _ in 0..kv_count {
            let key = self.string("metadata key")?;
            let type_id = self.u32()?;
            let value = self.value(&key, type_id, 0)?;
            metadata.push((key, value));
        }

        let alignment = metadata
            .iter()
            .find(|(k, _)| k == "general.alignment")
            .and_then(|(_, v)| v.as_u64())
            .unwrap_or(DEFAULT_ALIGNMENT);
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(GgufError::BadValue {
                path: self.path.clone(),
                key: "general.alignment".into(),
                reason: format!("{alignment} is not a power of two"),
            });
        }

        let tensors = self.tensor_table(tensor_count, alignment)?;
        let header_end = self.inner.stream_position().map_err(|e| self.io_err(e))?;
        let data_offset = header_end.div_ceil(alignment) * alignment;
        debug_assert!(data_offset >= header_end, "padding never rewinds");
        Ok(GgufFile {
            path: source_path.to_path_buf(),
            version,
            metadata,
            tensors,
            alignment,
            data_offset,
            // One file parsed here. `open_sharded` is what fills this in;
            // empty means "single file", which is the truth for this parse.
            shards: Vec::new(),
        })
    }

    /// The tensor table, with every entry's shape/dtype/offset validated.
    fn tensor_table(
        &mut self,
        count: u64,
        alignment: u64,
    ) -> Result<Vec<GgufTensorInfo>, GgufError> {
        assert!(count <= MAX_TENSORS, "caller bounded the count");
        assert!(alignment.is_power_of_two(), "caller validated alignment");
        let count_usize = to_usize(count, &self.path, "tensor count")?;
        let mut tensors: Vec<GgufTensorInfo> = Vec::with_capacity(count_usize);
        for index in 0..count_usize {
            let bad = |reason: String, path: &str| GgufError::BadTensor {
                path: path.to_string(),
                index,
                reason,
            };
            let name = self.string("tensor name")?;
            let n_dims = self.u32()?;
            if n_dims == 0 || n_dims > MAX_DIMS {
                return Err(bad(format!("{n_dims} dims (1..={MAX_DIMS})"), &self.path));
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(self.u64()?);
            }
            let type_id = self.u32()?;
            let Some(dtype) = GgmlType::from_id(type_id) else {
                return Err(bad(format!("unknown ggml type id {type_id}"), &self.path));
            };
            let offset = self.u64()?;
            if !offset.is_multiple_of(alignment) {
                return Err(bad(
                    format!("offset {offset} not {alignment}-aligned"),
                    &self.path,
                ));
            }
            let elements: u64 = dims.iter().product();
            if elements == 0 || !elements.is_multiple_of(dtype.block_size()) {
                return Err(bad(
                    format!("{elements} elements is not whole {dtype:?} blocks"),
                    &self.path,
                ));
            }
            if tensors.iter().any(|t| t.name == name) {
                return Err(bad(format!("duplicate tensor name '{name}'"), &self.path));
            }
            tensors.push(GgufTensorInfo {
                name,
                dims,
                dtype,
                offset,
            });
        }
        // Payloads must not overlap: a table whose declared dtype is not the
        // file's real packing (Prism's legacy `*-Q2_0.gguf` stores group-128
        // blocks under the group-64 id 42) sizes every tensor wrong and would
        // otherwise read the neighbour's bytes as weights, silently.
        let mut by_offset: Vec<usize> = (0..tensors.len()).collect();
        by_offset.sort_by_key(|&i| tensors[i].offset);
        for pair in by_offset.windows(2) {
            let (a, b) = (&tensors[pair[0]], &tensors[pair[1]]);
            let end = a.offset.saturating_add(a.byte_len());
            if end > b.offset {
                return Err(GgufError::BadTensor {
                    path: self.path.clone(),
                    index: pair[1],
                    reason: format!(
                        "'{}' at offset {} overlaps '{}' ({:?}, {} bytes ending at {end}) — \
                         the table's dtypes do not match the file's packing (a legacy \
                         group-128 Q2_0 stored under id 42?)",
                        b.name,
                        b.offset,
                        a.name,
                        a.dtype,
                        a.byte_len()
                    ),
                });
            }
        }
        Ok(tensors)
    }
}

// ---- Dequantization ------------------------------------------------------
//
// Exact ports of ggml's reference dequantizers (ggml-quants.c) for every
// dtype llama.cpp stores model weights in: plain floats, the 32-element
// legacy blocks Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, and the 256-element K-quant
// superblocks Q2_K–Q6_K. Layouts follow `GgmlType::bytes_per_block`.
// Q8_1/Q8_K are activation formats (dot-product scratch), never tensor
// storage — they stay loud errors.

/// Dequantize a whole tensor payload to f32. `bytes` must be whole blocks of
/// `dtype` (guaranteed for payload slices sized by [`GgufTensorInfo::byte_len`]).
pub fn dequantize(dtype: GgmlType, bytes: &[u8]) -> Result<Vec<f32>, String> {
    let bpb = usize::try_from(dtype.bytes_per_block()).map_err(|_| {
        format!(
            "{dtype:?} block is {} bytes, wider than usize",
            dtype.bytes_per_block()
        )
    })?;
    if bytes.is_empty() || !bytes.len().is_multiple_of(bpb) {
        return Err(format!(
            "{} bytes is not whole {dtype:?} blocks of {bpb}",
            bytes.len()
        ));
    }
    let blocks = bytes.len() / bpb;
    let block_elems = usize::try_from(dtype.block_size()).map_err(|_| {
        format!(
            "{dtype:?} block holds {} elements, more than usize",
            dtype.block_size()
        )
    })?;
    let mut out = Vec::with_capacity(blocks * block_elems);
    for block in bytes.chunks_exact(bpb) {
        match dtype {
            GgmlType::F32 => {
                out.push(f32::from_le_bytes(block.try_into().map_err(|_| {
                    format!("F32 block is {} bytes, not 4", block.len())
                })?))
            }
            GgmlType::F16 => out.push(f16_to_f32(u16::from_le_bytes([block[0], block[1]]))),
            GgmlType::BF16 => {
                out.push(f32::from_bits(
                    u32::from(u16::from_le_bytes([block[0], block[1]])) << 16,
                ));
            }
            GgmlType::Q4_0 => dequant_q4_0(block, &mut out),
            GgmlType::Q4_1 => dequant_q4_1(block, &mut out),
            GgmlType::Q5_0 => dequant_q5_0(block, &mut out),
            GgmlType::Q5_1 => dequant_q5_1(block, &mut out),
            GgmlType::Q8_0 => dequant_q8_0(block, &mut out),
            GgmlType::Q2_K => dequant_q2_k(block, &mut out),
            GgmlType::Q3_K => dequant_q3_k(block, &mut out),
            GgmlType::Q4_K => dequant_q4_k(block, &mut out),
            GgmlType::Q5_K => dequant_q5_k(block, &mut out),
            GgmlType::Q6_K => dequant_q6_k(block, &mut out),
            GgmlType::IQ4_XS => dequant_iq4_xs(block, &mut out),
            GgmlType::IQ4_NL => dequant_iq4_nl(block, &mut out),
            GgmlType::IQ2_XS => dequant_iq2_xs(block, &mut out),
            GgmlType::IQ2_S => dequant_iq2_s(block, &mut out),
            GgmlType::IQ3_XXS => dequant_iq3_xxs(block, &mut out),
            GgmlType::IQ3_S => dequant_iq3_s(block, &mut out),
            GgmlType::Q1_0 => dequant_q1_0(block, &mut out),
            GgmlType::Q2_0 => dequant_q2_slots(block, 64, &mut out),
            GgmlType::PQ2_0 => dequant_q2_slots(block, 128, &mut out),
            GgmlType::PTQ1_0 => dequant_ptq1_0(block, &mut out),
            other => return Err(format!("dequant for {other:?} is not implemented yet")),
        }
    }
    assert_eq!(out.len(), blocks * block_elems, "whole blocks out");
    Ok(out)
}

/// IEEE 754 half → f32 (no `half` dep on this path; exhaustive over u16 in
/// tests against the `half` crate the workspace already carries).
fn f16_to_f32(bits: u16) -> f32 {
    f32::from(half::f16::from_bits(bits))
}

/// Q1_0: f16 scale + 128 bits, LSB first; `x = bit ? d : −d`.
fn dequant_q1_0(block: &[u8], out: &mut Vec<f32>) {
    assert_eq!(block.len(), 18, "Q1_0 block is 18 bytes");
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    for &byte in &block[2..18] {
        for bit in 0..8 {
            out.push(if (byte >> bit) & 1 == 1 { d } else { -d });
        }
    }
}

/// Q2_0 (64 elements) and PQ2_0 (128): f16 scale + 2-bit code slots, four
/// per byte with element `j` at bits `2·(j % 4)`; `x = (code − 1)·d`, so
/// `00 → −d`, `01 → 0`, `10 → +d`, `11 → +2d` (reserved — the Prism fork's
/// `dequantize_row_q2_0`/`_pq2_0`).
fn dequant_q2_slots(block: &[u8], elems: usize, out: &mut Vec<f32>) {
    assert_eq!(block.len(), 2 + elems / 4, "Q2_0-family block size");
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    for &byte in &block[2..] {
        for shift in [0u8, 2, 4, 6] {
            let code = i32::from((byte >> shift) & 0x3);
            #[allow(clippy::cast_precision_loss)] // −1..=2
            out.push((code - 1) as f32 * d);
        }
    }
}

/// PTQ1_0: 128 ternary values as base-3 trits — 24 bytes holding five
/// trits each (120 values) in two stages of 16 then 8 bytes, 2 bytes holding
/// four each (8 values), then the f16 scale. A trit is read by multiplying
/// the byte by `3^n` (mod 256) and taking `(q·3) >> 8`, the encoder having
/// stored `ceil(q_base3 · 256 / 243)` — the Prism fork's
/// `dequantize_row_ptq1_0` (TQ1_0's scheme at group 128).
fn dequant_ptq1_0(block: &[u8], out: &mut Vec<f32>) {
    assert_eq!(block.len(), 28, "PTQ1_0 block is 28 bytes");
    const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];
    const STAGES: [usize; 3] = [32, 16, 8];
    let qs = &block[0..24];
    let qh = &block[24..26];
    let d = f16_to_f32(u16::from_le_bytes([block[26], block[27]]));
    let trit = |byte: u8, n: usize| -> f32 {
        let q = byte.wrapping_mul(POW3[n]);
        let xi = i32::from((u16::from(q) * 3) >> 8);
        #[allow(clippy::cast_precision_loss)] // −1..=1
        {
            (xi - 1) as f32 * d
        }
    };
    let mut j = 0usize;
    for &c in &STAGES {
        while j + c <= qs.len() {
            for n in 0..5 {
                for m in 0..c {
                    out.push(trit(qs[j + m], n));
                }
            }
            j += c;
        }
    }
    for n in 0..4 {
        for &h in qh {
            out.push(trit(h, n));
        }
    }
}

/// Q8_0: f16 scale + 32 signed bytes; `x = d * q`.
fn dequant_q8_0(block: &[u8], out: &mut Vec<f32>) {
    assert_eq!(block.len(), 34, "Q8_0 block is 34 bytes");
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    #[allow(clippy::cast_possible_wrap)] // bit-exact reinterpret is the format
    out.extend(block[2..34].iter().map(|&q| d * f32::from(q as i8)));
}

/// Q4_0: f16 scale + 16 bytes of 4-bit quants; `x = d·(q − 8)` — all 16 low
/// nibbles are elements 0..16, the high nibbles elements 16..32.
fn dequant_q4_0(block: &[u8], out: &mut Vec<f32>) {
    assert_eq!(block.len(), 18, "Q4_0 block is 18 bytes");
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[2..18];
    out.extend(qs.iter().map(|&b| d * (f32::from(b & 0x0F) - 8.0)));
    out.extend(qs.iter().map(|&b| d * (f32::from(b >> 4) - 8.0)));
}

/// Q4_1: f16 scale + f16 min + 16 bytes of 4-bit quants; `x = d·q + m`.
fn dequant_q4_1(block: &[u8], out: &mut Vec<f32>) {
    assert_eq!(block.len(), 20, "Q4_1 block is 20 bytes");
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let m = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let qs = &block[4..20];
    out.extend(qs.iter().map(|&b| d * f32::from(b & 0x0F) + m));
    out.extend(qs.iter().map(|&b| d * f32::from(b >> 4) + m));
}

/// Q5_0: f16 scale + 4 B of packed 5th bits + 16 B of 4-bit quants;
/// `x = d·(q − 16)` with bit `j` of `qh` topping up element `j`.
fn dequant_q5_0(block: &[u8], out: &mut Vec<f32>) {
    assert_eq!(block.len(), 22, "Q5_0 block is 22 bytes");
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
    let qs = &block[6..22];
    #[allow(clippy::cast_possible_truncation)] // masked to one nibble bit
    out.extend(qs.iter().enumerate().map(|(j, &b)| {
        let hi = ((qh >> j) << 4) as u8 & 0x10;
        d * (f32::from((b & 0x0F) | hi) - 16.0)
    }));
    #[allow(clippy::cast_possible_truncation)] // masked to one nibble bit
    out.extend(qs.iter().enumerate().map(|(j, &b)| {
        let hi = (qh >> (j + 12)) as u8 & 0x10;
        d * (f32::from((b >> 4) | hi) - 16.0)
    }));
}

/// Q5_1: f16 scale + f16 min + 4 B packed 5th bits + 16 B quants; `x = d·q + m`.
fn dequant_q5_1(block: &[u8], out: &mut Vec<f32>) {
    assert_eq!(block.len(), 24, "Q5_1 block is 24 bytes");
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let m = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    let qs = &block[8..24];
    #[allow(clippy::cast_possible_truncation)] // masked to one nibble bit
    out.extend(qs.iter().enumerate().map(|(j, &b)| {
        let hi = ((qh >> j) << 4) as u8 & 0x10;
        d * f32::from((b & 0x0F) | hi) + m
    }));
    #[allow(clippy::cast_possible_truncation)] // masked to one nibble bit
    out.extend(qs.iter().enumerate().map(|(j, &b)| {
        let hi = (qh >> (j + 12)) as u8 & 0x10;
        d * f32::from((b >> 4) | hi) + m
    }));
}

/// Q2_K: 256-element superblock — 16 packed (scale, min) nibbles + 64 B of
/// 2-bit quants + f16 d + f16 dmin; `x = d·sc·q − dmin·m` over 16 sub-blocks
/// of 16.
fn dequant_q2_k(block: &[u8], out: &mut Vec<f32>) {
    assert_eq!(block.len(), 84, "Q2_K superblock is 84 bytes");
    let scales = &block[0..16];
    let qs = &block[16..80];
    let d = f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[82], block[83]]));
    let mut is = 0;
    // Two halves of 128 values; each half reads 32 quant bytes at 4 shifts.
    for q in [&qs[0..32], &qs[32..64]] {
        for shift in [0u8, 2, 4, 6] {
            for part in [&q[0..16], &q[16..32]] {
                let sc = scales[is];
                is += 1;
                let dl = d * f32::from(sc & 0x0F);
                let ml = dmin * f32::from(sc >> 4);
                out.extend(part.iter().map(|&b| dl * f32::from((b >> shift) & 3) - ml));
            }
        }
    }
    assert_eq!(is, 16, "16 sub-block scales consumed");
}

/// Unpack Q3_K's 12 packed scale bytes into 16 signed 6-bit sub-scales
/// (ggml's kmask bit dance, done bytewise).
fn q3_k_scales(packed: &[u8]) -> [i8; 16] {
    assert_eq!(packed.len(), 12, "Q3_K scale block is 12 bytes");
    let mut sc = [0i8; 16];
    #[allow(clippy::cast_possible_wrap)] // 6-bit values reinterpret exactly
    for j in 0..4 {
        let hi = packed[8 + j]; // 2-bit tops for slots j, j+4, j+8, j+12
        sc[j] = ((packed[j] & 0x0F) | ((hi & 3) << 4)) as i8;
        sc[j + 4] = ((packed[j + 4] & 0x0F) | (((hi >> 2) & 3) << 4)) as i8;
        sc[j + 8] = ((packed[j] >> 4) | (((hi >> 4) & 3) << 4)) as i8;
        sc[j + 12] = ((packed[j + 4] >> 4) | ((hi >> 6) << 4)) as i8;
    }
    sc
}

/// Q3_K: 256-element superblock — 32 B high-bit mask + 64 B of 2-bit quants
/// + 12 B packed 6-bit sub-scales + f16 d; `x = d·(sc − 32)·(q − hm·4)`.
fn dequant_q3_k(block: &[u8], out: &mut Vec<f32>) {
    assert_eq!(block.len(), 110, "Q3_K superblock is 110 bytes");
    let hmask = &block[0..32];
    let qs = &block[32..96];
    let scales = q3_k_scales(&block[96..108]);
    let d = f16_to_f32(u16::from_le_bytes([block[108], block[109]]));
    let mut is = 0;
    let mut m: u8 = 1;
    for q in [&qs[0..32], &qs[32..64]] {
        for shift in [0u8, 2, 4, 6] {
            for base in [0usize, 16] {
                let dl = d * f32::from(i16::from(scales[is]) - 32);
                is += 1;
                for l in base..base + 16 {
                    let low = i16::from((q[l] >> shift) & 3);
                    let sub = if hmask[l] & m == 0 { 4 } else { 0 };
                    out.push(dl * f32::from(low - sub));
                }
            }
            m <<= 1; // the hmask bit advances per (half, shift) pair
        }
    }
    assert_eq!(is, 16, "16 sub-block scales consumed");
}

/// The Q4_K/Q5_K 6-bit (scale, min) pair for sub-block `j` — ggml's
/// `get_scale_min_k4`.
fn scale_min_k4(scales: &[u8], j: usize) -> (f32, f32) {
    assert_eq!(scales.len(), 12, "K-quant scale block is 12 bytes");
    assert!(j < 8, "8 sub-blocks per superblock");
    let (sc, m) = if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    };
    (f32::from(sc), f32::from(m))
}

/// Q4_K: 256-element superblock — f16 d + f16 dmin + 12 B packed 6-bit
/// (scale, min) pairs + 128 B of 4-bit quants; `x = d·sc·q − dmin·m`.
fn dequant_q4_k(block: &[u8], out: &mut Vec<f32>) {
    assert_eq!(block.len(), 144, "Q4_K superblock is 144 bytes");
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let qs = &block[16..144];
    // 4 chunks of 64 values; each chunk reads 32 bytes — low nibbles first.
    for chunk in 0..4 {
        let (sc1, m1) = scale_min_k4(scales, chunk * 2);
        let (sc2, m2) = scale_min_k4(scales, chunk * 2 + 1);
        let q = &qs[chunk * 32..chunk * 32 + 32];
        out.extend(q.iter().map(|&b| d * sc1 * f32::from(b & 0x0F) - dmin * m1));
        out.extend(q.iter().map(|&b| d * sc2 * f32::from(b >> 4) - dmin * m2));
    }
}

/// Q5_K: 256-element superblock — f16 d + f16 dmin + 12 B packed 6-bit
/// (scale, min) pairs + 32 B of 5th bits + 128 B of 4-bit quants;
/// `x = d·sc·q − dmin·m` with two `qh` bits per byte per chunk.
fn dequant_q5_k(block: &[u8], out: &mut Vec<f32>) {
    assert_eq!(block.len(), 176, "Q5_K superblock is 176 bytes");
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let qh = &block[16..48];
    let qs = &block[48..176];
    let mut u1: u8 = 1;
    let mut u2: u8 = 2;
    // 4 chunks of 64 values; each chunk reads 32 quant bytes — low nibbles
    // first — and one bit-plane pair of the shared 32-byte qh.
    for chunk in 0..4 {
        let (sc1, m1) = scale_min_k4(scales, chunk * 2);
        let (sc2, m2) = scale_min_k4(scales, chunk * 2 + 1);
        let q = &qs[chunk * 32..chunk * 32 + 32];
        out.extend(q.iter().zip(qh).map(|(&b, &h)| {
            let top = if h & u1 == 0 { 0 } else { 16 };
            d * sc1 * f32::from((b & 0x0F) + top) - dmin * m1
        }));
        out.extend(q.iter().zip(qh).map(|(&b, &h)| {
            let top = if h & u2 == 0 { 0 } else { 16 };
            d * sc2 * f32::from((b >> 4) + top) - dmin * m2
        }));
        u1 <<= 2;
        u2 <<= 2;
    }
    assert_eq!(u1, 0, "four bit-plane pairs consumed"); // 1<<8 wraps to 0
}

/// Q6_K: 256-element superblock — 128 B low-4 + 64 B high-2 + 16 i8
/// sub-scales + f16 d; `x = d·sc·(q − 32)`.
fn dequant_q6_k(block: &[u8], out: &mut Vec<f32>) {
    assert_eq!(block.len(), 210, "Q6_K superblock is 210 bytes");
    let (ql_all, rest) = block.split_at(128);
    let (qh_all, rest) = rest.split_at(64);
    let (scales, d_bytes) = rest.split_at(16);
    let d = f16_to_f32(u16::from_le_bytes([d_bytes[0], d_bytes[1]]));
    let start = out.len();
    out.resize(start + 256, 0.0);
    let y = &mut out[start..];
    // Two halves of 128 values, each consuming 64 ql / 32 qh / 8 scales.
    for half_idx in 0..2 {
        let ql = &ql_all[half_idx * 64..half_idx * 64 + 64];
        let qh = &qh_all[half_idx * 32..half_idx * 32 + 32];
        let sc = &scales[half_idx * 8..half_idx * 8 + 8];
        let base = half_idx * 128;
        for l in 0..32 {
            let is = l / 16;
            let q1 = i16::from((ql[l] & 0x0F) | ((qh[l] & 3) << 4)) - 32;
            let q2 = i16::from((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            let q3 = i16::from((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            let q4 = i16::from((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            #[allow(clippy::cast_possible_wrap)] // i8 sub-scales are the format
            let s = |i: usize| f32::from(sc[i] as i8);
            y[base + l] = d * s(is) * f32::from(q1);
            y[base + l + 32] = d * s(is + 2) * f32::from(q2);
            y[base + l + 64] = d * s(is + 4) * f32::from(q3);
            y[base + l + 96] = d * s(is + 6) * f32::from(q4);
        }
    }
}

/// IQ4_NL/IQ4_XS's 16-entry non-linear value table (llama.cpp
/// `kvalues_iq4nl`, frozen with the format).
const KVALUES_IQ4NL: [f32; 16] = [
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0, 69.0,
    89.0, 113.0,
];

/// IQ4_XS: 256-element superblock — f16 d + 8 six-bit sub-scales (low 4 bits
/// packed two-per-byte in `scales_l`, high 2 bits packed in `scales_h`) +
/// 128 B of 4-bit indices into [`KVALUES_IQ4NL`]; per 32-group,
/// `x = d·(sc − 32)·kvalues[q]` with the 16 low nibbles first
/// (llama.cpp `dequantize_row_iq4_xs`).
fn dequant_iq4_xs(block: &[u8], out: &mut Vec<f32>) {
    assert_eq!(block.len(), 136, "IQ4_XS superblock is 136 bytes");
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let scales_h = u16::from_le_bytes([block[2], block[3]]);
    let scales_l = &block[4..8];
    let qs = &block[8..136];
    for ib32 in 0..8usize {
        let lo = (scales_l[ib32 / 2] >> (4 * (ib32 % 2))) & 0x0F;
        let hi = ((scales_h >> (2 * ib32)) & 3) as u8;
        let sc = i16::from(lo | (hi << 4)) - 32;
        let dl = d * f32::from(sc);
        let q = &qs[16 * ib32..16 * ib32 + 16];
        out.extend(q.iter().map(|&b| dl * KVALUES_IQ4NL[usize::from(b & 0x0F)]));
        out.extend(q.iter().map(|&b| dl * KVALUES_IQ4NL[usize::from(b >> 4)]));
    }
}

/// IQ4_NL: 32-element block — f16 d + 16 B of 4-bit indices into
/// [`KVALUES_IQ4NL`]; low nibbles are elements 0..16, high 16..32
/// (llama.cpp `dequantize_row_iq4_nl`).
fn dequant_iq4_nl(block: &[u8], out: &mut Vec<f32>) {
    assert_eq!(block.len(), 18, "IQ4_NL block is 18 bytes");
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[2..18];
    out.extend(qs.iter().map(|&b| d * KVALUES_IQ4NL[usize::from(b & 0x0F)]));
    out.extend(qs.iter().map(|&b| d * KVALUES_IQ4NL[usize::from(b >> 4)]));
}

use crate::gguf_iq_grids::{IQ2S_GRID, IQ2XS_GRID, IQ3S_GRID, IQ3XXS_GRID, KSIGNS_IQ2XS};

/// Unpack a grid word's 8 byte-magnitudes with `signs`' 8 sign bits applied
/// (ggml's `kmask_iq2xs` is just bit `j`), scaled by `dl`.
fn push_signed_row8(out: &mut Vec<f32>, dl: f32, grid: u64, signs: u8) {
    for j in 0..8 {
        let mag = f32::from((grid >> (8 * j)) as u8);
        let sign = if signs & (1 << j) != 0 { -1.0 } else { 1.0 };
        out.push(dl * mag * sign);
    }
}

/// IQ2_XS: 256-element superblock — f16 d + 32 u16 words (9-bit grid index +
/// 7-bit sign index) + 8 packed 4-bit sub-scales;
/// `x = d·(0.5 + sc)·0.25·grid·±1` (llama.cpp `dequantize_row_iq2_xs`).
fn dequant_iq2_xs(block: &[u8], out: &mut Vec<f32>) {
    assert_eq!(block.len(), 74, "IQ2_XS superblock is 74 bytes");
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qs: Vec<u16> = block[2..66]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect();
    let scales = &block[66..74];
    for ib32 in 0..8usize {
        let db = [
            d * (0.5 + f32::from(scales[ib32] & 0x0F)) * 0.25,
            d * (0.5 + f32::from(scales[ib32] >> 4)) * 0.25,
        ];
        for l in 0..4usize {
            let word = qs[4 * ib32 + l];
            let grid = IQ2XS_GRID[usize::from(word & 511)];
            let signs = KSIGNS_IQ2XS[usize::from(word >> 9)];
            push_signed_row8(out, db[l / 2], grid, signs);
        }
    }
}

/// IQ2_S: 256-element superblock — f16 d + 64 B `qs` (32 low index bytes,
/// then 32 sign bytes) + 8 B `qh` (index bits 8..10) + 8 packed sub-scales
/// (llama.cpp `dequantize_row_iq2_s`).
fn dequant_iq2_s(block: &[u8], out: &mut Vec<f32>) {
    assert_eq!(block.len(), 82, "IQ2_S superblock is 82 bytes");
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[2..34]; // low 8 bits of grid indices, 4 per 32-group
    let signs = &block[34..66]; // one sign byte per grid row
    let qh = &block[66..74];
    let scales = &block[74..82];
    for ib32 in 0..8usize {
        let db = [
            d * (0.5 + f32::from(scales[ib32] & 0x0F)) * 0.25,
            d * (0.5 + f32::from(scales[ib32] >> 4)) * 0.25,
        ];
        for l in 0..4usize {
            let hi = (u16::from(qh[ib32]) << (8 - 2 * l)) & 0x300;
            let grid = IQ2S_GRID[usize::from(u16::from(qs[4 * ib32 + l]) | hi)];
            push_signed_row8(out, db[l / 2], grid, signs[4 * ib32 + l]);
        }
    }
}

/// Unpack a u32 grid word's 4 byte-magnitudes with 4 sign bits (offset
/// `bit0` into ggml's 8-bit sign mask), scaled by `dl`.
fn push_signed_row4(out: &mut Vec<f32>, dl: f32, grid: u32, signs: u8, bit0: u8) {
    for j in 0..4u8 {
        let mag = f32::from((grid >> (8 * j)) as u8);
        let sign = if signs & (1 << (bit0 + j)) != 0 {
            -1.0
        } else {
            1.0
        };
        out.push(dl * mag * sign);
    }
}

/// IQ3_XXS: 256-element superblock — f16 d + 64 index bytes (into the
/// 256-entry grid, 4 values each) + 8 u32 words carrying a 4-bit scale and
/// four 7-bit sign indices (llama.cpp `dequantize_row_iq3_xxs`).
fn dequant_iq3_xxs(block: &[u8], out: &mut Vec<f32>) {
    assert_eq!(block.len(), 98, "IQ3_XXS superblock is 98 bytes");
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[2..66];
    let sas = &block[66..98]; // scales-and-signs, one u32 per 32-group
    for ib32 in 0..8usize {
        let aux32 = u32::from_le_bytes(sas[4 * ib32..4 * ib32 + 4].try_into().expect("4"));
        let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
        for l in 0..4usize {
            let signs = KSIGNS_IQ2XS[usize::try_from((aux32 >> (7 * l)) & 127).expect("7 bits")];
            let g1 = IQ3XXS_GRID[usize::from(qs[8 * ib32 + 2 * l])];
            let g2 = IQ3XXS_GRID[usize::from(qs[8 * ib32 + 2 * l + 1])];
            push_signed_row4(out, db, g1, signs, 0);
            push_signed_row4(out, db, g2, signs, 4);
        }
    }
}

/// IQ3_S: 256-element superblock — f16 d + 64 index bytes + 8 B `qh` (index
/// bit 8) + 32 sign bytes + 4 packed 4-bit sub-scales;
/// `x = d·(1 + 2·sc)·grid·±1` (llama.cpp `dequantize_row_iq3_s`).
fn dequant_iq3_s(block: &[u8], out: &mut Vec<f32>) {
    assert_eq!(block.len(), 110, "IQ3_S superblock is 110 bytes");
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[2..66];
    let qh = &block[66..74];
    let signs = &block[74..106];
    let scales = &block[106..110];
    for ib32 in 0..8usize {
        let db = d * (1.0 + 2.0 * f32::from((scales[ib32 / 2] >> (4 * (ib32 % 2))) & 0x0F));
        for l in 0..4usize {
            let h = u16::from(qh[ib32]);
            let i1 = usize::from(u16::from(qs[8 * ib32 + 2 * l]) | ((h << (8 - 2 * l)) & 256));
            let i2 = usize::from(u16::from(qs[8 * ib32 + 2 * l + 1]) | ((h << (7 - 2 * l)) & 256));
            let sign_byte = signs[4 * ib32 + l];
            push_signed_row4(out, db, IQ3S_GRID[i1], sign_byte, 0);
            push_signed_row4(out, db, IQ3S_GRID[i2], sign_byte, 4);
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Write;

    /// Minimal in-memory GGUF builder for tests.
    pub(crate) struct TestGguf {
        buf: Vec<u8>,
        tensor_count: u64,
        kv_count: u64,
        kvs: Vec<u8>,
        tensors: Vec<u8>,
    }

    impl TestGguf {
        pub(crate) fn new() -> Self {
            Self {
                buf: Vec::new(),
                tensor_count: 0,
                kv_count: 0,
                kvs: Vec::new(),
                tensors: Vec::new(),
            }
        }

        fn push_str(out: &mut Vec<u8>, s: &str) {
            out.extend_from_slice(&(s.len() as u64).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }

        pub(crate) fn kv_str(mut self, key: &str, value: &str) -> Self {
            Self::push_str(&mut self.kvs, key);
            self.kvs.extend_from_slice(&8u32.to_le_bytes());
            Self::push_str(&mut self.kvs, value);
            self.kv_count += 1;
            self
        }

        pub(crate) fn kv_u32(mut self, key: &str, value: u32) -> Self {
            Self::push_str(&mut self.kvs, key);
            self.kvs.extend_from_slice(&4u32.to_le_bytes());
            self.kvs.extend_from_slice(&value.to_le_bytes());
            self.kv_count += 1;
            self
        }

        pub(crate) fn kv_str_array(mut self, key: &str, values: &[&str]) -> Self {
            Self::push_str(&mut self.kvs, key);
            self.kvs.extend_from_slice(&9u32.to_le_bytes());
            self.kvs.extend_from_slice(&8u32.to_le_bytes());
            self.kvs
                .extend_from_slice(&(values.len() as u64).to_le_bytes());
            for v in values {
                Self::push_str(&mut self.kvs, v);
            }
            self.kv_count += 1;
            self
        }

        pub(crate) fn kv_i32_array(mut self, key: &str, values: &[i32]) -> Self {
            Self::push_str(&mut self.kvs, key);
            self.kvs.extend_from_slice(&9u32.to_le_bytes());
            self.kvs.extend_from_slice(&5u32.to_le_bytes());
            self.kvs
                .extend_from_slice(&(values.len() as u64).to_le_bytes());
            for v in values {
                self.kvs.extend_from_slice(&v.to_le_bytes());
            }
            self.kv_count += 1;
            self
        }

        pub(crate) fn tensor(
            mut self,
            name: &str,
            dims: &[u64],
            type_id: u32,
            offset: u64,
        ) -> Self {
            Self::push_str(&mut self.tensors, name);
            self.tensors
                .extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                self.tensors.extend_from_slice(&d.to_le_bytes());
            }
            self.tensors.extend_from_slice(&type_id.to_le_bytes());
            self.tensors.extend_from_slice(&offset.to_le_bytes());
            self.tensor_count += 1;
            self
        }

        pub(crate) fn build(mut self) -> Vec<u8> {
            self.buf.extend_from_slice(&MAGIC);
            self.buf.extend_from_slice(&3u32.to_le_bytes());
            self.buf.extend_from_slice(&self.tensor_count.to_le_bytes());
            self.buf.extend_from_slice(&self.kv_count.to_le_bytes());
            self.buf.extend_from_slice(&self.kvs);
            self.buf.extend_from_slice(&self.tensors);
            self.buf
        }

        /// Header + alignment padding + tensor payload bytes (offsets in the
        /// tensor table are relative to the padded data start).
        pub(crate) fn build_with_payload(self, payload: &[u8]) -> Vec<u8> {
            let mut buf = self.build();
            let data_offset = (buf.len() as u64).div_ceil(DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT;
            buf.resize(usize::try_from(data_offset).expect("small test file"), 0);
            buf.extend_from_slice(payload);
            buf
        }
    }

    /// Write a split GGUF SET to a temp dir under `gguf-split`'s naming and
    /// run `f` on the first shard's path. `shards` is one (bytes) per shard,
    /// in shard order.
    fn with_split_set<R>(shards: &[Vec<u8>], f: impl FnOnce(&Path) -> R) -> R {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "mummu-gguf-split-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let n = shards.len();
        let mut first = None;
        for (i, bytes) in shards.iter().enumerate() {
            let path = dir.join(format!("m-{:05}-of-{n:05}.gguf", i + 1));
            let mut fh = std::fs::File::create(&path).expect("shard file");
            fh.write_all(bytes).expect("write shard");
            if i == 0 {
                first = Some(path);
            }
        }
        let out = f(first.as_deref().expect("at least one shard"));
        let _ = std::fs::remove_dir_all(&dir);
        out
    }

    /// Two shards, each carrying one f32 tensor, with `split.*` KV as
    /// `gguf-split` writes it.
    fn two_shard_set() -> Vec<Vec<u8>> {
        // 8 f32 = 32 bytes, one per shard, each at its OWN shard's offset 0 —
        // which is the property that makes a per-shard payload base necessary.
        let a: Vec<u8> = (0..8u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let b: Vec<u8> = (0..8u32)
            .flat_map(|i| (100.0 + i as f32).to_le_bytes())
            .collect();
        let s0 = TestGguf::new()
            .kv_str("general.architecture", "qwen4exp")
            .kv_u32("split.count", 2)
            .kv_u32("split.no", 0)
            .kv_u32("split.tensors.count", 2)
            .tensor("a.weight", &[8], 0, 0)
            .build_with_payload(&a);
        // Shard 2 carries a STUB kv set, exactly as the real tool writes it.
        let s1 = TestGguf::new()
            .kv_u32("split.count", 2)
            .kv_u32("split.no", 1)
            .tensor("b.weight", &[8], 0, 0)
            .build_with_payload(&b);
        vec![s0, s1]
    }

    #[test]
    fn open_sharded_merges_both_shards_into_one_namespace() {
        with_split_set(&two_shard_set(), |first| {
            let f = GgufFile::open_sharded(first).expect("split set opens");
            assert_eq!(f.shard_count(), 2, "both shards counted");
            assert_eq!(f.tensors.len(), 2, "tensor tables concatenated");
            assert!(f.tensor("a.weight").is_some(), "shard 1 tensor visible");
            assert!(f.tensor("b.weight").is_some(), "shard 2 tensor visible");
            // Metadata comes from the richest shard, not shard order.
            assert_eq!(f.architecture(), Some("qwen4exp"));
        });
    }

    #[test]
    fn a_tensor_reads_from_its_own_shards_payload_base() {
        // The regression this whole feature exists to prevent: both tensors
        // sit at offset 0 of their own shard, so resolving either against
        // shard 1's base returns the WRONG shard's bytes rather than failing.
        with_split_set(&two_shard_set(), |first| {
            let f = GgufFile::open_sharded(first).expect("split set opens");
            let a = f.read_tensor_f32("a.weight").expect("shard 1 payload");
            let b = f.read_tensor_f32("b.weight").expect("shard 2 payload");
            assert_eq!(a[0], 0.0, "shard 1 tensor reads shard 1 bytes");
            assert_eq!(b[0], 100.0, "shard 2 tensor reads SHARD 2 bytes");
            assert_eq!(a.len(), 8);
            assert_eq!(b.len(), 8);
        });
    }

    #[test]
    fn a_single_file_gguf_is_unchanged_by_the_sharded_path() {
        let bytes = TestGguf::new()
            .kv_str("general.architecture", "qwen2")
            .tensor("only.weight", &[8], 0, 0)
            .build_with_payload(
                &(0..8u32)
                    .flat_map(|i| (i as f32).to_le_bytes())
                    .collect::<Vec<u8>>(),
            );
        with_split_set(&[bytes], |first| {
            let f = GgufFile::open_sharded(first).expect("single file opens");
            // No split.count => no shard table at all, and shard_count() still
            // reports 1 so callers need no special case.
            assert!(f.shards.is_empty(), "single file records no shards");
            assert_eq!(f.shard_count(), 1);
            assert_eq!(f.read_tensor_f32("only.weight").expect("payload")[0], 0.0);
        });
    }

    #[test]
    fn a_missing_shard_is_refused_by_name_not_discovered_late() {
        let mut set = two_shard_set();
        set.truncate(1); // declares split.count = 2, only shard 1 written
        with_split_set(&set, |first| {
            // The set still names itself -00001-of-00001 because only one file
            // was written, so rewrite the path to what the KV claims.
            let dir = first.parent().expect("dir");
            let claimed = dir.join("m-00001-of-00002.gguf");
            std::fs::rename(first, &claimed).expect("rename to claimed name");
            let err = GgufFile::open_sharded(&claimed).expect_err("missing shard refused");
            let msg = format!("{err}");
            assert!(
                msg.contains("missing shard"),
                "names the absent file: {msg}"
            );
        });
    }

    #[test]
    fn a_duplicate_tensor_name_across_shards_is_refused() {
        // Two shards both claiming "a.weight": lookup order would silently
        // decide which payload wins, i.e. wrong weights with no error.
        let payload: Vec<u8> = (0..8u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let dup = TestGguf::new()
            .kv_u32("split.count", 2)
            .kv_u32("split.no", 1)
            .tensor("a.weight", &[8], 0, 0)
            .build_with_payload(&payload);
        let set = vec![two_shard_set().remove(0), dup];
        with_split_set(&set, |first| {
            let err = GgufFile::open_sharded(first).expect_err("duplicate refused");
            let msg = format!("{err}");
            assert!(
                msg.contains("more than one shard"),
                "explains the clash: {msg}"
            );
        });
    }

    #[test]
    fn a_tensor_count_that_disagrees_with_the_shards_is_refused() {
        let mut set = two_shard_set();
        // Claim 3 tensors while the shards carry 2.
        set[0] = TestGguf::new()
            .kv_str("general.architecture", "qwen4exp")
            .kv_u32("split.count", 2)
            .kv_u32("split.no", 0)
            .kv_u32("split.tensors.count", 3)
            .tensor("a.weight", &[8], 0, 0)
            .build_with_payload(
                &(0..8u32)
                    .flat_map(|i| (i as f32).to_le_bytes())
                    .collect::<Vec<u8>>(),
            );
        with_split_set(&set, |first| {
            let err = GgufFile::open_sharded(first).expect_err("count mismatch refused");
            assert!(
                format!("{err}").contains("declares 3"),
                "reports both counts"
            );
        });
    }

    /// Write `bytes` to a fresh temp file and run `f` on the parse result
    /// while the file still exists (payload reads re-open the path).
    pub(crate) fn with_gguf_bytes<R>(
        bytes: &[u8],
        f: impl FnOnce(Result<GgufFile, GgufError>) -> R,
    ) -> R {
        use std::sync::atomic::{AtomicU64, Ordering};
        // Parallel tests in one process must never share a temp file.
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join("mummu-gguf-tests");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join(format!(
            "t-{}-{}.gguf",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = File::create(&path).expect("temp file");
        file.write_all(bytes).expect("write");
        drop(file);
        let result = f(GgufFile::open(&path));
        let _ = std::fs::remove_file(&path);
        result
    }

    /// A unique scratch path for tests that write their own output file.
    fn scratch_path(tag: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join("mummu-gguf-tests");
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir.join(format!(
            "{tag}-{}-{}.safetensors",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn open_bytes(bytes: &[u8]) -> Result<GgufFile, GgufError> {
        with_gguf_bytes(bytes, |r| r)
    }

    #[test]
    fn minimal_file_round_trips() {
        let bytes = TestGguf::new()
            .kv_str("general.architecture", "qwen2")
            .kv_u32("qwen2.block_count", 28)
            .kv_str_array("tokenizer.ggml.tokens", &["a", "b", "c"])
            .tensor("token_embd.weight", &[64, 2], 0, 0)
            .tensor("blk.0.attn_q.weight", &[256], 12, 512)
            .build();
        let f = open_bytes(&bytes).expect("parses");
        assert_eq!(f.version, 3);
        assert_eq!(f.architecture(), Some("qwen2"));
        assert_eq!(
            f.get("qwen2.block_count").and_then(GgufValue::as_u64),
            Some(28)
        );
        assert_eq!(
            f.get("tokenizer.ggml.tokens")
                .and_then(GgufValue::as_array)
                .map(<[GgufValue]>::len),
            Some(3)
        );
        let embd = f.tensor("token_embd.weight").expect("present");
        assert_eq!(embd.dims, vec![64, 2]);
        assert_eq!(embd.dtype, GgmlType::F32);
        assert_eq!(embd.byte_len(), 64 * 2 * 4);
        let q = f.tensor("blk.0.attn_q.weight").expect("present");
        assert_eq!(q.dtype, GgmlType::Q4_K);
        assert_eq!(q.byte_len(), 144); // one 256-element Q4_K superblock
        assert_eq!(f.alignment, DEFAULT_ALIGNMENT);
        assert!(f.data_offset.is_multiple_of(f.alignment));
        assert!(f.data_offset >= (bytes.len() as u64));
    }

    #[test]
    fn bad_magic_is_rejected() {
        let mut bytes = TestGguf::new().build();
        bytes[..4].copy_from_slice(b"FUGG");
        assert!(matches!(
            open_bytes(&bytes),
            Err(GgufError::BadMagic { .. })
        ));
    }

    #[test]
    fn unsupported_version_is_rejected() {
        let mut bytes = TestGguf::new().build();
        bytes[4..8].copy_from_slice(&1u32.to_le_bytes());
        assert!(matches!(
            open_bytes(&bytes),
            Err(GgufError::UnsupportedVersion { version: 1, .. })
        ));
    }

    #[test]
    fn truncated_file_is_an_io_error_not_a_hang() {
        let bytes = TestGguf::new()
            .kv_str("general.architecture", "qwen2")
            .build();
        assert!(matches!(
            open_bytes(&bytes[..bytes.len() - 3]),
            Err(GgufError::Io { .. })
        ));
    }

    #[test]
    fn oversized_counts_are_rejected() {
        let mut bytes = TestGguf::new().build();
        // tensor_count lives at bytes 8..16.
        bytes[8..16].copy_from_slice(&(MAX_TENSORS + 1).to_le_bytes());
        assert!(matches!(
            open_bytes(&bytes),
            Err(GgufError::OverBound { what: "tensor", .. })
        ));
    }

    #[test]
    fn unknown_value_type_and_ggml_type_are_rejected() {
        let mut with_kv = TestGguf::new().kv_u32("some.key", 1).build();
        // The kv's type id (4 = u32) sits right after the 8-byte key string
        // prefix + 8 bytes of key: magic(4)+ver(4)+counts(16)+len(8)+key(8).
        with_kv[40..44].copy_from_slice(&99u32.to_le_bytes());
        assert!(matches!(
            open_bytes(&with_kv),
            Err(GgufError::BadValue { .. })
        ));

        let with_tensor = TestGguf::new().tensor("t", &[32], 63, 0).build();
        assert!(matches!(
            open_bytes(&with_tensor),
            Err(GgufError::BadTensor { .. })
        ));
    }

    // ---- dequant ---------------------------------------------------------

    fn f16_bytes(v: f32) -> [u8; 2] {
        half::f16::from_f32(v).to_bits().to_le_bytes()
    }

    #[test]
    fn float_widths_dequantize_exactly() {
        let f32_bytes = 1.5f32.to_le_bytes();
        assert_eq!(dequantize(GgmlType::F32, &f32_bytes).unwrap(), vec![1.5]);
        assert_eq!(
            dequantize(GgmlType::F16, &f16_bytes(-0.25)).unwrap(),
            vec![-0.25]
        );
        // bf16 is the top half of the f32 bit pattern.
        let bf16 = (2.0f32.to_bits() >> 16) as u16;
        assert_eq!(
            dequantize(GgmlType::BF16, &bf16.to_le_bytes()).unwrap(),
            vec![2.0]
        );
    }

    #[test]
    fn q8_0_block_matches_hand_computation() {
        let mut block = Vec::new();
        block.extend_from_slice(&f16_bytes(0.5));
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        block.extend((0..32).map(|i| (i - 16) as i8 as u8));
        let out = dequantize(GgmlType::Q8_0, &block).unwrap();
        assert_eq!(out.len(), 32);
        assert_eq!(out[0], 0.5 * -16.0);
        assert_eq!(out[16], 0.0);
        assert_eq!(out[31], 0.5 * 15.0);
    }

    #[test]
    fn q4_k_superblock_matches_hand_computation() {
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&f16_bytes(1.0)); // d
        block[2..4].copy_from_slice(&f16_bytes(0.5)); // dmin
        // Sub-block 0: sc=2, m=1 · sub-block 1: sc=3, m=0 (direct 6-bit slots).
        block[4] = 2;
        block[5] = 3;
        block[8] = 1;
        // Sub-block 4: packed slot — sc = scales[8] & 0xF = 1, m = scales[8] >> 4 = 2.
        block[12] = 0x21;
        // First quant byte of chunk 0: low nibble 1 (sub 0), high nibble 5 (sub 1).
        block[16] = 0x51;
        // First quant byte of chunk 2 (sub-blocks 4/5): low nibble 4.
        block[16 + 64] = 0x04;
        let out = dequantize(GgmlType::Q4_K, &block).unwrap();
        assert_eq!(out.len(), 256);
        assert_eq!(out[0], 2.0 * 1.0 - 0.5 * 1.0); // d·sc0·q − dmin·m0 = 1.5
        assert_eq!(out[32], 3.0 * 5.0); // sub 1: m=0
        assert_eq!(out[128], 1.0 * 4.0 - 0.5 * 2.0); // sub 4 via packed scales
        // A zero quant in sub-block 0 still subtracts the min.
        assert_eq!(out[1], -0.5);
    }

    #[test]
    fn q6_k_superblock_matches_hand_computation() {
        let mut block = vec![0u8; 210];
        block[0] = 0x0F; // ql[0]: low 4 bits = 15
        block[128] = 0b0000_0011; // qh[0]: high 2 bits = 3 for q1
        block[192] = 2; // scales[0] = 2
        block[194] = 1; // scales[2] = 1
        block[208..210].copy_from_slice(&f16_bytes(1.0)); // d
        let out = dequantize(GgmlType::Q6_K, &block).unwrap();
        assert_eq!(out.len(), 256);
        // q1 = (15 | 3<<4) − 32 = 31, scale 2 → 62.
        assert_eq!(out[0], 62.0);
        // q2 = (0 | 0) − 32 = −32, scale sc[2]=1 → −32.
        assert_eq!(out[32], -32.0);
        // Zero scale zeroes the value even though q3 = −32.
        assert_eq!(out[64], 0.0);
    }

    #[test]
    fn iq4_xs_superblock_matches_hand_computation() {
        let mut block = vec![0u8; 136];
        block[0..2].copy_from_slice(&f16_bytes(2.0)); // d
        // Sub-scale 0 = 33 (low 4 bits = 1, high 2 bits = 2 → 1|2<<4 = 33):
        // scales_l[0] low nibble = 1; scales_h bits 0..2 = 2.
        block[4] = 0x01;
        block[2..4].copy_from_slice(&2u16.to_le_bytes());
        // Sub-scale 1 = 0 → dl = 2·(0−32) = −64 for group 1.
        // qs[0]: low nibble index 8 (→ +1), high nibble index 15 (→ +113).
        block[8] = 0xF8; // low nibble = 8, high nibble = 15
        let out = dequantize(GgmlType::IQ4_XS, &block).unwrap();
        assert_eq!(out.len(), 256);
        // Group 0: dl = 2·(33−32) = 2. Element 0 = 2·kvalues[8] = 2·1.
        assert_eq!(out[0], 2.0);
        // High nibbles are elements 16..32 of the group: 2·kvalues[15].
        assert_eq!(out[16], 2.0 * 113.0);
        // Untouched qs bytes are index 0 → kvalues[0] = −127, scaled.
        assert_eq!(out[1], 2.0 * -127.0);
        // Group 1's zero sub-scale gives dl = −64: element 32 = −64·−127.
        assert_eq!(out[32], -64.0 * -127.0);
    }

    #[test]
    fn iq4_nl_block_matches_hand_computation() {
        let mut b = vec![0u8; 18];
        b[0..2].copy_from_slice(&f16_bytes(2.0));
        b[2] = 0xF8; // low nibble 8 → kvalues[8] = 1; high 15 → kvalues[15] = 113
        let out = dequantize(GgmlType::IQ4_NL, &b).unwrap();
        assert_eq!(out.len(), 32);
        assert_eq!(out[0], 2.0);
        assert_eq!(out[16], 2.0 * 113.0);
        assert_eq!(out[1], 2.0 * -127.0); // zero nibble hits kvalues[0]
    }

    #[test]
    fn iq2_xs_superblock_matches_hand_computation() {
        let mut b = vec![0u8; 74];
        b[0..2].copy_from_slice(&f16_bytes(1.0));
        // qs word 0: grid index 1 (0x…2b in byte 0), sign index 1 (KSIGNS[1]
        // = 129: bits 0 and 7 negative).
        b[2..4].copy_from_slice(&0x0201u16.to_le_bytes());
        b[66] = 0x21; // scales[0]: low nibble 1 → db0 = 0.375, high 2 → db1 = 0.625
        let out = dequantize(GgmlType::IQ2_XS, &b).unwrap();
        assert_eq!(out.len(), 256);
        assert_eq!(out[0], -0.375 * 43.0); // grid byte 0 = 0x2b, sign bit 0
        assert_eq!(out[1], 0.375 * 8.0);
        assert_eq!(out[7], -0.375 * 8.0); // sign bit 7
        // Row l = 2 (elements 16..24) rides db1; untouched words are
        // grid 0 (all 8s), sign 0.
        assert_eq!(out[16], 0.625 * 8.0);
    }

    #[test]
    fn iq2_s_superblock_matches_hand_computation() {
        let mut b = vec![0u8; 82];
        b[0..2].copy_from_slice(&f16_bytes(2.0));
        b[34] = 129; // signs byte for row 0: bits 0 and 7
        b[74] = 0x01; // scales[0] low nibble 1 → db0 = 2·1.5·0.25 = 0.75
        let out = dequantize(GgmlType::IQ2_S, &b).unwrap();
        assert_eq!(out.len(), 256);
        assert_eq!(out[0], -0.75 * 8.0); // grid 0 is all 8s
        assert_eq!(out[1], 0.75 * 8.0);
        assert_eq!(out[7], -0.75 * 8.0);
        assert_eq!(out[8], 0.75 * 8.0); // row 1 still rides db0
        assert_eq!(out[16], 0.5 * 0.25 * 2.0 * 8.0); // rows 2..4: zero high nibble → 0.5·d·0.25
    }

    #[test]
    fn iq3_xxs_superblock_matches_hand_computation() {
        let mut b = vec![0u8; 98];
        b[0..2].copy_from_slice(&f16_bytes(2.0));
        b[2] = 1; // grid1 index 1 = 0x04040414: byte 0 = 20, rest 4
        // aux32 for group 0: scale bits 1 (db = 2·1.5·0.5 = 1.5), sign index 1.
        b[66..70].copy_from_slice(&((1u32 << 28) | 1).to_le_bytes());
        let out = dequantize(GgmlType::IQ3_XXS, &b).unwrap();
        assert_eq!(out.len(), 256);
        assert_eq!(out[0], -1.5 * 20.0); // KSIGNS[1] bit 0
        assert_eq!(out[1], 1.5 * 4.0);
        assert_eq!(out[4], 1.5 * 4.0); // grid2 (index 0) is all 4s
        assert_eq!(out[7], -1.5 * 4.0); // KSIGNS[1] bit 7 lands on grid2's row
    }

    #[test]
    fn iq3_s_superblock_matches_hand_computation() {
        let mut b = vec![0u8; 110];
        b[0..2].copy_from_slice(&f16_bytes(1.0));
        b[2] = 1; // grid1 index 1 = 0x01010103: byte 0 = 3, rest 1
        b[74] = 1; // signs byte row 0: bit 0
        b[106] = 0x01; // scales[0] low nibble 1 → db = 1 + 2·1 = 3
        let out = dequantize(GgmlType::IQ3_S, &b).unwrap();
        assert_eq!(out.len(), 256);
        assert_eq!(out[0], -3.0 * 3.0);
        assert_eq!(out[1], 3.0);
        assert_eq!(out[4], 3.0); // grid2 (index 0) all 1s, no sign
        // Group 1 shares scales[0]'s high nibble (0) → db = 1.
        assert_eq!(out[32], 1.0);
    }

    #[test]
    fn q4_0_and_q4_1_blocks_match_hand_computation() {
        // Q4_0: symmetric around 8 — low nibbles are elements 0..16.
        let mut b = vec![0u8; 18];
        b[0..2].copy_from_slice(&f16_bytes(2.0));
        b[2] = 0x31; // low nibble 1 → elem 0, high nibble 3 → elem 16
        let out = dequantize(GgmlType::Q4_0, &b).unwrap();
        assert_eq!(out.len(), 32);
        assert_eq!(out[0], (1.0 - 8.0) * 2.0);
        assert_eq!(out[16], (3.0 - 8.0) * 2.0);
        assert_eq!(out[1], -16.0); // zero nibble is −8·d, not 0

        // Q4_1: affine — the min shifts every element.
        let mut b = vec![0u8; 20];
        b[0..2].copy_from_slice(&f16_bytes(2.0));
        b[2..4].copy_from_slice(&f16_bytes(1.0));
        b[4] = 0x31;
        let out = dequantize(GgmlType::Q4_1, &b).unwrap();
        assert_eq!(out[0], 1.0 * 2.0 + 1.0);
        assert_eq!(out[16], 3.0 * 2.0 + 1.0);
        assert_eq!(out[1], 1.0);
    }

    #[test]
    fn q5_0_and_q5_1_high_bits_land_on_the_right_elements() {
        // Q5_0: qh bit 0 tops element 0, bit 16 tops element 16.
        let mut b = vec![0u8; 22];
        b[0..2].copy_from_slice(&f16_bytes(1.0));
        b[2..6].copy_from_slice(&(1u32 | (1 << 16)).to_le_bytes());
        b[6] = 0x21; // low nibble 1 → elem 0, high nibble 2 → elem 16
        let out = dequantize(GgmlType::Q5_0, &b).unwrap();
        assert_eq!(out.len(), 32);
        assert_eq!(out[0], (1.0 + 16.0) - 16.0);
        assert_eq!(out[16], (2.0 + 16.0) - 16.0);
        assert_eq!(out[1], -16.0); // no high bit, zero nibble

        // Q5_1: same bit packing, affine.
        let mut b = vec![0u8; 24];
        b[0..2].copy_from_slice(&f16_bytes(1.0));
        b[2..4].copy_from_slice(&f16_bytes(1.0));
        b[4..8].copy_from_slice(&1u32.to_le_bytes());
        b[8] = 0x01;
        let out = dequantize(GgmlType::Q5_1, &b).unwrap();
        assert_eq!(out[0], (1.0 + 16.0) * 1.0 + 1.0);
        assert_eq!(out[16], 1.0); // high nibble 0, qh bit 16 unset → just m
    }

    #[test]
    fn q2_k_superblock_matches_hand_computation() {
        let mut b = vec![0u8; 84];
        b[80..82].copy_from_slice(&f16_bytes(1.0)); // d
        b[82..84].copy_from_slice(&f16_bytes(0.5)); // dmin
        b[0] = 0x12; // sub 0: sc=2, min=1
        b[1] = 0x01; // sub 1: sc=1, min=0
        b[8] = 0x0F; // sub 8 (second half, shift 0): sc=15, min=0
        b[16] = 3; // qs[0] bits 0–1 → elem 0
        b[32] = 1; // qs[16] → elem 16 (sub 1)
        b[48] = 2; // qs[32] → elem 128 (second half)
        let out = dequantize(GgmlType::Q2_K, &b).unwrap();
        assert_eq!(out.len(), 256);
        assert_eq!(out[0], 2.0 * 3.0 - 0.5); // d·sc·q − dmin·m
        assert_eq!(out[1], -0.5); // zero quant still subtracts the min
        assert_eq!(out[16], 1.0);
        assert_eq!(out[128], 15.0 * 2.0); // second half reads qs[32..]
    }

    #[test]
    fn q3_k_superblock_matches_hand_computation() {
        let mut b = vec![0u8; 110];
        b[108..110].copy_from_slice(&f16_bytes(1.0)); // d
        // scales: slot 0 = 2|32 = 34 → dl 2; slot 1 = 1|32 = 33 → dl 1;
        // slot 8 = (0x32>>4)|0 = 3 → dl 3−32 = −29 (tests the >>4 packing).
        b[96] = 0x32;
        b[97] = 0x01;
        b[104] = 0b10; // top bits of slot 0
        b[105] = 0b10; // top bits of slot 1
        b[0] = 1; // hmask[0] bit 0 → element 0 keeps its high bit (no −4)
        b[16] = 1; // hmask[16] bit 0 → element 16 too
        b[32] = 3; // qs[0] → elem 0
        b[48] = 2; // qs[16] → elem 16
        let out = dequantize(GgmlType::Q3_K, &b).unwrap();
        assert_eq!(out.len(), 256);
        assert_eq!(out[0], 2.0 * 3.0); // high bit set → q unshifted
        assert_eq!(out[1], 2.0 * -4.0); // high bit clear → q − 4
        assert_eq!(out[16], 1.0 * 2.0);
        // Second half, shift 0 (is=8): hmask[0] bit 4 clear → (0−4)·(3−32).
        assert_eq!(out[128], -4.0 * (3.0 - 32.0));
    }

    #[test]
    fn q5_k_superblock_matches_hand_computation() {
        let mut b = vec![0u8; 176];
        b[0..2].copy_from_slice(&f16_bytes(1.0)); // d
        b[2..4].copy_from_slice(&f16_bytes(1.0)); // dmin
        b[4] = 2; // sub 0 scale
        b[5] = 3; // sub 1 scale
        b[8] = 1; // sub 0 min
        b[16] = 1; // qh[0] bit 0 → elem 0 gets +16 (u1 = 1)
        b[48] = 0x21; // ql[0]: low 1 → elem 0, high 2 → elem 32
        let out = dequantize(GgmlType::Q5_K, &b).unwrap();
        assert_eq!(out.len(), 256);
        assert_eq!(out[0], 2.0 * (1.0 + 16.0) - 1.0);
        assert_eq!(out[1], -1.0); // zero quant still subtracts the min
        assert_eq!(out[32], 3.0 * 2.0); // qh bit 1 unset → no +16; m1 = 0
    }

    #[test]
    fn dequant_rejects_partial_blocks_and_unimplemented_types() {
        assert!(dequantize(GgmlType::Q8_0, &[0u8; 33]).is_err());
        assert!(dequantize(GgmlType::Q8_0, &[]).is_err());
        // Q8_K is an activation format, never tensor storage.
        assert!(dequantize(GgmlType::Q8_K, &[0u8; 292]).is_err());
    }

    /// The ternary family's ids, block widths and byte sizes, straight from
    /// the Prism fork's `ggml.h` / `ggml-common.h` (upstream carries 41/42).
    #[test]
    fn ternary_family_ids_and_layouts_match_the_fork() {
        for (id, ty, block, bytes) in [
            (41, GgmlType::Q1_0, 128, 18),
            (42, GgmlType::Q2_0, 64, 18),
            (142, GgmlType::PQ2_0, 128, 34),
            (143, GgmlType::PTQ1_0, 128, 28),
        ] {
            assert_eq!(GgmlType::from_id(id), Some(ty));
            assert_eq!(ty.block_size(), block, "{ty:?}");
            assert_eq!(ty.bytes_per_block(), bytes, "{ty:?}");
            assert!(ty.is_ternary_family());
        }
        assert!(!GgmlType::Q4_0.is_ternary_family());
        assert!(!GgmlType::IQ2_XS.is_ternary_family());
    }

    /// Q2_0/PQ2_0: element `j` is the 2-bit slot at bits `2·(j % 4)` of byte
    /// `j / 4`, decoding `(code − 1)·d` — the fork's `dequantize_row_q2_0`.
    #[test]
    fn q2_slots_decode_low_bits_first_as_code_minus_one() {
        let d = half::f16::from_f32(1.5).to_bits().to_le_bytes();
        // Byte 0b11_10_01_00 holds codes 0,1,2,3 in element order.
        let mut pq = vec![d[0], d[1]];
        pq.extend(std::iter::repeat_n(0b1110_0100u8, 32));
        let out = dequantize(GgmlType::PQ2_0, &pq).unwrap();
        assert_eq!(out.len(), 128);
        assert_eq!(&out[..4], &[-1.5, 0.0, 1.5, 3.0]);
        assert_eq!(&out[124..], &[-1.5, 0.0, 1.5, 3.0]);
        // Q2_0 is the same codec at 64 elements / 16 bytes.
        let mut q = vec![d[0], d[1]];
        q.extend(std::iter::repeat_n(0b0110_1001u8, 16)); // codes 1,2,2,1
        let out = dequantize(GgmlType::Q2_0, &q).unwrap();
        assert_eq!(out.len(), 64);
        assert_eq!(&out[..4], &[0.0, 1.5, 1.5, 0.0]);
        // A block of code-1 slots is all zeros whatever the scale says.
        let mut z = vec![d[0], d[1]];
        z.extend(std::iter::repeat_n(0b0101_0101u8, 32));
        assert!(
            dequantize(GgmlType::PQ2_0, &z)
                .unwrap()
                .iter()
                .all(|&v| v == 0.0)
        );
    }

    /// Q1_0: bit `j % 8` of byte `j / 8`, LSB first, `1 → +d`, `0 → −d`.
    #[test]
    fn q1_0_reads_bits_lsb_first() {
        let d = half::f16::from_f32(0.25).to_bits().to_le_bytes();
        let mut b = vec![d[0], d[1]];
        b.push(0b0000_0001); // element 0 set, 1..8 clear
        b.extend(std::iter::repeat_n(0xFFu8, 15));
        let out = dequantize(GgmlType::Q1_0, &b).unwrap();
        assert_eq!(out.len(), 128);
        assert_eq!(out[0], 0.25);
        assert!(out[1..8].iter().all(|&v| v == -0.25));
        assert!(out[8..].iter().all(|&v| v == 0.25));
    }

    /// The fork's `quantize_row_ptq1_0_ref`, ported verbatim: the encoder
    /// side of the trit packing, so the decoder is pinned by a round trip of
    /// every trit position rather than by one hand-built byte.
    fn encode_ptq1_0(x: &[f32]) -> Vec<u8> {
        assert_eq!(x.len(), 128);
        let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
        let id = if amax > 0.0 { 1.0 / amax } else { 0.0 };
        let mut qs = [0u8; 24];
        let mut qh = [0u8; 2];
        let mut x = x;
        let mut j = 0usize;
        for &c in &[32usize, 16, 8] {
            while j + c <= qs.len() {
                for m in 0..c {
                    let mut q: u8 = 0;
                    for n in 0..5 {
                        let xi = (x[m + n * c] * id).round() as i32 + 1;
                        q = q.wrapping_mul(3).wrapping_add(u8::try_from(xi).unwrap());
                    }
                    qs[j + m] = u8::try_from((u16::from(q) * 256).div_ceil(243)).unwrap();
                }
                x = &x[5 * c..];
                j += c;
            }
        }
        for h in 0..2usize {
            let mut q: u8 = 0;
            for m in 0..4 {
                let xi = (x[h + m * 2] * id).round() as i32 + 1;
                q = q.wrapping_mul(3).wrapping_add(u8::try_from(xi).unwrap());
            }
            q = q.wrapping_mul(3);
            qh[h] = u8::try_from((u16::from(q) * 256).div_ceil(243)).unwrap();
        }
        let mut out = qs.to_vec();
        out.extend_from_slice(&qh);
        out.extend_from_slice(&half::f16::from_f32(amax).to_bits().to_le_bytes());
        out
    }

    #[test]
    fn ptq1_0_round_trips_the_forks_reference_encoder() {
        // A deterministic ternary pattern that puts every trit value at
        // every position of every stage (16-byte, 8-byte, and the qh tail).
        let d = 0.75f32;
        let vals: Vec<f32> = (0..128)
            .map(|i: i32| d * ((i * 7 + i / 5) % 3 - 1) as f32)
            .collect();
        let block = encode_ptq1_0(&vals);
        assert_eq!(block.len(), 28);
        let out = dequantize(GgmlType::PTQ1_0, &block).unwrap();
        assert_eq!(
            out, vals,
            "every trit position must decode to what was encoded"
        );
        // Whole-row: two blocks decode independently, in order.
        let mut two = block.clone();
        let neg: Vec<f32> = vals.iter().map(|v| -v).collect();
        two.extend(encode_ptq1_0(&neg));
        let out = dequantize(GgmlType::PTQ1_0, &two).unwrap();
        assert_eq!(&out[..128], &vals[..]);
        assert_eq!(&out[128..], &neg[..]);
    }

    /// A table whose declared dtypes size the payloads past each other is
    /// refused: reading it would take one tensor's bytes for another's.
    #[test]
    fn overlapping_tensor_payloads_are_rejected() {
        // A 32-element Q8_0 tensor is 34 bytes; a second tensor at the next
        // 32-aligned offset lands inside it.
        let bytes = TestGguf::new()
            .kv_str("general.architecture", "qwen2")
            .tensor("a.weight", &[32], 8, 0)
            .tensor("b.weight", &[8], 0, 32)
            .build();
        with_gguf_bytes(&bytes, |f| {
            let err = f.expect_err("overlap must be refused").to_string();
            assert!(err.contains("overlaps"), "{err}");
            assert!(
                err.contains("b.weight") && err.contains("a.weight"),
                "{err}"
            );
        });
        // The same table with the second tensor past the first is fine.
        let bytes = TestGguf::new()
            .kv_str("general.architecture", "qwen2")
            .tensor("a.weight", &[32], 8, 0)
            .tensor("b.weight", &[8], 0, 64)
            .build();
        with_gguf_bytes(&bytes, |f| {
            f.expect("non-overlapping table parses");
        });
    }

    #[test]
    fn dequant_to_safetensors_reverses_dims_and_round_trips_bytes() {
        // The dequant pass publishes its own progress (it is the counted
        // bar four architectures load behind), so this shares the lock with
        // every other test that touches that process-wide state.
        let _serial = crate::progress::test_serial();
        // One F32 tensor with ggml dims [2, 3] and payload 1..=6.
        let payload: Vec<u8> = (1..=6).flat_map(|v| (v as f32).to_le_bytes()).collect();
        let bytes = TestGguf::new()
            .kv_str("general.architecture", "qwen2")
            .tensor("token_embd.weight", &[2, 3], 0, 0)
            .build_with_payload(&payload);
        with_gguf_bytes(&bytes, |f| {
            let f = f.expect("parses");
            let blob = f
                .dequant_to_safetensors(&|i| Some(GgufMap::Rename(format!("model.{}", i.name))))
                .expect("serializes");
            let header_len = u64::from_le_bytes(blob[0..8].try_into().unwrap());
            let json: serde_json::Value =
                serde_json::from_slice(&blob[8..8 + usize::try_from(header_len).unwrap()])
                    .expect("header is valid JSON");
            let entry = &json["model.token_embd.weight"];
            assert_eq!(entry["dtype"], "F32");
            assert_eq!(entry["shape"], serde_json::json!([3, 2])); // reversed
            assert_eq!(entry["data_offsets"], serde_json::json!([0, 24]));
            // F32 → f32 is byte-identical.
            assert_eq!(
                &blob[8 + usize::try_from(header_len).unwrap()..],
                &payload[..]
            );

            // An unmapped tensor is a loud error, never a silent skip.
            assert!(matches!(
                f.dequant_to_safetensors(&|_| None),
                Err(GgufError::BadTensor { .. })
            ));
        });
    }

    #[test]
    fn dequant_to_safetensors_rejects_rename_collisions() {
        // The dequant pass publishes its own progress (it is the counted
        // bar four architectures load behind), so this shares the lock with
        // every other test that touches that process-wide state.
        let _serial = crate::progress::test_serial();
        let payload = [0u8; 64]; // two 8-element F32 tensors, offsets 0 and 32
        let bytes = TestGguf::new()
            .tensor("a.weight", &[8], 0, 0)
            .tensor("b.weight", &[8], 0, 32)
            .build_with_payload(&payload);
        with_gguf_bytes(&bytes, |f| {
            let f = f.expect("parses");
            assert!(matches!(
                f.dequant_to_safetensors(&|_| Some(GgufMap::Rename("same".into()))),
                Err(GgufError::BadTensor { .. })
            ));
        });
    }

    #[test]
    fn dequanting_to_a_file_is_byte_identical_to_dequanting_in_memory() {
        // The dequant pass publishes its own progress (it is the counted
        // bar four architectures load behind), so this shares the lock with
        // every other test that touches that process-wide state.
        let _serial = crate::progress::test_serial();
        // Mixed dtypes and widths, so the pin covers the quantized path and
        // the tensor-to-tensor offset arithmetic, not just one F32 copy.
        // Q8_0 blocks are 34 B for 32 elements: two blocks = 68 B at offset 0,
        // then a 6-element F32 tensor at the next 32-aligned offset.
        let mut payload = vec![0u8; 96];
        for (i, b) in payload.iter_mut().enumerate().take(68) {
            *b = u8::try_from(i % 251).expect("bounded by the modulus");
        }
        payload.extend((1..=6u8).flat_map(|v| f32::from(v).to_le_bytes()));
        let bytes = TestGguf::new()
            .kv_str("general.architecture", "qwen2")
            .tensor("blk.0.attn_q.weight", &[32, 2], 8, 0) // Q8_0
            .tensor("token_embd.weight", &[2, 3], 0, 96) // F32
            .build_with_payload(&payload);

        with_gguf_bytes(&bytes, |f| {
            let f = f.expect("parses");
            let map = |i: &GgufTensorInfo| Some(GgufMap::Rename(format!("model.{}", i.name)));
            let blob = f.dequant_to_safetensors(&map).expect("serializes");

            let out = scratch_path("dequant-pin");
            let written = f
                .dequant_to_safetensors_file(&map, &out)
                .expect("streams to a file");
            let from_file = std::fs::read(&out).expect("reads back");
            let _ = std::fs::remove_file(&out);

            assert_eq!(
                blob, from_file,
                "the in-memory and to-file dequants must stay ONE importer"
            );
            // The returned count is the payload, header excluded — so the
            // caller can size the model without re-statting the file.
            let header_len = u64::from_le_bytes(blob[0..8].try_into().expect("8 bytes"));
            assert_eq!(written, blob.len() as u64 - 8 - header_len);
            // 64 Q8_0 elements + 6 F32 elements, all at f32.
            assert_eq!(written, (64 + 6) * 4);
        });
    }

    #[test]
    fn a_bad_map_is_rejected_before_any_payload_is_read() {
        // The dequant pass publishes its own progress (it is the counted
        // bar four architectures load behind), so this shares the lock with
        // every other test that touches that process-wide state.
        let _serial = crate::progress::test_serial();
        // The tensor table claims a payload far past the end of the file, so
        // ANY read of it is an io error. A map error must still surface as the
        // map error: planning happens first, by construction.
        let bytes = TestGguf::new()
            .tensor("a.weight", &[8], 0, 0)
            .tensor("b.weight", &[1 << 20], 0, 32)
            .build_with_payload(&[0u8; 64]);
        with_gguf_bytes(&bytes, |f| {
            let f = f.expect("parses");
            assert!(matches!(
                f.dequant_to_safetensors(
                    &|i| (i.name != "b.weight").then(|| GgufMap::Rename(i.name.clone()))
                ),
                Err(GgufError::BadTensor { index: 1, .. })
            ));
            // And the collision check too — it is the second claim planning
            // makes, on a tensor whose payload is unreadable.
            assert!(matches!(
                f.dequant_to_safetensors(&|_| Some(GgufMap::Rename("same".into()))),
                Err(GgufError::BadTensor { index: 1, .. })
            ));
        });
    }

    #[test]
    fn misaligned_offsets_partial_blocks_and_duplicates_are_rejected() {
        let misaligned = TestGguf::new().tensor("t", &[32], 8, 7).build();
        assert!(matches!(
            open_bytes(&misaligned),
            Err(GgufError::BadTensor { .. })
        ));
        // 100 elements is not whole 256-element Q4_K superblocks.
        let partial = TestGguf::new().tensor("t", &[100], 12, 0).build();
        assert!(matches!(
            open_bytes(&partial),
            Err(GgufError::BadTensor { .. })
        ));
        let dup = TestGguf::new()
            .tensor("t", &[32], 8, 0)
            .tensor("t", &[32], 8, 64)
            .build();
        assert!(matches!(open_bytes(&dup), Err(GgufError::BadTensor { .. })));
    }
}
