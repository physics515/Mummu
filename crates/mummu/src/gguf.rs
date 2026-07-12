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
use std::io::{BufReader, Read, Seek};
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

    /// The value as f32, if it is one.
    #[must_use]
    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Self::F32(v) => Some(v),
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
    BF16,
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
            30 => Self::BF16,
            _ => return None,
        };
        Some(ty)
    }

    /// Elements per quantization block (1 for plain float types).
    #[must_use]
    pub fn block_size(self) -> u64 {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K => 256,
        }
    }

    /// Bytes one block occupies on disk (ggml's type sizes).
    #[must_use]
    pub fn bytes_per_block(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18,  // f16 d + 16 B qs
            Self::Q4_1 => 20,  // f16 d + f16 m + 16 B qs
            Self::Q5_0 => 22,  // f16 d + 4 B qh + 16 B qs
            Self::Q5_1 => 24,  // f16 d + f16 m + 4 B qh + 16 B qs
            Self::Q8_0 => 34,  // f16 d + 32 i8
            Self::Q8_1 => 36,  // f16 d + f16 s + 32 i8
            Self::Q2_K => 84,  // 16 B scales + 64 B qs + f16 d + f16 dmin
            Self::Q3_K => 110, // 32 B hmask + 64 B qs + 12 B scales + f16 d
            Self::Q4_K => 144, // f16 d + f16 dmin + 12 B scales + 128 B qs
            Self::Q5_K => 176, // Q4_K + 32 B qh
            Self::Q6_K => 210, // 128 B ql + 64 B qh + 16 i8 scales + f16 d
            Self::Q8_K => 292, // f32 d + 256 i8 + 16 i16 bsums
        }
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

/// A parsed GGUF header: typed metadata + the located tensor table.
#[derive(Debug)]
pub struct GgufFile {
    pub version: u32,
    /// Metadata in file order (keys are unique per the spec).
    pub metadata: Vec<(String, GgufValue)>,
    pub tensors: Vec<GgufTensorInfo>,
    /// Payload alignment (`general.alignment`, default 32).
    pub alignment: u64,
    /// Absolute file offset where the aligned tensor payload blob begins.
    pub data_offset: u64,
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
        let parsed = r.read_file()?;
        // Positive space: the header must end before the payload it locates.
        assert!(
            parsed.data_offset.is_multiple_of(parsed.alignment),
            "data offset is aligned by construction"
        );
        Ok(parsed)
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
        let mut buf = vec![0u8; usize::try_from(len).expect("bounded above")];
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
                let mut items = Vec::with_capacity(usize::try_from(len).expect("bounded above"));
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
    fn read_file(&mut self) -> Result<GgufFile, GgufError> {
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

        let mut metadata = Vec::with_capacity(usize::try_from(kv_count).expect("bounded above"));
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
            version,
            metadata,
            tensors,
            alignment,
            data_offset,
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
        let mut tensors: Vec<GgufTensorInfo> =
            Vec::with_capacity(usize::try_from(count).expect("bounded above"));
        for index in 0..usize::try_from(count).expect("bounded above") {
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
        Ok(tensors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Minimal in-memory GGUF builder for tests.
    struct TestGguf {
        buf: Vec<u8>,
        tensor_count: u64,
        kv_count: u64,
        kvs: Vec<u8>,
        tensors: Vec<u8>,
    }

    impl TestGguf {
        fn new() -> Self {
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

        fn kv_str(mut self, key: &str, value: &str) -> Self {
            Self::push_str(&mut self.kvs, key);
            self.kvs.extend_from_slice(&8u32.to_le_bytes());
            Self::push_str(&mut self.kvs, value);
            self.kv_count += 1;
            self
        }

        fn kv_u32(mut self, key: &str, value: u32) -> Self {
            Self::push_str(&mut self.kvs, key);
            self.kvs.extend_from_slice(&4u32.to_le_bytes());
            self.kvs.extend_from_slice(&value.to_le_bytes());
            self.kv_count += 1;
            self
        }

        fn kv_str_array(mut self, key: &str, values: &[&str]) -> Self {
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

        fn tensor(mut self, name: &str, dims: &[u64], type_id: u32, offset: u64) -> Self {
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

        fn build(mut self) -> Vec<u8> {
            self.buf.extend_from_slice(&MAGIC);
            self.buf.extend_from_slice(&3u32.to_le_bytes());
            self.buf.extend_from_slice(&self.tensor_count.to_le_bytes());
            self.buf.extend_from_slice(&self.kv_count.to_le_bytes());
            self.buf.extend_from_slice(&self.kvs);
            self.buf.extend_from_slice(&self.tensors);
            self.buf
        }
    }

    fn open_bytes(bytes: &[u8]) -> Result<GgufFile, GgufError> {
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
        let mut f = File::create(&path).expect("temp file");
        f.write_all(bytes).expect("write");
        drop(f);
        let result = GgufFile::open(&path);
        let _ = std::fs::remove_file(&path);
        result
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
