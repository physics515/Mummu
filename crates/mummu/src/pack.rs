//! The `.mummu` pack — mummu's own multi-precision model artifact (P9
//! stage 3). Import converts a source checkpoint **once** into a directory
//! holding every tensor at every stored precision, so any device can pull
//! exactly the tensors and the precision it runs best — per MoE **expert** —
//! without re-importing, and the planner can re-tier at will.
//!
//! Layout of `<model>.mummu/`:
//!
//! - `manifest.json` — version, source, per-tensor entries: pack name,
//!   role, stored shape, and per-precision `{values, scales}` byte ranges.
//! - `header.gguf` — the source GGUF's header bytes (metadata + tensor table,
//!   no payload): `GgufFile::open` parses it, so every config / tokenizer
//!   reader the GGUF path already has works on a pack unchanged.
//! - `q4.bin`, `q8.bin`, `f16.bin`, `f32.bin` — one blob per stored
//!   precision; each tensor's bytes are contiguous within it.
//!
//! Stored precisions stop at f32: weights born bf16/Q4 carry no more
//! information, so f64 is a *compute* option derived from f32 where a
//! backend supports it, not a storage level. Quantized levels use block-32
//! symmetric scales (burn's `Q8S`/`Q4S` semantics) in **burn's canonical
//! quantized `TensorData` layout** (`TensorData::quantized`), which every
//! backend ingests via `q_from_data` — loading is a copy, never a
//! re-quantization. Linear weights are stored already transposed to burn's
//! `[in, out]`; expert banks are split into per-expert members on import.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use burn::tensor::quantization::QuantScheme;
use burn::tensor::{Device, Tensor, TensorData};

use crate::gguf::{GgufFile, GgufTensorInfo};
use crate::quant::QuantPolicy;
// `scheme()` is an extension trait: the ladder lives in `mummu-mix`, which
// has no burn dependency, so the burn binding is bolted on here.
use crate::quant::SchemeExt;

/// Pack format version (bump on any incompatible manifest/blob change).
pub const PACK_VERSION: u32 = 1;
/// Block width of the quantized levels (must match `QuantPolicy::scheme`).
pub const BLOCK: usize = 32;

/// A stored precision level.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    Q4,
    Q8,
    F16,
    F32,
}

impl Precision {
    pub const ALL: [Self; 4] = [Self::Q4, Self::Q8, Self::F16, Self::F32];

    /// Blob file name inside the pack.
    #[must_use]
    pub fn blob_name(self) -> &'static str {
        match self {
            Self::Q4 => "q4.bin",
            Self::Q8 => "q8.bin",
            Self::F16 => "f16.bin",
            Self::F32 => "f32.bin",
        }
    }

    /// The quant policy this level corresponds to (`Off` for floats).
    #[must_use]
    pub fn policy(self) -> QuantPolicy {
        match self {
            Self::Q4 => QuantPolicy::Q4,
            Self::Q8 => QuantPolicy::Q8,
            Self::F16 | Self::F32 => QuantPolicy::Off,
        }
    }

    /// Parse `q4,q8,f16,f32` lists (the `MUMMU_PACK_PRECISIONS` convention).
    pub fn parse_list(s: &str) -> Result<Vec<Self>, String> {
        let mut out = Vec::new();
        for item in s.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            out.push(match item.to_ascii_lowercase().as_str() {
                "q4" | "int4" => Self::Q4,
                "q8" | "int8" => Self::Q8,
                "f16" | "half" => Self::F16,
                "f32" | "float" => Self::F32,
                other => return Err(format!("unknown precision {other:?}")),
            });
        }
        if out.is_empty() {
            return Err("no precisions given".into());
        }
        out.sort();
        out.dedup();
        Ok(out)
    }
}

/// What a tensor is to the model — what the loaders need to place it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Role {
    /// A 2-D projection weight, stored `[in, out]` (burn Linear layout).
    Linear,
    /// The token embedding, stored `[vocab, hidden]`; never quantized.
    Embedding,
    /// A 1-D vector (norm gamma, bias, per-head scalars).
    Vector,
    /// A depthwise conv kernel `[channels, 1, k]`.
    Conv,
    /// One member of a split MoE expert bank, stored `[in, out]`.
    Expert {
        layer: usize,
        index: usize,
        proj: String,
    },
}

/// Where one precision of one tensor lives inside its blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Blob {
    pub values_offset: u64,
    pub values_len: u64,
    /// Zero-length for float levels.
    pub scales_offset: u64,
    pub scales_len: u64,
}

/// One tensor in the pack.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TensorEntry {
    /// The pack name — the source tensor name, with `/e{index}` appended for
    /// split expert members. Loaders map it to their module paths.
    pub name: String,
    pub role: Role,
    /// Row-major shape AS STORED (linears already `[in, out]`).
    pub shape: Vec<usize>,
    pub precisions: BTreeMap<Precision, Blob>,
}

/// The pack manifest.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Manifest {
    pub version: u32,
    pub source_file: String,
    pub source_bytes: u64,
    pub architecture: String,
    pub precisions: Vec<Precision>,
    pub tensors: Vec<TensorEntry>,
    /// P9 stage 3(c): the dense FFNs partitioned into neuron clusters (see
    /// `crate::partition`). Absent on packs imported before partitioning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ffn_partition: Option<FfnPartition>,
}

/// One contiguous cluster of a (permuted) FFN intermediate dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ClusterSpan {
    pub start: usize,
    pub len: usize,
}

/// One measured point of the skip trade-off: at energy threshold `tau`,
/// how far the skipped model strays from the exact one.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SkipPoint {
    pub tau: f32,
    /// Max |Δ log-prob| over the vocabulary at the measured positions.
    pub max_delta_logprob: f32,
    /// Fraction of measured positions whose argmax was unchanged.
    pub argmax_agreement: f32,
    /// Mean fraction of clusters actually computed.
    pub kept_fraction: f32,
}

/// The FFN partition of a dense model: per layer, the cluster spans of the
/// permuted intermediate dim and the three entry names; plus, once
/// calibrated, a hotness prior per cluster and the skip table.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct FfnPartition {
    pub layers: Vec<Vec<ClusterSpan>>,
    /// `[gate, up, down]` pack names per layer.
    pub names: Vec<[String; 3]>,
    /// Per-layer, per-cluster activation energy share from calibration
    /// (empty until `pack-calibrate` ran).
    #[serde(default)]
    pub hotness: Vec<Vec<f32>>,
    /// Measured skip trade-off (empty until calibrated) — the planner may
    /// only pick a `tau` that appears here.
    #[serde(default)]
    pub skip_table: Vec<SkipPoint>,
}

/// How the importer should treat one source tensor.
#[derive(Debug, Clone)]
pub enum ImportAction {
    /// Drop it (e.g. NextN/MTP blocks).
    Skip,
    /// A 2-D linear weight: transpose GGUF's `[out, in]` to `[in, out]`.
    Linear,
    /// The embedding table: keep `[vocab, hidden]`, float only.
    Embedding,
    /// A 1-D vector: float only.
    Vector,
    /// A squeezed depthwise conv kernel `[k, channels]` → `[channels, 1, k]`.
    Conv,
    /// A fused expert bank `[experts, out, in]`: split into members, each
    /// stored `[in, out]` as `Role::Expert`.
    ExpertBank { layer: usize, proj: String },
}

// ---------------------------------------------------------------------------
// Quantizer: block-32 symmetric, the same semantics as burn's Q8S/Q4S
// min-max calibration (scale = max|block| / range_max). These values + scales
// ARE the model at that level; burn reconstructs tensors from them verbatim.
// ---------------------------------------------------------------------------

/// Quantize a row-major tensor whose LAST dim is a multiple of [`BLOCK`]:
/// returns (i8 values, f32 scale per block). Blocks run along rows — a
/// block never straddles two rows.
pub fn quantize_blocks(
    values: &[f32],
    last_dim: usize,
    precision: Precision,
) -> (Vec<i8>, Vec<f32>) {
    assert!(
        last_dim.is_multiple_of(BLOCK) && values.len().is_multiple_of(last_dim),
        "quantize_blocks: last dim {last_dim} must divide by {BLOCK} and the length"
    );
    let range_max: f32 = match precision {
        Precision::Q8 => 127.0,
        Precision::Q4 => 7.0,
        _ => panic!("quantize_blocks: {precision:?} is not a quantized level"),
    };
    let mut q = Vec::with_capacity(values.len());
    let mut scales = Vec::with_capacity(values.len() / BLOCK);
    for block in values.as_chunks::<BLOCK>().0 {
        let alpha = block.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let scale = if alpha > 0.0 { alpha / range_max } else { 1.0 };
        scales.push(scale);
        let inv = 1.0 / scale;
        q.extend(
            block
                .iter()
                .map(|&x| (x * inv).round().clamp(-range_max, range_max) as i8),
        );
    }
    (q, scales)
}

/// Pack i8 values holding 4-bit range into nibbles (two per byte, low
/// nibble first) for the on-disk Q4 blob.
fn pack_nibbles(values: &[i8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len().div_ceil(2));
    for pair in values.chunks(2) {
        let lo = (pair[0] as u8) & 0x0F;
        let hi = pair.get(1).map_or(0, |&v| (v as u8) & 0x0F);
        out.push(lo | (hi << 4));
    }
    out
}

/// Inverse of [`pack_nibbles`]: sign-extend each nibble back to i8.
fn unpack_nibbles(bytes: &[u8], n: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(n);
    for &b in bytes {
        for nib in [b & 0x0F, b >> 4] {
            if out.len() == n {
                break;
            }
            // 4-bit two's complement sign extension.
            out.push(if nib & 0x8 != 0 {
                (nib | 0xF0) as i8
            } else {
                nib as i8
            });
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Writer / importer
// ---------------------------------------------------------------------------

struct BlobWriter {
    file: std::io::BufWriter<std::fs::File>,
    len: u64,
    /// Bytes written since the last `sync_data` — a multi-hundred-GB import
    /// must not leave that much dirty page cache behind (a Docker VM's OOM
    /// killer counts it against the process long before writeback catches
    /// up on a bind mount), so blobs are synced every [`SYNC_EVERY`] bytes.
    unsynced: u64,
}

/// Dirty-bytes bound per blob between `sync_data` calls.
const SYNC_EVERY: u64 = 1 << 30;

impl BlobWriter {
    fn append(&mut self, bytes: &[u8]) -> std::io::Result<(u64, u64)> {
        let off = self.len;
        self.file.write_all(bytes)?;
        self.len += bytes.len() as u64;
        self.unsynced += bytes.len() as u64;
        if self.unsynced >= SYNC_EVERY {
            self.file.flush()?;
            self.file.get_ref().sync_data()?;
            self.unsynced = 0;
        }
        Ok((off, bytes.len() as u64))
    }
}

/// Import a GGUF into a pack at `out_dir` with the given stored precisions.
/// `map` classifies every source tensor; the importer reads each tensor
/// once (dequantizing whatever the source stored), lays it out for the
/// loaders, and writes every requested level. Float-only roles (embedding,
/// vectors, convs) get only the float levels requested (f16/f32; if neither
/// was requested, f32 is added for them). Quantized levels apply only where
/// `QuantPolicy::eligible` would.
pub fn import_gguf(
    gguf_path: &Path,
    out_dir: &Path,
    precisions: &[Precision],
    map: &dyn Fn(&GgufTensorInfo) -> Option<ImportAction>,
    mut on_progress: impl FnMut(usize, usize, &str),
) -> Result<Manifest, String> {
    let f = GgufFile::open(gguf_path).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(out_dir).map_err(|e| format!("create {}: {e}", out_dir.display()))?;

    // Header copy: the first `data_offset` bytes are exactly metadata +
    // tensor table — a valid payload-less GGUF for every header reader.
    {
        let mut src = std::fs::File::open(gguf_path).map_err(|e| e.to_string())?;
        let mut header = vec![0u8; usize::try_from(f.data_offset).expect("header fits")];
        src.read_exact(&mut header)
            .map_err(|e| format!("read header: {e}"))?;
        std::fs::write(out_dir.join("header.gguf"), &header).map_err(|e| e.to_string())?;
    }

    let mut precisions: Vec<Precision> = precisions.to_vec();
    precisions.sort();
    precisions.dedup();
    let float_levels: Vec<Precision> = {
        let mut v: Vec<Precision> = precisions
            .iter()
            .copied()
            .filter(|p| matches!(p, Precision::F16 | Precision::F32))
            .collect();
        if v.is_empty() {
            v.push(Precision::F32);
        }
        v
    };
    let mut all_levels = precisions.clone();
    for p in &float_levels {
        if !all_levels.contains(p) {
            all_levels.push(*p);
        }
    }
    all_levels.sort();

    let mut writers: BTreeMap<Precision, BlobWriter> = BTreeMap::new();
    for &p in &all_levels {
        let file = std::fs::File::create(out_dir.join(p.blob_name())).map_err(|e| e.to_string())?;
        writers.insert(
            p,
            BlobWriter {
                file: std::io::BufWriter::with_capacity(8 << 20, file),
                len: 0,
                unsynced: 0,
            },
        );
    }

    let mut entries: Vec<TensorEntry> = Vec::new();
    let total = f.tensors.len();
    for (i, info) in f.tensors.iter().enumerate() {
        on_progress(i, total, &info.name);
        let action = map(info).ok_or_else(|| format!("unmapped tensor name '{}'", info.name))?;
        if matches!(action, ImportAction::Skip) {
            continue;
        }
        let dims_rev: Vec<usize> = info.dims.iter().rev().map(|&d| d as usize).collect();
        let values = f.read_tensor_f32(&info.name).map_err(|e| e.to_string())?;

        // Materialize the stored layout(s): one or many (expert bank) tensors.
        let mut items: Vec<(String, Role, Vec<usize>, Vec<f32>)> = Vec::new();
        match action {
            ImportAction::Skip => unreachable!(),
            ImportAction::Linear => {
                let &[out, inp] = dims_rev.as_slice() else {
                    return Err(format!(
                        "'{}' linear must be 2-D, got {dims_rev:?}",
                        info.name
                    ));
                };
                items.push((
                    info.name.clone(),
                    Role::Linear,
                    vec![inp, out],
                    transpose(&values, out, inp),
                ));
            }
            ImportAction::Embedding => {
                items.push((info.name.clone(), Role::Embedding, dims_rev.clone(), values));
            }
            ImportAction::Vector => {
                items.push((info.name.clone(), Role::Vector, dims_rev.clone(), values));
            }
            ImportAction::Conv => {
                // ggml ne = [k, ch] ⇒ row-major [ch, k] == checkpoint [ch, 1, k] bytes.
                let &[ch, k] = dims_rev.as_slice() else {
                    return Err(format!(
                        "'{}' conv must be 2-D squeezed, got {dims_rev:?}",
                        info.name
                    ));
                };
                items.push((info.name.clone(), Role::Conv, vec![ch, 1, k], values));
            }
            ImportAction::ExpertBank { layer, proj } => {
                let &[e, out, inp] = dims_rev.as_slice() else {
                    return Err(format!(
                        "'{}' expert bank must be 3-D, got {dims_rev:?}",
                        info.name
                    ));
                };
                let stride = out * inp;
                for expert in 0..e {
                    let member = &values[expert * stride..(expert + 1) * stride];
                    items.push((
                        format!("{}/e{expert}", info.name),
                        Role::Expert {
                            layer,
                            index: expert,
                            proj: proj.clone(),
                        },
                        vec![inp, out],
                        transpose(member, out, inp),
                    ));
                }
            }
        }

        for (name, role, shape, data) in items {
            let quantizable = matches!(role, Role::Linear | Role::Expert { .. })
                && QuantPolicy::Q8.eligible(&shape);
            let mut per: BTreeMap<Precision, Blob> = BTreeMap::new();
            for &p in &all_levels {
                let stored = match p {
                    Precision::F32 | Precision::F16 => {
                        float_levels.contains(&p) || !quantizable && p == Precision::F32
                    }
                    Precision::Q4 | Precision::Q8 => quantizable && precisions.contains(&p),
                };
                if !stored {
                    continue;
                }
                let w = writers.get_mut(&p).expect("writer exists");
                let blob = match p {
                    Precision::F32 => {
                        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
                        let (o, l) = w.append(&bytes).map_err(|e| e.to_string())?;
                        Blob {
                            values_offset: o,
                            values_len: l,
                            scales_offset: 0,
                            scales_len: 0,
                        }
                    }
                    Precision::F16 => {
                        let bytes: Vec<u8> = data
                            .iter()
                            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
                            .collect();
                        let (o, l) = w.append(&bytes).map_err(|e| e.to_string())?;
                        Blob {
                            values_offset: o,
                            values_len: l,
                            scales_offset: 0,
                            scales_len: 0,
                        }
                    }
                    Precision::Q8 | Precision::Q4 => {
                        let last = *shape.last().expect("non-empty shape");
                        let (q, scales) = quantize_blocks(&data, last, p);
                        let vbytes: Vec<u8> = if p == Precision::Q4 {
                            pack_nibbles(&q)
                        } else {
                            q.iter().map(|&v| v as u8).collect()
                        };
                        let (vo, vl) = w.append(&vbytes).map_err(|e| e.to_string())?;
                        let sbytes: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
                        let (so, sl) = w.append(&sbytes).map_err(|e| e.to_string())?;
                        Blob {
                            values_offset: vo,
                            values_len: vl,
                            scales_offset: so,
                            scales_len: sl,
                        }
                    }
                };
                per.insert(p, blob);
            }
            entries.push(TensorEntry {
                name,
                role,
                shape,
                precisions: per,
            });
        }
    }
    for w in writers.values_mut() {
        w.file.flush().map_err(|e| e.to_string())?;
        w.file.get_ref().sync_all().map_err(|e| e.to_string())?;
    }

    let manifest = Manifest {
        version: PACK_VERSION,
        source_file: gguf_path
            .file_name()
            .map_or_else(String::new, |n| n.to_string_lossy().into_owned()),
        source_bytes: std::fs::metadata(gguf_path).map(|m| m.len()).unwrap_or(0),
        architecture: f.architecture().unwrap_or("").to_string(),
        precisions: all_levels,
        tensors: entries,
        ffn_partition: None,
    };
    let json = serde_json::to_string_pretty(&manifest).map_err(|e| e.to_string())?;
    std::fs::write(out_dir.join("manifest.json"), json).map_err(|e| e.to_string())?;
    Ok(manifest)
}

/// Row-major `[out, in]` → `[in, out]`.
fn transpose(values: &[f32], out: usize, inp: usize) -> Vec<f32> {
    debug_assert_eq!(values.len(), out * inp);
    let mut t = vec![0.0f32; values.len()];
    for o in 0..out {
        for i in 0..inp {
            t[i * out + o] = values[o * inp + i];
        }
    }
    t
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// An opened pack: manifest + lazily-read blobs.
/// Burn's canonical quantized `TensorData` bytes for block-quantized i8/i4
/// values: the values packed little-endian into u32 words (8 nibbles or 4
/// bytes per word, element `j` at bits `j·bits`), then the f32 block scales
/// appended — exactly what `Tensor::quantize(..).into_data()` yields and
/// what every backend's `q_from_data` consumes. (`TensorData::quantized`
/// itself only handles 8-bit values under the default `PackedU32` store:
/// its Q4 reader unpacks nibbles the constructor never packed.)
pub fn quantized_tensor_data(
    values: &[i8],
    scales: &[f32],
    shape: impl Into<burn::tensor::Shape>,
    scheme: QuantScheme,
) -> TensorData {
    use burn::tensor::quantization::QuantValue;
    let shape: burn::tensor::Shape = shape.into();
    let bits = match scheme.value {
        QuantValue::Q8S | QuantValue::Q8F => 8,
        QuantValue::Q4S | QuantValue::Q4F => 4,
        other => panic!("quantized_tensor_data: unsupported value type {other:?}"),
    };
    let per_word = 32 / bits;
    let mask = (1u32 << bits) - 1;
    let mut words: Vec<u32> = Vec::with_capacity(values.len().div_ceil(per_word) + scales.len());
    for chunk in values.chunks(per_word) {
        let mut w = 0u32;
        for (j, &v) in chunk.iter().enumerate() {
            w |= ((v as u8 as u32) & mask) << (j * bits);
        }
        words.push(w);
    }
    words.extend(scales.iter().map(|s| s.to_bits()));
    TensorData::from_bytes(
        burn::tensor::Bytes::from_elems(words),
        shape,
        burn::tensor::DType::QFloat(scheme),
    )
}

pub struct Pack {
    pub dir: PathBuf,
    pub manifest: Manifest,
}

impl Pack {
    /// Is `dir` a pack (has a readable manifest)?
    #[must_use]
    pub fn is_pack(dir: &Path) -> bool {
        dir.join("manifest.json").is_file() && dir.join("header.gguf").is_file()
    }

    pub fn open(dir: &Path) -> Result<Self, String> {
        let json = std::fs::read_to_string(dir.join("manifest.json"))
            .map_err(|e| format!("read manifest: {e}"))?;
        let manifest: Manifest =
            serde_json::from_str(&json).map_err(|e| format!("parse manifest: {e}"))?;
        if manifest.version != PACK_VERSION {
            return Err(format!(
                "pack version {} is not the supported {PACK_VERSION}",
                manifest.version
            ));
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            manifest,
        })
    }

    /// Write the manifest back (after partitioning / calibration).
    pub fn save_manifest(&self) -> Result<(), String> {
        let json = serde_json::to_string_pretty(&self.manifest).map_err(|e| e.to_string())?;
        let tmp = self.dir.join("manifest.json.tmp");
        std::fs::write(&tmp, json).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, self.dir.join("manifest.json")).map_err(|e| e.to_string())
    }

    /// Overwrite every stored level of `entry` from new f32 values of the
    /// same shape, **in place** (same byte sizes — the quantized levels are
    /// re-quantized along the same last dim). Used by the FFN partitioner,
    /// whose permutation keeps shapes.
    pub fn rewrite_entry(&self, entry: &TensorEntry, values: &[f32]) -> Result<(), String> {
        let numel: usize = entry.shape.iter().product();
        if values.len() != numel {
            return Err(format!(
                "rewrite '{}': {} values for shape {:?}",
                entry.name,
                values.len(),
                entry.shape
            ));
        }
        let last = *entry.shape.last().expect("non-empty shape");
        for (&p, blob) in &entry.precisions {
            let (vbytes, sbytes): (Vec<u8>, Vec<u8>) = match p {
                Precision::F32 => (
                    values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    Vec::new(),
                ),
                Precision::F16 => (
                    values
                        .iter()
                        .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
                        .collect(),
                    Vec::new(),
                ),
                Precision::Q8 | Precision::Q4 => {
                    let (q, scales) = quantize_blocks(values, last, p);
                    let v = if p == Precision::Q4 {
                        pack_nibbles(&q)
                    } else {
                        q.iter().map(|&x| x as u8).collect()
                    };
                    (v, scales.iter().flat_map(|s| s.to_le_bytes()).collect())
                }
            };
            if vbytes.len() as u64 != blob.values_len || sbytes.len() as u64 != blob.scales_len {
                return Err(format!(
                    "rewrite '{}' {p:?}: size changed ({} / {} vs {} / {})",
                    entry.name,
                    vbytes.len(),
                    sbytes.len(),
                    blob.values_len,
                    blob.scales_len
                ));
            }
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .open(self.dir.join(p.blob_name()))
                .map_err(|e| format!("open {} for write: {e}", p.blob_name()))?;
            file.seek(SeekFrom::Start(blob.values_offset))
                .map_err(|e| e.to_string())?;
            file.write_all(&vbytes).map_err(|e| e.to_string())?;
            if !sbytes.is_empty() {
                file.seek(SeekFrom::Start(blob.scales_offset))
                    .map_err(|e| e.to_string())?;
                file.write_all(&sbytes).map_err(|e| e.to_string())?;
            }
            file.sync_data().map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// A 2-D entry's **columns** `ranges` (each `(start, len)`, concatenated
    /// in order) as a tensor at `precision`: `[rows, Σ len]`. Quantized
    /// levels are sliced at block granularity (ranges must be block-aligned)
    /// straight from the stored bytes — no re-quantization.
    pub fn tensor_cols(
        &self,
        entry: &TensorEntry,
        precision: Precision,
        ranges: &[(usize, usize)],
        device: &Device,
    ) -> Result<Tensor<2>, String> {
        let &[rows, cols] = entry.shape.as_slice() else {
            return Err(format!("'{}' is not 2-D", entry.name));
        };
        let width: usize = ranges.iter().map(|r| r.1).sum();
        match precision {
            Precision::Q4 | Precision::Q8 => {
                if ranges
                    .iter()
                    .any(|&(s, l)| !s.is_multiple_of(BLOCK) || !l.is_multiple_of(BLOCK))
                {
                    return Err("column ranges must be block-aligned for quantized levels".into());
                }
                let (values, scales) = self.read_quant(entry, precision)?;
                let bpr = cols / BLOCK; // blocks per row
                let mut v = Vec::with_capacity(rows * width);
                let mut s = Vec::with_capacity(rows * width / BLOCK);
                for r in 0..rows {
                    for &(start, len) in ranges {
                        v.extend_from_slice(&values[r * cols + start..r * cols + start + len]);
                        s.extend_from_slice(
                            &scales[r * bpr + start / BLOCK..r * bpr + (start + len) / BLOCK],
                        );
                    }
                }
                let scheme = precision.policy().scheme().expect("quantized level");
                Ok(Tensor::from_data(
                    quantized_tensor_data(&v, &s, [rows, width], scheme),
                    device,
                ))
            }
            Precision::F16 | Precision::F32 => {
                let all = self.read_floats(entry, precision)?;
                let mut v = Vec::with_capacity(rows * width);
                for r in 0..rows {
                    for &(start, len) in ranges {
                        v.extend_from_slice(&all[r * cols + start..r * cols + start + len]);
                    }
                }
                let dtype = crate::backend::float_dtype(device);
                Ok(Tensor::from_data(
                    TensorData::new(v, [rows, width]),
                    (device, dtype),
                ))
            }
        }
    }

    /// A 2-D entry's **rows** `ranges` (concatenated) as a tensor at
    /// `precision`: `[Σ len, cols]`.
    pub fn tensor_rows(
        &self,
        entry: &TensorEntry,
        precision: Precision,
        ranges: &[(usize, usize)],
        device: &Device,
    ) -> Result<Tensor<2>, String> {
        let &[_rows, cols] = entry.shape.as_slice() else {
            return Err(format!("'{}' is not 2-D", entry.name));
        };
        let height: usize = ranges.iter().map(|r| r.1).sum();
        match precision {
            Precision::Q4 | Precision::Q8 => {
                let (values, scales) = self.read_quant(entry, precision)?;
                let bpr = cols / BLOCK;
                let mut v = Vec::with_capacity(height * cols);
                let mut s = Vec::with_capacity(height * bpr);
                for &(start, len) in ranges {
                    v.extend_from_slice(&values[start * cols..(start + len) * cols]);
                    s.extend_from_slice(&scales[start * bpr..(start + len) * bpr]);
                }
                let scheme = precision.policy().scheme().expect("quantized level");
                Ok(Tensor::from_data(
                    quantized_tensor_data(&v, &s, [height, cols], scheme),
                    device,
                ))
            }
            Precision::F16 | Precision::F32 => {
                let all = self.read_floats(entry, precision)?;
                let mut v = Vec::with_capacity(height * cols);
                for &(start, len) in ranges {
                    v.extend_from_slice(&all[start * cols..(start + len) * cols]);
                }
                let dtype = crate::backend::float_dtype(device);
                Ok(Tensor::from_data(
                    TensorData::new(v, [height, cols]),
                    (device, dtype),
                ))
            }
        }
    }

    /// The source header as a payload-less GGUF — config + tokenizer readers
    /// take it as-is.
    pub fn header(&self) -> Result<GgufFile, String> {
        GgufFile::open(&self.dir.join("header.gguf")).map_err(|e| e.to_string())
    }

    pub fn entry(&self, name: &str) -> Option<&TensorEntry> {
        self.manifest.tensors.iter().find(|t| t.name == name)
    }

    fn read_range(&self, precision: Precision, offset: u64, len: u64) -> Result<Vec<u8>, String> {
        let mut file = std::fs::File::open(self.dir.join(precision.blob_name()))
            .map_err(|e| format!("open {}: {e}", precision.blob_name()))?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| e.to_string())?;
        let mut buf = vec![0u8; usize::try_from(len).expect("blob fits")];
        file.read_exact(&mut buf)
            .map_err(|e| format!("read {}: {e}", precision.blob_name()))?;
        Ok(buf)
    }

    /// The f32 values of a tensor read from the REQUESTED float level when
    /// stored: `F16` reads the half-width blob and widens — same tensor in
    /// RAM, half the bytes off the disk, which matters because the pack
    /// lives on an HDD RAID that runs at 100% for the whole load. Falls back
    /// to [`Self::read_f32`]'s best-available order when the requested level
    /// is absent.
    ///
    /// Exists because the float arms of `tensor`/`tensor_cols`/`tensor_rows`
    /// used to funnel through `read_f32`, which prefers f32 — so every
    /// "F16" load silently read `f32.bin`, including the probe that once
    /// "measured" F16 speed on flex.
    pub fn read_floats(&self, entry: &TensorEntry, prefer: Precision) -> Result<Vec<f32>, String> {
        if prefer == Precision::F16
            && let Some(b) = entry.precisions.get(&Precision::F16)
        {
            let bytes = self.read_range(Precision::F16, b.values_offset, b.values_len)?;
            return Ok(bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::f16::from_le_bytes(*c).to_f32())
                .collect());
        }
        self.read_f32(entry)
    }

    /// The f32 values of a tensor at its best float level (f32, else f16
    /// widened, else a quantized level dequantized).
    pub fn read_f32(&self, entry: &TensorEntry) -> Result<Vec<f32>, String> {
        if let Some(b) = entry.precisions.get(&Precision::F32) {
            let bytes = self.read_range(Precision::F32, b.values_offset, b.values_len)?;
            return Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect());
        }
        if let Some(b) = entry.precisions.get(&Precision::F16) {
            let bytes = self.read_range(Precision::F16, b.values_offset, b.values_len)?;
            return Ok(bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::f16::from_le_bytes(*c).to_f32())
                .collect());
        }
        for p in [Precision::Q8, Precision::Q4] {
            if entry.precisions.contains_key(&p) {
                let (q, scales) = self.read_quant(entry, p)?;
                let mut out = Vec::with_capacity(q.len());
                for (i, &v) in q.iter().enumerate() {
                    out.push(f32::from(v) * scales[i / BLOCK]);
                }
                return Ok(out);
            }
        }
        Err(format!("'{}' has no stored precision", entry.name))
    }

    /// The (i8 values, f32 block scales) of a quantized level.
    pub fn read_quant(
        &self,
        entry: &TensorEntry,
        precision: Precision,
    ) -> Result<(Vec<i8>, Vec<f32>), String> {
        let b = entry
            .precisions
            .get(&precision)
            .ok_or_else(|| format!("'{}' has no {precision:?} level", entry.name))?;
        let n: usize = entry.shape.iter().product();
        let vbytes = self.read_range(precision, b.values_offset, b.values_len)?;
        let values: Vec<i8> = match precision {
            Precision::Q4 => unpack_nibbles(&vbytes, n),
            Precision::Q8 => vbytes.iter().map(|&b| b as i8).collect(),
            _ => return Err(format!("{precision:?} is not a quantized level")),
        };
        let sbytes = self.read_range(precision, b.scales_offset, b.scales_len)?;
        let scales: Vec<f32> = sbytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        if values.len() != n || scales.len() != n / BLOCK {
            return Err(format!("'{}' {precision:?} blob size mismatch", entry.name));
        }
        Ok((values, scales))
    }

    /// Build a device tensor for `entry` at `precision`. Quantized levels
    /// arrive through burn's canonical quantized `TensorData` (no
    /// re-quantization); float levels through the backend's float dtype.
    pub fn tensor<const D: usize>(
        &self,
        entry: &TensorEntry,
        precision: Precision,
        device: &Device,
    ) -> Result<Tensor<D>, String> {
        let shape: [usize; D] = entry.shape.clone().try_into().map_err(|_| {
            format!(
                "'{}' is rank {}, asked for {D}",
                entry.name,
                entry.shape.len()
            )
        })?;
        match precision {
            Precision::Q4 | Precision::Q8 => {
                let (values, scales) = self.read_quant(entry, precision)?;
                let scheme: QuantScheme = precision
                    .policy()
                    .scheme()
                    .expect("quantized level has a scheme");
                let data = quantized_tensor_data(&values, &scales, shape, scheme);
                Ok(Tensor::from_data(data, device))
            }
            Precision::F16 | Precision::F32 => {
                let values = self.read_floats(entry, precision)?;
                let dtype = crate::backend::float_dtype(device);
                Ok(Tensor::from_data(
                    TensorData::new(values, shape),
                    (device, dtype),
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_quantizer_roundtrips_within_half_scale() {
        let vals: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.37).sin() * 3.0).collect();
        for p in [Precision::Q8, Precision::Q4] {
            let (q, scales) = quantize_blocks(&vals, 64, p);
            assert_eq!(scales.len(), 8);
            let range_max = if p == Precision::Q8 { 127.0 } else { 7.0 };
            for (i, (&v, &qq)) in vals.iter().zip(&q).enumerate() {
                let back = f32::from(qq) * scales[i / BLOCK];
                assert!(
                    (v - back).abs() <= scales[i / BLOCK] / 2.0 + 1e-6,
                    "{p:?} elem {i}: {v} vs {back} (scale {})",
                    scales[i / BLOCK]
                );
                assert!(qq.abs() as f32 <= range_max);
            }
        }
    }

    #[test]
    fn nibble_packing_roundtrips() {
        let vals: Vec<i8> = (-7..=7).chain([0, 7, -7, 3]).collect();
        let packed = pack_nibbles(&vals);
        assert_eq!(packed.len(), vals.len().div_ceil(2));
        assert_eq!(unpack_nibbles(&packed, vals.len()), vals);
    }

    #[test]
    fn precision_list_parses_and_dedups() {
        assert_eq!(
            Precision::parse_list("f32,q4,q8,q4").unwrap(),
            vec![Precision::Q4, Precision::Q8, Precision::F32]
        );
        assert!(Precision::parse_list("q3").is_err());
    }

    /// A quantized level written by the pack quantizer reconstructs to a
    /// burn tensor whose dequantized values match our own dequant exactly.
    #[test]
    fn canonical_quantized_tensor_data_roundtrip() {
        let device = crate::backend::cpu_device();
        for (p, rows, cols) in [
            (Precision::Q8, 32, 64),
            (Precision::Q4, 32, 64),
            // The real shape the 2B gate trips on — exercises burn's Q4 packing at scale.
            (Precision::Q4, 2048, 6144),
        ] {
            let n = rows * cols;
            let vals: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.11).cos()).collect();
            let (q, scales) = quantize_blocks(&vals, cols, p);
            let scheme = p.policy().scheme().unwrap();
            let data = quantized_tensor_data(&q, &scales, [rows, cols], scheme);
            let t = Tensor::<2>::from_data(data, &device);
            let back = t.dequantize().into_data().try_to_vec::<f32>().unwrap();
            assert_eq!(back.len(), n, "{p:?} [{rows}, {cols}]");
            for (i, (&qq, &b)) in q.iter().zip(&back).enumerate().step_by(97) {
                let ours = f32::from(qq) * scales[i / BLOCK];
                assert!((ours - b).abs() < 1e-5, "{p:?} elem {i}: {ours} vs {b}");
            }
        }
    }

    #[test]
    fn column_and_row_slices_round_trip_through_a_written_pack() {
        // Build a tiny two-tensor pack by hand, then slice cluster ranges
        // back out at f32 and Q8 and compare to the source rows/columns.
        let dir = std::env::temp_dir().join(format!("mummu-pack-slice-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (rows, cols) = (64usize, 128usize); // both dims multiples of BLOCK
        let gate: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32) * 0.017).sin())
            .collect();
        let down: Vec<f32> = (0..cols * rows)
            .map(|i| ((i as f32) * 0.023).cos())
            .collect();
        // Write f32 and Q8 blobs with two entries.
        let mut f32w = BlobWriter {
            file: std::io::BufWriter::new(std::fs::File::create(dir.join("f32.bin")).unwrap()),
            len: 0,
            unsynced: 0,
        };
        let mut q8w = BlobWriter {
            file: std::io::BufWriter::new(std::fs::File::create(dir.join("q8.bin")).unwrap()),
            len: 0,
            unsynced: 0,
        };
        let mut entry = |name: &str, vals: &[f32], shape: Vec<usize>| -> TensorEntry {
            let last = *shape.last().unwrap();
            let fbytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            let (fo, fl) = f32w.append(&fbytes).unwrap();
            let (q, sc) = quantize_blocks(vals, last, Precision::Q8);
            let qb: Vec<u8> = q.iter().map(|&x| x as u8).collect();
            let sb: Vec<u8> = sc.iter().flat_map(|s| s.to_le_bytes()).collect();
            let (qo, ql) = q8w.append(&qb).unwrap();
            let (so, sl) = q8w.append(&sb).unwrap();
            TensorEntry {
                name: name.into(),
                role: Role::Linear,
                shape,
                precisions: [
                    (
                        Precision::F32,
                        Blob {
                            values_offset: fo,
                            values_len: fl,
                            scales_offset: 0,
                            scales_len: 0,
                        },
                    ),
                    (
                        Precision::Q8,
                        Blob {
                            values_offset: qo,
                            values_len: ql,
                            scales_offset: so,
                            scales_len: sl,
                        },
                    ),
                ]
                .into_iter()
                .collect(),
            }
        };
        let ge = entry("gate", &gate, vec![rows, cols]);
        let de = entry("down", &down, vec![cols, rows]);
        f32w.file.flush().unwrap();
        q8w.file.flush().unwrap();
        let manifest = Manifest {
            version: PACK_VERSION,
            source_file: String::new(),
            source_bytes: 0,
            architecture: "test".into(),
            precisions: vec![Precision::F32, Precision::Q8],
            tensors: vec![ge.clone(), de.clone()],
            ffn_partition: None,
        };
        std::fs::write(
            dir.join("manifest.json"),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();
        // No header.gguf here — construct Pack directly.
        let pack = Pack {
            dir: dir.clone(),
            manifest,
        };
        let device = crate::backend::cpu_device();
        // Columns [32,32) and [96,32) of gate → [rows, 64].
        let ranges = [(32usize, 32usize), (96, 32)];
        let cslab = pack
            .tensor_cols(&ge, Precision::F32, &ranges, &device)
            .unwrap();
        assert_eq!(cslab.dims(), [rows, 64]);
        let got = cslab.into_data().try_to_vec::<f32>().unwrap();
        for r in 0..rows {
            for (k, &(start, len)) in ranges.iter().enumerate() {
                let base: usize = ranges[..k].iter().map(|x| x.1).sum();
                for j in 0..len {
                    let want = gate[r * cols + start + j];
                    assert!(
                        (got[r * 64 + base + j] - want).abs() < 1e-6,
                        "col f32 r{r} j{j}"
                    );
                }
            }
        }
        // Same ranges as rows of down → [64, rows].
        let rslab = pack
            .tensor_rows(&de, Precision::F32, &ranges, &device)
            .unwrap();
        assert_eq!(rslab.dims(), [64, rows]);
        let gotr = rslab.into_data().try_to_vec::<f32>().unwrap();
        for (k, &(start, len)) in ranges.iter().enumerate() {
            let base: usize = ranges[..k].iter().map(|x| x.1).sum();
            for i in 0..len {
                for c in 0..rows {
                    let want = down[(start + i) * rows + c];
                    assert!(
                        (gotr[(base + i) * rows + c] - want).abs() < 1e-6,
                        "row f32 i{i} c{c}"
                    );
                }
            }
        }
        // Q8 column slice dequantizes close to the source.
        let cq = pack
            .tensor_cols(&ge, Precision::Q8, &ranges, &device)
            .unwrap();
        let dq = cq.dequantize().into_data().try_to_vec::<f32>().unwrap();
        for r in 0..rows {
            for (k, &(start, len)) in ranges.iter().enumerate() {
                let base: usize = ranges[..k].iter().map(|x| x.1).sum();
                for j in 0..len {
                    let want = gate[r * cols + start + j];
                    assert!(
                        (dq[r * 64 + base + j] - want).abs() < 0.05,
                        "col q8 r{r} j{j}"
                    );
                }
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
