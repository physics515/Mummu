//! Safetensors **reader** + a fusing rewriter for checkpoints whose on-disk
//! layout does not match the module layout.
//!
//! `burn-store` reads safetensors for us on the ordinary path, so this module
//! exists for the two things it cannot do:
//!
//! 1. **Sharded checkpoints.** Anything past ~5 GB ships as
//!    `model-0000N-of-0000M.safetensors` + a `model.safetensors.index.json`
//!    weight map. `import::weights_file` only ever finds a single
//!    `model.safetensors`.
//! 2. **N:1 tensor fusion.** `burn-store`'s remapping is 1:1 (rename); an MoE
//!    checkpoint stores every expert separately
//!    (`mlp.experts.{0..63}.gate_proj.weight`) while the module holds ONE
//!    fused `[experts, out, in]` param — exactly the ggml `ffn_*_exps` layout
//!    a GGUF ships pre-fused.
//!
//! [`fuse_checkpoint`] reads every shard's header, plans the output layout,
//! validates it, and only then copies payload bytes — so a checkpoint missing
//! expert 37 fails before a single weight byte is read, rather than loading
//! clean and computing wrong. The result is an ordinary in-memory safetensors
//! blob that goes through the SAME `SafetensorsStore::from_bytes` +
//! adapter-chain + `load_checked` pipeline as every other import path (the
//! GGUF path's `dequant_to_safetensors` is the precedent).
//!
//! Source dtypes are preserved verbatim — the bf16→backend-float cast stays
//! where it already lives, in `CastFloatAdapter` on the load pipeline.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Largest safetensors JSON header accepted, per shard. Real headers are a
/// few hundred KB (one entry per tensor); 64 MiB is a corrupt or hostile file.
const MAX_HEADER_BYTES: u64 = 64 << 20;

/// Largest tensor count accepted across a whole checkpoint. A 64-expert MoE
/// at 16 layers already declares ~3 000; 1M is a runaway index.
const MAX_TENSORS: usize = 1 << 20;

/// Largest fused payload [`fuse_checkpoint`] will build in RAM. Matches the
/// GGUF path's ceiling (the reference machine has 128 GB) — note this blob
/// keeps the SOURCE dtype, so a bf16 checkpoint costs half its f32 footprint.
const MAX_FUSED_BYTES: u64 = 48 << 30;

/// Largest SINGLE tensor (or fused member) the streaming fuse will buffer.
/// The widest real one we carry is a 64-expert projection member at ~4 MiB;
/// 4 GiB is a corrupt header claiming an absurd tensor.
const MAX_PART_BYTES: u64 = 4 << 30;

/// Largest number of shards in an index (mirrors `hub::MAX_SHARDS`).
const MAX_SHARDS: usize = 256;

/// What went wrong reading or fusing a safetensors checkpoint.
#[derive(Debug, thiserror::Error)]
pub enum SafetensorsError {
    #[error("safetensors {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("safetensors {path}: header is not valid JSON: {reason}")]
    BadHeader { path: String, reason: String },
    #[error("safetensors {path}: {what} {count} exceeds the {bound} bound")]
    OverBound {
        path: String,
        what: &'static str,
        count: u64,
        bound: u64,
    },
    #[error("safetensors {path}: tensor '{name}': {reason}")]
    BadTensor {
        path: String,
        name: String,
        reason: String,
    },
    /// A fused group is not exactly `count` distinct members `0..count`. This
    /// is the load-bearing check: a silently short expert bank would load
    /// clean and compute wrong.
    #[error("fused tensor '{target}': {reason}")]
    BadGroup { target: String, reason: String },
    #[error("no safetensors checkpoint in {0} (looked for model.safetensors and *.index.json)")]
    NoCheckpoint(PathBuf),
    /// The index names a shard that is not on disk. A sibling `.part` means an
    /// interrupted download, which is worth saying out loud: the alternative is
    /// this surfacing as a bare `os error 2` from deep inside the load.
    #[error("incomplete checkpoint {dir}: index names {missing}, which is not present{hint}")]
    MissingShard {
        dir: PathBuf,
        missing: String,
        hint: &'static str,
    },
}

/// `u64` -> `usize` as an error rather than a panic.
///
/// Every caller has already bounded `value` against a file length or one of
/// the `MAX_*` ceilings, so on a 64-bit target this cannot fail — but an
/// import path is exactly where "cannot fail" should still be an `Err`
/// instead of an `.expect()`, so a 32-bit build degrades to a clean error.
fn to_usize(value: u64, path: &Path, what: &'static str) -> Result<usize, SafetensorsError> {
    debug_assert!(
        value <= usize::MAX as u64,
        "{what} fits usize on this target"
    );
    usize::try_from(value).map_err(|_| SafetensorsError::OverBound {
        path: path.display().to_string(),
        what,
        count: value,
        bound: usize::MAX as u64,
    })
}

/// One tensor as the on-disk header describes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorEntry {
    /// Safetensors dtype token, verbatim (`BF16`, `F32`, `I64`, …).
    pub dtype: String,
    /// Row-major shape.
    pub shape: Vec<u64>,
    /// `[start, end)` within the shard's payload region.
    pub offsets: (u64, u64),
}

impl TensorEntry {
    /// Payload length in bytes.
    #[must_use]
    pub fn byte_len(&self) -> u64 {
        debug_assert!(
            self.offsets.1 >= self.offsets.0,
            "offsets validated on read"
        );
        self.offsets.1 - self.offsets.0
    }

    /// Elements implied by the shape (1 for a scalar — an empty shape).
    #[must_use]
    pub fn element_count(&self) -> u64 {
        self.shape.iter().product()
    }
}

/// A parsed safetensors header: the tensor table plus where its payload starts.
#[derive(Debug)]
pub struct SafetensorsHeader {
    pub path: PathBuf,
    /// Entries in **file order** (the order payload bytes appear), which is
    /// what makes a sequential read cheap.
    pub tensors: Vec<(String, TensorEntry)>,
    /// Absolute file offset where the payload region begins.
    pub data_offset: u64,
}

impl SafetensorsHeader {
    /// Read and validate one shard's header. No payload bytes are read.
    pub fn open(path: &Path) -> Result<Self, SafetensorsError> {
        let io = |source: std::io::Error| SafetensorsError::Io {
            path: path.display().to_string(),
            source,
        };
        let mut file = BufReader::new(File::open(path).map_err(io)?);
        let file_len = file.get_ref().metadata().map_err(io)?.len();

        let mut len_bytes = [0u8; 8];
        file.read_exact(&mut len_bytes).map_err(io)?;
        let header_len = u64::from_le_bytes(len_bytes);
        if header_len > MAX_HEADER_BYTES || header_len.saturating_add(8) > file_len {
            return Err(SafetensorsError::OverBound {
                path: path.display().to_string(),
                what: "header bytes",
                count: header_len,
                bound: MAX_HEADER_BYTES.min(file_len),
            });
        }
        let mut header = vec![0u8; to_usize(header_len, path, "header bytes")?];
        file.read_exact(&mut header).map_err(io)?;

        let data_offset = 8 + header_len;
        assert!(data_offset <= file_len, "payload starts inside the file");
        let payload_len = file_len - data_offset;

        let parsed = Self::parse_header(&header, path, payload_len)?;
        Ok(Self {
            path: path.to_path_buf(),
            tensors: parsed,
            data_offset,
        })
    }

    /// Parse the JSON header into a file-ordered tensor table, validating every
    /// entry against the payload region it claims.
    fn parse_header(
        header: &[u8],
        path: &Path,
        payload_len: u64,
    ) -> Result<Vec<(String, TensorEntry)>, SafetensorsError> {
        let bad_header = |reason: String| SafetensorsError::BadHeader {
            path: path.display().to_string(),
            reason,
        };
        let json: serde_json::Value =
            serde_json::from_slice(header).map_err(|e| bad_header(e.to_string()))?;
        let object = json
            .as_object()
            .ok_or_else(|| bad_header("header is not a JSON object".into()))?;
        if object.len() > MAX_TENSORS {
            return Err(SafetensorsError::OverBound {
                path: path.display().to_string(),
                what: "tensor entries",
                count: object.len() as u64,
                bound: MAX_TENSORS as u64,
            });
        }

        let mut tensors = Vec::with_capacity(object.len());
        for (name, value) in object {
            // `__metadata__` is a free-form string map, not a tensor.
            if name == "__metadata__" {
                continue;
            }
            let bad = |reason: String| SafetensorsError::BadTensor {
                path: path.display().to_string(),
                name: name.clone(),
                reason,
            };
            let entry = value
                .as_object()
                .ok_or_else(|| bad("entry is not an object".into()))?;
            let dtype = entry
                .get("dtype")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| bad("missing 'dtype'".into()))?
                .to_string();
            let element_bytes =
                dtype_bytes(&dtype).ok_or_else(|| bad(format!("unsupported dtype '{dtype}'")))?;
            let shape: Vec<u64> = entry
                .get("shape")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| bad("missing 'shape'".into()))?
                .iter()
                .map(|d| d.as_u64().ok_or_else(|| bad("non-integer dim".into())))
                .collect::<Result<_, _>>()?;
            let offsets = entry
                .get("data_offsets")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| bad("missing 'data_offsets'".into()))?;
            if offsets.len() != 2 {
                return Err(bad(format!("data_offsets has {} entries", offsets.len())));
            }
            let start = offsets[0]
                .as_u64()
                .ok_or_else(|| bad("non-integer data_offset".into()))?;
            let end = offsets[1]
                .as_u64()
                .ok_or_else(|| bad("non-integer data_offset".into()))?;
            if end < start || end > payload_len {
                return Err(bad(format!(
                    "data_offsets [{start}, {end}) outside the {payload_len}-byte payload"
                )));
            }
            // Positive AND negative space: the declared shape must be exactly
            // the bytes claimed — a mismatch is a corrupt or mis-declared
            // tensor, never something to load and hope about.
            let expected = shape
                .iter()
                .try_fold(element_bytes, |acc: u64, d| acc.checked_mul(*d))
                .ok_or_else(|| bad("shape overflows u64 bytes".into()))?;
            if expected != end - start {
                return Err(bad(format!(
                    "shape {shape:?} of {dtype} implies {expected} bytes, header claims {}",
                    end - start
                )));
            }
            tensors.push((
                name.clone(),
                TensorEntry {
                    dtype,
                    shape,
                    offsets: (start, end),
                },
            ));
        }
        // File order (by payload offset) makes the copy pass a sequential read.
        tensors.sort_by_key(|(_, t)| t.offsets.0);
        Ok(tensors)
    }

    /// Read one tensor's payload bytes.
    fn read_payload(&self, entry: &TensorEntry, into: &mut [u8]) -> Result<(), SafetensorsError> {
        assert_eq!(
            into.len() as u64,
            entry.byte_len(),
            "read_payload: destination sized to the entry"
        );
        let io = |source: std::io::Error| SafetensorsError::Io {
            path: self.path.display().to_string(),
            source,
        };
        let mut file = File::open(&self.path).map_err(io)?;
        file.seek(SeekFrom::Start(self.data_offset + entry.offsets.0))
            .map_err(io)?;
        file.read_exact(into).map_err(io)
    }
}

/// Bytes per element for a safetensors dtype token, or `None` if unsupported.
#[must_use]
pub fn dtype_bytes(dtype: &str) -> Option<u64> {
    Some(match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => 1,
        "U16" | "I16" | "F16" | "BF16" => 2,
        "U32" | "I32" | "F32" => 4,
        "U64" | "I64" | "F64" => 8,
        _ => return None,
    })
}

/// What [`fuse_checkpoint`] should do with a source tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fuse {
    /// Copy through under this (possibly renamed) target name.
    Keep(String),
    /// Contribute to `target` as member `index` of `count`, stacked along a
    /// NEW leading axis — the N:1 case (`mlp.experts.7.gate_proj.weight` is
    /// member 7 of the fused `mlp.experts.gate`).
    Stack {
        target: String,
        index: usize,
        count: usize,
    },
    /// Drop this tensor (present in the checkpoint, unused by the module).
    Drop,
}

/// Every safetensors shard of a checkpoint dir, in index order.
///
/// A single `model.safetensors` is the one-shard case; otherwise
/// `model.safetensors.index.json`'s weight map names the shards.
pub fn checkpoint_shards(dir: &Path) -> Result<Vec<PathBuf>, SafetensorsError> {
    let single = dir.join("model.safetensors");
    if single.is_file() {
        return Ok(vec![single]);
    }
    let index_path = dir.join("model.safetensors.index.json");
    if !index_path.is_file() {
        return Err(SafetensorsError::NoCheckpoint(dir.to_path_buf()));
    }
    let bytes = std::fs::read(&index_path).map_err(|source| SafetensorsError::Io {
        path: index_path.display().to_string(),
        source,
    })?;
    let names = crate::hub::shards_from_index(&bytes, &index_path).map_err(|e| {
        SafetensorsError::BadHeader {
            path: index_path.display().to_string(),
            reason: e.to_string(),
        }
    })?;
    assert!(
        !names.is_empty() && names.len() <= MAX_SHARDS,
        "shards_from_index bounds the count"
    );

    // The index is a manifest, not evidence. Check every shard is actually on
    // disk HERE, so a half-fetched checkpoint is one clear error at planning
    // time rather than an `os error 2` raised after the first shards have
    // already been opened — and so callers can use this function as the
    // "is the checkpoint complete?" question it looks like.
    let paths: Vec<PathBuf> = names.into_iter().map(|n| dir.join(n)).collect();
    for path in &paths {
        if path.is_file() {
            continue;
        }
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        let interrupted = path.with_file_name(format!("{name}.part")).is_file();
        return Err(SafetensorsError::MissingShard {
            dir: dir.to_path_buf(),
            missing: name.into_owned(),
            hint: if interrupted {
                " — a .part sibling is present, so the download was interrupted"
            } else {
                ""
            },
        });
    }
    debug_assert!(
        paths.iter().all(|p| p.is_file()),
        "every shard verified present before returning"
    );
    Ok(paths)
}

/// Read every shard of the checkpoint in `dir` and build ONE in-memory
/// safetensors blob, applying `map` to each source tensor name.
///
/// Fused (`Fuse::Stack`) groups are concatenated in **numeric member order**,
/// gaining a leading `count` axis — `count` tensors of shape `[out, in]`
/// become one `[count, out, in]`. Every group must be exactly complete;
/// a missing or duplicate member is a loud [`SafetensorsError::BadGroup`],
/// raised during planning, before any payload byte is read.
pub fn fuse_checkpoint(
    dir: &Path,
    map: &dyn Fn(&str) -> Option<Fuse>,
) -> Result<Vec<u8>, SafetensorsError> {
    let mut blob = Vec::new();
    fuse_into(dir, map, &mut blob)?;
    Ok(blob)
}

/// [`fuse_checkpoint`] straight to a file, never holding the payload in RAM.
///
/// This is the variant a real checkpoint wants. The in-memory form needs the
/// whole fused payload resident — 13.8 GB for OLMoE-1B-7B — *on top of* the
/// model the load then builds (~28 GB at f32), and that sum is what a 128 GB
/// box with other tenants actually fails to satisfy. Writing to disk trades
/// the spike for temp space and lets `SafetensorsStore::from_file` page the
/// weights in as it needs them.
pub fn fuse_checkpoint_to_file(
    dir: &Path,
    map: &dyn Fn(&str) -> Option<Fuse>,
    out: &Path,
) -> Result<u64, SafetensorsError> {
    let io = |source: std::io::Error| SafetensorsError::Io {
        path: out.display().to_string(),
        source,
    };
    let file = File::create(out).map_err(io)?;
    let mut sink = std::io::BufWriter::with_capacity(1 << 20, file);
    let written = fuse_into(dir, map, &mut sink)?;
    sink.flush().map_err(io)?;
    sink.into_inner()
        .map_err(|e| SafetensorsError::Io {
            path: out.display().to_string(),
            source: e.into_error(),
        })?
        .sync_all()
        .map_err(io)?;
    Ok(written)
}

/// The shared fuse: plan, then stream header + payload into `sink` in output
/// order, one part at a time.
///
/// Writes are strictly ascending because `plan` carries tensors in output
/// order and each tensor's `parts` are already in destination order — so this
/// never needs the payload addressable at once, only ONE part at a time.
fn fuse_into<W: std::io::Write>(
    dir: &Path,
    map: &dyn Fn(&str) -> Option<Fuse>,
    sink: &mut W,
) -> Result<u64, SafetensorsError> {
    let shard_paths = checkpoint_shards(dir)?;
    let headers = shard_paths
        .iter()
        .map(|p| SafetensorsHeader::open(p))
        .collect::<Result<Vec<_>, _>>()?;

    let plan = plan_output(&headers, map)?;
    let total: u64 = plan.iter().map(|p| p.len).sum();
    if total > MAX_FUSED_BYTES {
        return Err(SafetensorsError::OverBound {
            path: dir.display().to_string(),
            what: "fused payload bytes",
            count: total,
            bound: MAX_FUSED_BYTES,
        });
    }

    // Header first: names, dtypes, shapes, and the contiguous offsets the
    // copy pass will fill.
    let mut header = String::from("{");
    for (i, p) in plan.iter().enumerate() {
        if i > 0 {
            header.push(',');
        }
        let json_name =
            serde_json::to_string(&p.name).map_err(|e| SafetensorsError::BadTensor {
                path: dir.display().to_string(),
                name: p.name.clone(),
                reason: format!("name is not encodable as JSON: {e}"),
            })?;
        header.push_str(&format!(
            "{json_name}:{{\"dtype\":\"{}\",\"shape\":{:?},\"data_offsets\":[{},{}]}}",
            p.dtype,
            p.shape,
            p.start,
            p.start + p.len,
        ));
    }
    header.push('}');

    let io = |source: std::io::Error| SafetensorsError::Io {
        path: dir.display().to_string(),
        source,
    };
    sink.write_all(&(header.len() as u64).to_le_bytes())
        .map_err(io)?;
    sink.write_all(header.as_bytes()).map_err(io)?;

    // One reusable buffer, sized to the largest single part rather than to the
    // payload: for a 64-expert bank that is one expert projection (~4 MB), not
    // the 13.8 GB whole.
    let widest = plan
        .iter()
        .flat_map(|p| p.parts.iter())
        .map(|(_, entry, _)| entry.byte_len())
        .max()
        .unwrap_or(0);
    if widest > MAX_PART_BYTES {
        return Err(SafetensorsError::OverBound {
            path: dir.display().to_string(),
            what: "single tensor bytes",
            count: widest,
            bound: MAX_PART_BYTES,
        });
    }
    let mut buf = vec![0u8; to_usize(widest, dir, "single tensor bytes")?];

    let mut written = 0u64;
    for p in &plan {
        debug_assert_eq!(written, p.start, "tensors are written in output order");
        for (shard, entry, dst_within) in &p.parts {
            debug_assert_eq!(
                written,
                p.start + dst_within,
                "parts are written in destination order"
            );
            let len = to_usize(entry.byte_len(), dir, "member byte length")?;
            let slot = &mut buf[..len];
            headers[*shard].read_payload(entry, slot)?;
            sink.write_all(slot).map_err(io)?;
            written += len as u64;
        }
    }
    assert_eq!(
        written, total,
        "every planned byte was written exactly once"
    );
    Ok(written)
}

/// Plan the output layout and validate it. Nothing here reads payload bytes —
/// a malformed checkpoint fails before the expensive pass.
fn plan_output(
    headers: &[SafetensorsHeader],
    map: &dyn Fn(&str) -> Option<Fuse>,
) -> Result<Vec<PlannedNamed>, SafetensorsError> {
    assert!(
        !headers.is_empty(),
        "planning needs at least one shard header"
    );
    assert!(
        headers.len() <= MAX_SHARDS,
        "shard count is bounded before planning"
    );

    // Collect, in first-seen order, what each target is built from.
    let mut order: Vec<String> = Vec::new();
    let mut groups: HashMap<String, Group> = HashMap::new();

    for (shard, header) in headers.iter().enumerate() {
        for (name, entry) in &header.tensors {
            let action = map(name).ok_or_else(|| SafetensorsError::BadTensor {
                path: header.path.display().to_string(),
                name: name.clone(),
                reason: "unmapped tensor name".into(),
            })?;
            let (target, member, count) = match action {
                Fuse::Drop => continue,
                Fuse::Keep(target) => (target, 0usize, 1usize),
                Fuse::Stack {
                    target,
                    index,
                    count,
                } => {
                    if index >= count {
                        return Err(SafetensorsError::BadGroup {
                            target,
                            reason: format!("member index {index} outside 0..{count}"),
                        });
                    }
                    (target, index, count)
                }
            };
            let group = groups.entry(target.clone()).or_insert_with(|| {
                order.push(target.clone());
                Group {
                    count,
                    stacked: count > 1,
                    members: Vec::new(),
                }
            });
            if group.count != count {
                return Err(SafetensorsError::BadGroup {
                    target,
                    reason: format!("member count disagrees ({} vs {count})", group.count),
                });
            }
            if group.members.iter().any(|(i, _, _)| *i == member) {
                return Err(SafetensorsError::BadGroup {
                    target,
                    reason: format!("duplicate member {member}"),
                });
            }
            group.members.push((member, shard, entry.clone()));
        }
    }

    let mut plan = Vec::with_capacity(order.len());
    let mut cursor = 0u64;
    for target in order {
        // `order` is pushed only from `or_insert_with`, so every target in it
        // has a group and none repeats — an `Err` here would mean that
        // invariant broke, not that the checkpoint is bad.
        let Some(mut group) = groups.remove(&target) else {
            return Err(SafetensorsError::BadGroup {
                target,
                reason: "planned target has no collected members".into(),
            });
        };
        // THE load-bearing check: exactly `count` members, ids 0..count.
        if group.members.len() != group.count {
            return Err(SafetensorsError::BadGroup {
                target,
                reason: format!(
                    "{} of {} members present — a short group would load clean and compute wrong",
                    group.members.len(),
                    group.count
                ),
            });
        }
        // Numeric member order, NOT the lexicographic order the names imply
        // (…experts.10 sorts before …experts.2 as text).
        group.members.sort_by_key(|(i, _, _)| *i);
        debug_assert!(
            group
                .members
                .iter()
                .enumerate()
                .all(|(i, (m, _, _))| i == *m),
            "a complete, duplicate-free group is exactly 0..count once sorted"
        );

        let first = &group.members[0].2;
        for (_, _, entry) in &group.members {
            if entry.dtype != first.dtype || entry.shape != first.shape {
                return Err(SafetensorsError::BadGroup {
                    target,
                    reason: format!(
                        "members disagree on layout ({} {:?} vs {} {:?})",
                        first.dtype, first.shape, entry.dtype, entry.shape
                    ),
                });
            }
        }

        let shape = if group.stacked {
            std::iter::once(group.count as u64)
                .chain(first.shape.iter().copied())
                .collect()
        } else {
            first.shape.clone()
        };
        let member_bytes = first.byte_len();
        let len = member_bytes * group.count as u64;
        let parts = group
            .members
            .iter()
            .enumerate()
            .map(|(slot, (_, shard, entry))| (*shard, entry.clone(), slot as u64 * member_bytes))
            .collect();
        plan.push(PlannedNamed {
            name: target,
            dtype: first.dtype.clone(),
            shape,
            start: cursor,
            len,
            parts,
        });
        cursor += len;
    }
    assert!(
        !plan.is_empty(),
        "a checkpoint dir with shards yields at least one output tensor"
    );
    Ok(plan)
}

/// Accumulator for one output tensor while planning.
struct Group {
    count: usize,
    stacked: bool,
    members: Vec<(usize, usize, TensorEntry)>,
}

/// A planned output tensor: where it lands in the blob and what fills it.
struct PlannedNamed {
    name: String,
    dtype: String,
    shape: Vec<u64>,
    start: u64,
    len: u64,
    /// Sources, in destination order: (shard index, entry, byte offset within
    /// this tensor). One part for `Keep`, `count` parts for `Stack`.
    parts: Vec<(usize, TensorEntry, u64)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic safetensors file: `(name, dtype, shape, bytes)`.
    fn write_st(path: &Path, tensors: &[(&str, &str, Vec<u64>, Vec<u8>)]) {
        let mut header = String::from("{");
        let mut data: Vec<u8> = Vec::new();
        for (i, (name, dtype, shape, bytes)) in tensors.iter().enumerate() {
            if i > 0 {
                header.push(',');
            }
            let start = data.len();
            data.extend_from_slice(bytes);
            header.push_str(&format!(
                "{:?}:{{\"dtype\":\"{dtype}\",\"shape\":{shape:?},\"data_offsets\":[{start},{}]}}",
                name,
                data.len()
            ));
        }
        header.push('}');
        let mut blob = (header.len() as u64).to_le_bytes().to_vec();
        blob.extend_from_slice(header.as_bytes());
        blob.extend_from_slice(&data);
        std::fs::write(path, blob).unwrap();
    }

    /// A fresh empty scratch dir under the OS temp root.
    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("mummu_st_{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// One f32 tensor's little-endian bytes.
    fn f32s(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// Read a tensor back out of a fused blob by name.
    fn read_back(blob: &[u8], name: &str) -> (String, Vec<u64>, Vec<f32>) {
        let header_len = u64::from_le_bytes(blob[..8].try_into().unwrap()) as usize;
        let json: serde_json::Value = serde_json::from_slice(&blob[8..8 + header_len]).unwrap();
        let entry = &json[name];
        let dtype = entry["dtype"].as_str().unwrap().to_string();
        let shape: Vec<u64> = entry["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_u64().unwrap())
            .collect();
        let start = entry["data_offsets"][0].as_u64().unwrap() as usize;
        let end = entry["data_offsets"][1].as_u64().unwrap() as usize;
        let base = 8 + header_len;
        let (words, rest) = blob[base + start..base + end].as_chunks::<4>();
        assert!(rest.is_empty(), "f32 payload is a whole number of words");
        let values = words.iter().copied().map(f32::from_le_bytes).collect();
        (dtype, shape, values)
    }

    #[test]
    fn header_parses_shape_dtype_and_offsets() {
        let dir = scratch("header");
        let path = dir.join("model.safetensors");
        write_st(
            &path,
            &[
                ("a", "F32", vec![2, 2], f32s(&[1.0, 2.0, 3.0, 4.0])),
                ("b", "F32", vec![3], f32s(&[5.0, 6.0, 7.0])),
            ],
        );
        let h = SafetensorsHeader::open(&path).unwrap();
        assert_eq!(h.tensors.len(), 2);
        assert_eq!(h.tensors[0].0, "a");
        assert_eq!(h.tensors[0].1.shape, vec![2, 2]);
        assert_eq!(h.tensors[0].1.byte_len(), 16);
        assert_eq!(h.tensors[0].1.element_count(), 4);
        assert_eq!(h.tensors[1].1.offsets, (16, 28));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn header_rejects_a_shape_that_disagrees_with_its_bytes() {
        let dir = scratch("badshape");
        let path = dir.join("model.safetensors");
        // Declares [4] f32 (16 B) but only 8 B of payload are claimed.
        let mut blob = Vec::new();
        let header = r#"{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,8]}}"#.to_string();
        blob.extend_from_slice(&(header.len() as u64).to_le_bytes());
        blob.extend_from_slice(header.as_bytes());
        blob.extend_from_slice(&[0u8; 8]);
        std::fs::write(&path, blob).unwrap();

        let err = SafetensorsHeader::open(&path).unwrap_err();
        assert!(
            matches!(&err, SafetensorsError::BadTensor { reason, .. }
                if reason.contains("implies 16 bytes")),
            "got {err}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn header_rejects_an_unsupported_dtype() {
        let dir = scratch("baddtype");
        let path = dir.join("model.safetensors");
        write_st(&path, &[("a", "COMPLEX128", vec![1], vec![0u8; 16])]);
        let err = SafetensorsHeader::open(&path).unwrap_err();
        assert!(
            matches!(&err, SafetensorsError::BadTensor { reason, .. }
                if reason.contains("unsupported dtype")),
            "got {err}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// THE ordering test. Members are stacked in NUMERIC order; the names sort
    /// lexicographically as 0, 1, 10, 2, … so a text sort silently permutes
    /// the expert bank — a model that loads clean and computes wrong.
    #[test]
    fn stack_orders_members_numerically_not_lexicographically() {
        let dir = scratch("order");
        let count = 12usize;
        let tensors: Vec<(String, &str, Vec<u64>, Vec<u8>)> = (0..count)
            .map(|i| {
                (
                    format!("mlp.experts.{i}.gate_proj.weight"),
                    "F32",
                    vec![1],
                    // Expert i holds exactly the value i.
                    f32s(&[i as f32]),
                )
            })
            .collect();
        let refs: Vec<(&str, &str, Vec<u64>, Vec<u8>)> = tensors
            .iter()
            .map(|(n, d, s, b)| (n.as_str(), *d, s.clone(), b.clone()))
            .collect();
        write_st(&dir.join("model.safetensors"), &refs);

        let blob = fuse_checkpoint(&dir, &|name| {
            let idx: usize = name
                .strip_prefix("mlp.experts.")?
                .strip_suffix(".gate_proj.weight")?
                .parse()
                .ok()?;
            Some(Fuse::Stack {
                target: "mlp.experts.gate".into(),
                index: idx,
                count,
            })
        })
        .unwrap();

        let (dtype, shape, values) = read_back(&blob, "mlp.experts.gate");
        assert_eq!(dtype, "F32");
        assert_eq!(shape, vec![count as u64, 1], "gains a leading expert axis");
        // Slot i must hold expert i — the whole point.
        assert_eq!(
            values,
            (0..count).map(|i| i as f32).collect::<Vec<_>>(),
            "experts must stack in numeric order (lexicographic would give 0,1,10,11,2,…)"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn stack_rejects_a_short_group() {
        let dir = scratch("short");
        // Declare 4 members but ship only 3.
        write_st(
            &dir.join("model.safetensors"),
            &[
                ("e.0", "F32", vec![1], f32s(&[0.0])),
                ("e.1", "F32", vec![1], f32s(&[1.0])),
                ("e.2", "F32", vec![1], f32s(&[2.0])),
            ],
        );
        let err = fuse_checkpoint(&dir, &|name| {
            let idx: usize = name.strip_prefix("e.")?.parse().ok()?;
            Some(Fuse::Stack {
                target: "e".into(),
                index: idx,
                count: 4,
            })
        })
        .unwrap_err();
        assert!(
            matches!(&err, SafetensorsError::BadGroup { reason, .. }
                if reason.contains("3 of 4 members")),
            "got {err}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn stack_rejects_duplicate_members_and_layout_disagreement() {
        let dir = scratch("dup");
        write_st(
            &dir.join("model.safetensors"),
            &[
                ("a", "F32", vec![1], f32s(&[0.0])),
                ("b", "F32", vec![1], f32s(&[1.0])),
            ],
        );
        // Both map to member 0 of the same target.
        let err = fuse_checkpoint(&dir, &|_| {
            Some(Fuse::Stack {
                target: "t".into(),
                index: 0,
                count: 2,
            })
        })
        .unwrap_err();
        assert!(
            matches!(&err, SafetensorsError::BadGroup { reason, .. }
                if reason.contains("duplicate member 0")),
            "got {err}"
        );

        // Same group, different shapes.
        let dir2 = scratch("layout");
        write_st(
            &dir2.join("model.safetensors"),
            &[
                ("a", "F32", vec![1], f32s(&[0.0])),
                ("b", "F32", vec![2], f32s(&[1.0, 2.0])),
            ],
        );
        let err = fuse_checkpoint(&dir2, &|name| {
            Some(Fuse::Stack {
                target: "t".into(),
                index: usize::from(name == "b"),
                count: 2,
            })
        })
        .unwrap_err();
        assert!(
            matches!(&err, SafetensorsError::BadGroup { reason, .. }
                if reason.contains("disagree on layout")),
            "got {err}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
        std::fs::remove_dir_all(&dir2).unwrap();
    }

    #[test]
    fn unmapped_tensor_is_a_loud_error_and_drop_is_explicit() {
        let dir = scratch("unmapped");
        write_st(
            &dir.join("model.safetensors"),
            &[
                ("keep", "F32", vec![1], f32s(&[1.0])),
                ("junk", "F32", vec![1], f32s(&[2.0])),
            ],
        );
        // No mapping for "junk" -> loud.
        let err = fuse_checkpoint(&dir, &|name| {
            (name == "keep").then(|| Fuse::Keep("keep".into()))
        })
        .unwrap_err();
        assert!(
            matches!(&err, SafetensorsError::BadTensor { reason, .. }
                if reason.contains("unmapped tensor name")),
            "got {err}"
        );
        // Explicitly dropping it is fine, and it leaves the blob.
        let blob = fuse_checkpoint(&dir, &|name| {
            Some(if name == "keep" {
                Fuse::Keep("keep".into())
            } else {
                Fuse::Drop
            })
        })
        .unwrap();
        let (_, shape, values) = read_back(&blob, "keep");
        assert_eq!((shape, values), (vec![1], vec![1.0]));
        let header_len = u64::from_le_bytes(blob[..8].try_into().unwrap()) as usize;
        let header = std::str::from_utf8(&blob[8..8 + header_len]).unwrap();
        assert!(!header.contains("junk"), "dropped tensor is absent");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A group whose members live in DIFFERENT shards still fuses correctly —
    /// a real 3-shard checkpoint can split a layer across a boundary.
    #[test]
    fn shards_are_discovered_and_fused_across_boundaries() {
        let dir = scratch("shards");
        write_st(
            &dir.join("model-00001-of-00002.safetensors"),
            &[("e.1", "F32", vec![1], f32s(&[11.0]))],
        );
        write_st(
            &dir.join("model-00002-of-00002.safetensors"),
            &[("e.0", "F32", vec![1], f32s(&[10.0]))],
        );
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            br#"{"weight_map":{"e.1":"model-00001-of-00002.safetensors",
                              "e.0":"model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();

        let shards = checkpoint_shards(&dir).unwrap();
        assert_eq!(shards.len(), 2, "both shards discovered from the index");

        let blob = fuse_checkpoint(&dir, &|name| {
            let idx: usize = name.strip_prefix("e.")?.parse().ok()?;
            Some(Fuse::Stack {
                target: "e".into(),
                index: idx,
                count: 2,
            })
        })
        .unwrap();
        let (_, shape, values) = read_back(&blob, "e");
        assert_eq!(shape, vec![2, 1]);
        // Member 0 came from shard 2, member 1 from shard 1 — order is by
        // member index, never by shard order.
        assert_eq!(values, vec![10.0, 11.0]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_dir_with_no_checkpoint_is_a_loud_error() {
        let dir = scratch("empty");
        assert!(matches!(
            checkpoint_shards(&dir),
            Err(SafetensorsError::NoCheckpoint(_))
        ));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// An index is a manifest, not evidence. A half-fetched checkpoint must
    /// fail HERE — before any shard is opened — or it surfaces as a bare
    /// `os error 2` from inside the load, and any caller using
    /// `checkpoint_shards(..).is_ok()` as "is this checkpoint complete?"
    /// silently believes an interrupted download is ready to use.
    #[test]
    fn an_index_naming_a_missing_shard_is_a_loud_error_not_a_late_os_error() {
        let dir = scratch("missing_shard");
        write_st(
            &dir.join("model-00001-of-00002.safetensors"),
            &[("e.0", "F32", vec![1], f32s(&[10.0]))],
        );
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            br#"{"weight_map":{"e.0":"model-00001-of-00002.safetensors",
                               "e.1":"model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();

        // Shard 2 is absent entirely: named, no hint.
        match checkpoint_shards(&dir) {
            Err(SafetensorsError::MissingShard { missing, hint, .. }) => {
                assert_eq!(missing, "model-00002-of-00002.safetensors");
                assert!(
                    hint.is_empty(),
                    "no .part sibling, so no interrupted-download hint"
                );
            }
            other => panic!("expected MissingShard, got {other:?}"),
        }

        // The same shard as an interrupted download: the error says so, which
        // is the difference between "re-fetch me" and "this repo is broken".
        std::fs::write(
            dir.join("model-00002-of-00002.safetensors.part"),
            b"partial",
        )
        .unwrap();
        match checkpoint_shards(&dir) {
            Err(SafetensorsError::MissingShard { hint, .. }) => {
                assert!(
                    hint.contains("interrupted"),
                    "hint names the .part sibling: {hint:?}"
                );
            }
            other => panic!("expected MissingShard, got {other:?}"),
        }

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The streaming fuse and the in-memory fuse must be the SAME bytes —
    /// otherwise "load a big model from a file" and "load a small one from
    /// RAM" are two different importers, and only one of them is tested.
    #[test]
    fn fusing_to_a_file_is_byte_identical_to_fusing_in_memory() {
        let dir = scratch("fuse_to_file");
        write_st(
            &dir.join("model-00001-of-00002.safetensors"),
            &[
                ("e.1", "F32", vec![2], f32s(&[11.0, 12.0])),
                ("keep", "F32", vec![1], f32s(&[7.0])),
            ],
        );
        write_st(
            &dir.join("model-00002-of-00002.safetensors"),
            &[("e.0", "F32", vec![2], f32s(&[10.0, 9.0]))],
        );
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            br#"{"weight_map":{"e.1":"model-00001-of-00002.safetensors",
                               "keep":"model-00001-of-00002.safetensors",
                               "e.0":"model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();

        let map = |name: &str| -> Option<Fuse> {
            if name == "keep" {
                return Some(Fuse::Keep("keep".into()));
            }
            let idx: usize = name.strip_prefix("e.")?.parse().ok()?;
            Some(Fuse::Stack {
                target: "e".into(),
                index: idx,
                count: 2,
            })
        };

        let in_memory = fuse_checkpoint(&dir, &map).unwrap();
        let out = dir.join("fused.safetensors");
        let written = fuse_checkpoint_to_file(&dir, &map, &out).unwrap();
        let on_disk = std::fs::read(&out).unwrap();

        assert_eq!(
            in_memory, on_disk,
            "the two fuse paths must agree byte for byte"
        );
        // The return value counts PAYLOAD bytes, so the file is that plus the
        // 8-byte length prefix and the JSON header.
        assert!(
            written < on_disk.len() as u64,
            "payload {written} B sits inside a {} B file",
            on_disk.len()
        );

        // And the fused content is still correct, not merely self-consistent.
        let (_, shape, values) = read_back(&on_disk, "e");
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(values, vec![10.0, 9.0, 11.0, 12.0], "numeric member order");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn dtype_bytes_covers_the_float_and_int_tokens() {
        assert_eq!(dtype_bytes("BF16"), Some(2));
        assert_eq!(dtype_bytes("F32"), Some(4));
        assert_eq!(dtype_bytes("I64"), Some(8));
        assert_eq!(dtype_bytes("BOOL"), Some(1));
        assert_eq!(dtype_bytes("NOPE"), None);
    }
}
