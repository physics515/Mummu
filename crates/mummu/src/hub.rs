//! Model downloads: HuggingFace Hub (or any HTTP host) → the local model
//! cache. Streaming, **resumable** (a `<file>.part` picks up where a killed
//! download stopped, via HTTP `Range`), length-verified, and
//! **sharded-checkpoint aware** (`model.safetensors.index.json` → fetch every
//! shard). Progress surfaces through a callback so app settings UIs can show
//! it (P8). Completed files are cache-first: an existing destination is never
//! re-fetched.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Streaming copy granularity: big enough to amortize syscalls, small enough
/// to keep progress callbacks responsive.
const CHUNK_BYTES: usize = 64 * 1024;

/// Hard per-file ceiling — larger than any model shard we'd fetch (shards are
/// conventionally ≤ ~10 GB); anything past this is a wiring bug or a hostile
/// server, not a model.
const MAX_FILE_BYTES: u64 = 64 << 30;

/// Ceiling on shards in an index — the largest public checkpoints ship tens.
const MAX_SHARDS: usize = 512;

/// Everything that can go wrong fetching a model.
#[derive(Debug, thiserror::Error)]
pub enum HubError {
    #[error("http {url}: {reason}")]
    Http { url: String, reason: String },
    #[error("io {path}: {reason}")]
    Io { path: PathBuf, reason: String },
    #[error("{url}: expected {expected} bytes, received {received}")]
    Incomplete {
        url: String,
        expected: u64,
        received: u64,
    },
    #[error("shard index {path}: {reason}")]
    BadIndex { path: PathBuf, reason: String },
}

/// Download progress for one file, reported after every chunk.
#[derive(Debug, Clone)]
pub struct Progress<'a> {
    pub file: &'a str,
    pub received_bytes: u64,
    /// Total including any resumed prefix; `None` when the server omits it.
    pub total_bytes: Option<u64>,
}

/// `https://huggingface.co/{repo}/resolve/{revision}/{file}` — the Hub's
/// stable raw-file endpoint.
#[must_use]
pub fn hub_file_url(repo: &str, revision: &str, file: &str) -> String {
    assert!(
        !repo.is_empty() && repo.contains('/'),
        "repo must be owner/name, got {repo:?}"
    );
    assert!(!revision.is_empty(), "revision must be non-empty");
    format!("https://huggingface.co/{repo}/resolve/{revision}/{file}")
}

/// The unique shard files referenced by a `*.index.json` (weight_map values,
/// deduped, sorted for a deterministic fetch order).
pub fn shards_from_index(index_json: &[u8], index_path: &Path) -> Result<Vec<String>, HubError> {
    let v: serde_json::Value =
        serde_json::from_slice(index_json).map_err(|e| HubError::BadIndex {
            path: index_path.to_path_buf(),
            reason: e.to_string(),
        })?;
    let map = v["weight_map"]
        .as_object()
        .ok_or_else(|| HubError::BadIndex {
            path: index_path.to_path_buf(),
            reason: "no weight_map object".into(),
        })?;
    let mut shards: Vec<String> = map
        .values()
        .filter_map(|s| s.as_str().map(str::to_string))
        .collect();
    shards.sort_unstable();
    shards.dedup();
    if shards.is_empty() || shards.len() > MAX_SHARDS {
        return Err(HubError::BadIndex {
            path: index_path.to_path_buf(),
            reason: format!("{} shards (expected 1..={MAX_SHARDS})", shards.len()),
        });
    }
    Ok(shards)
}

/// The in-flight twin of `dest` (`<dest>.part`).
fn part_path(dest: &Path) -> PathBuf {
    let mut p = dest.as_os_str().to_owned();
    p.push(".part");
    PathBuf::from(p)
}

/// Fetch `url` into `dest`, streaming through `<dest>.part` and resuming any
/// earlier partial download. No-op when `dest` already exists (cache-first).
/// `on_progress` fires after every chunk with cumulative counts.
pub fn fetch_file(
    url: &str,
    dest: &Path,
    mut on_progress: impl FnMut(Progress<'_>),
) -> Result<(), HubError> {
    assert!(url.starts_with("https://"), "refusing non-https url: {url}");
    if dest.exists() {
        return Ok(()); // cache hit — never re-fetch a completed file
    }
    let file_label = dest
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    assert!(!file_label.is_empty(), "dest must name a file: {dest:?}");
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| HubError::Io {
            path: parent.to_path_buf(),
            reason: e.to_string(),
        })?;
    }

    let part = part_path(dest);
    let resume_from = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);

    let mut req = ureq::get(url);
    if resume_from > 0 {
        req = req.header("Range", format!("bytes={resume_from}-"));
    }
    let mut resp = req.call().map_err(|e| HubError::Http {
        url: url.into(),
        reason: e.to_string(),
    })?;
    // A server that ignores Range (200 instead of 206) restarts the body from
    // byte 0 — truncate our part file to match, never splice mismatched halves.
    let resumed = resp.status() == 206 && resume_from > 0;
    let body_len: Option<u64> = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok());
    let already = if resumed { resume_from } else { 0 };
    let total = body_len.map(|l| l + already);
    if let Some(t) = total {
        assert!(
            t <= MAX_FILE_BYTES,
            "{url}: {t} bytes exceeds the file bound"
        );
    }

    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .append(resumed)
        .write(true)
        .truncate(!resumed)
        .open(&part)
        .map_err(|e| HubError::Io {
            path: part.clone(),
            reason: e.to_string(),
        })?;

    let mut reader = resp.body_mut().as_reader();
    let mut received = already;
    let mut buf = vec![0u8; CHUNK_BYTES];
    loop {
        let n = reader.read(&mut buf).map_err(|e| HubError::Http {
            url: url.into(),
            reason: e.to_string(),
        })?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n]).map_err(|e| HubError::Io {
            path: part.clone(),
            reason: e.to_string(),
        })?;
        received += n as u64;
        assert!(
            received <= MAX_FILE_BYTES,
            "{url}: stream exceeded the file bound"
        );
        on_progress(Progress {
            file: &file_label,
            received_bytes: received,
            total_bytes: total,
        });
    }
    drop(out);

    if let Some(expected) = total
        && received != expected
    {
        // Keep the .part for a future resume; report loudly.
        return Err(HubError::Incomplete {
            url: url.into(),
            expected,
            received,
        });
    }
    std::fs::rename(&part, dest).map_err(|e| HubError::Io {
        path: dest.to_path_buf(),
        reason: e.to_string(),
    })
}

/// Fetch a whole model from the Hub into `dest_dir`: `config.json`,
/// `tokenizer.json`, and the weights — `model.safetensors` when the repo is
/// single-file, else every shard listed by `model.safetensors.index.json`.
/// Returns `dest_dir` ready for the per-model `load_from_dir`.
pub fn fetch_model(
    repo: &str,
    revision: &str,
    dest_dir: &Path,
    mut on_progress: impl FnMut(Progress<'_>),
) -> Result<PathBuf, HubError> {
    for file in ["config.json", "tokenizer.json"] {
        fetch_file(
            &hub_file_url(repo, revision, file),
            &dest_dir.join(file),
            &mut on_progress,
        )?;
    }
    // Single-file first (the common case for the small-model tiers we target).
    let single = fetch_file(
        &hub_file_url(repo, revision, "model.safetensors"),
        &dest_dir.join("model.safetensors"),
        &mut on_progress,
    );
    if single.is_ok() {
        return Ok(dest_dir.to_path_buf());
    }
    // Fall back to a sharded checkpoint; if there's no index either, report
    // the original single-file error (the more useful signal).
    let index_name = "model.safetensors.index.json";
    let index_dest = dest_dir.join(index_name);
    if fetch_file(
        &hub_file_url(repo, revision, index_name),
        &index_dest,
        &mut on_progress,
    )
    .is_err()
    {
        return single.map(|()| dest_dir.to_path_buf());
    }
    let index_bytes = std::fs::read(&index_dest).map_err(|e| HubError::Io {
        path: index_dest.clone(),
        reason: e.to_string(),
    })?;
    for shard in shards_from_index(&index_bytes, &index_dest)? {
        fetch_file(
            &hub_file_url(repo, revision, &shard),
            &dest_dir.join(&shard),
            &mut on_progress,
        )?;
    }
    Ok(dest_dir.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hub_url_has_the_resolve_shape() {
        assert_eq!(
            hub_file_url("Qwen/Qwen2.5-1.5B-Instruct", "main", "config.json"),
            "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/config.json"
        );
    }

    #[test]
    #[should_panic(expected = "owner/name")]
    fn bare_repo_names_are_rejected() {
        let _ = hub_file_url("qwen", "main", "config.json");
    }

    #[test]
    fn shard_index_dedupes_and_sorts() {
        let idx = br#"{"metadata":{},"weight_map":{
            "a.weight":"model-00002-of-00002.safetensors",
            "b.weight":"model-00001-of-00002.safetensors",
            "c.weight":"model-00001-of-00002.safetensors"}}"#;
        let shards = shards_from_index(idx, Path::new("x.index.json")).unwrap();
        assert_eq!(
            shards,
            vec![
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors"
            ]
        );
    }

    #[test]
    fn shard_index_without_weight_map_is_rejected() {
        let err = shards_from_index(b"{}", Path::new("x.index.json"));
        assert!(matches!(err, Err(HubError::BadIndex { .. })));
    }

    #[test]
    fn part_path_appends_suffix() {
        assert_eq!(
            part_path(Path::new("m/model.safetensors")),
            Path::new("m/model.safetensors.part")
        );
    }

    #[test]
    fn existing_dest_is_a_cache_hit_without_any_http() {
        // A bogus URL proves no request is made when the file already exists.
        let dir = std::env::temp_dir().join("mummu-hub-test-cache-hit");
        std::fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("present.bin");
        std::fs::write(&dest, b"already here").unwrap();
        let mut calls = 0;
        fetch_file("https://invalid.invalid/x", &dest, |_| calls += 1).unwrap();
        assert_eq!(calls, 0, "cache hit must not stream");
        std::fs::remove_dir_all(&dir).ok();
    }
}
