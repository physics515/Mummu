//! Model downloads: HuggingFace Hub (or any HTTP host) → the local model
//! cache. Streaming, **resumable** (a `<file>.part` picks up where a killed
//! download stopped, via HTTP `Range`), **integrity-checked** (streamed
//! sha256 against the Hub's announced LFS `X-Linked-ETag`, length as the
//! fallback), and **sharded-checkpoint aware** (`model.safetensors.index.json`
//! → fetch every shard). Progress surfaces through a callback so app settings
//! UIs can show it (P8). Completed files are cache-first: an existing
//! destination is never re-fetched unless [`FetchOptions::verify_cached`]
//! asks for a re-hash.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

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
    #[error("{url}: sha256 mismatch — announced {expected}, computed {computed}")]
    Corrupt {
        url: String,
        expected: String,
        computed: String,
    },
}

/// Options for [`fetch_file_with`] / [`fetch_model_with`].
#[derive(Debug, Clone, Copy, Default)]
pub struct FetchOptions {
    /// Re-hash an already-complete destination against the server's announced
    /// sha256 (one extra HEAD request per file); on a mismatch the corrupt
    /// copy is deleted and re-fetched once. Off by default: a completed file
    /// was already verified as it streamed in.
    pub verify_cached: bool,
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

/// Lowercase hex of a digest.
fn hex64(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    debug_assert_eq!(bytes.len(), 32, "sha256 digests are 32 bytes");
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// A quoted 64-hex etag value is a content sha256 (the Hub's `X-Linked-ETag`
/// for LFS files). Git-style etags (40-hex sha1, or non-hex) parse to `None` —
/// they name a revision, not the bytes, and cannot verify a stream.
fn parse_sha256_etag(raw: &str) -> Option<String> {
    let v = raw.trim().trim_start_matches("W/").trim_matches('"');
    let is_sha256 = v.len() == 64 && v.bytes().all(|b| b.is_ascii_hexdigit());
    is_sha256.then(|| v.to_ascii_lowercase())
}

/// Ask the server for the file's content sha256: a redirect-stopped HEAD reads
/// the Hub's `X-Linked-ETag` from the `resolve/` endpoint itself, before the
/// CDN handoff would replace the headers. `Ok(None)` when nothing usable is
/// announced (non-LFS files, other hosts, HEAD rejected) — those downloads
/// stay length-verified only. Transport failures are loud: the GET would fail
/// the same way.
fn announced_sha256(url: &str) -> Result<Option<String>, HubError> {
    assert!(url.starts_with("https://"), "refusing non-https url: {url}");
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .max_redirects(0)
        .http_status_as_error(false)
        .build()
        .into();
    let resp = agent.head(url).call().map_err(|e| HubError::Http {
        url: url.into(),
        reason: e.to_string(),
    })?;
    Ok(["x-linked-etag", "etag"].iter().find_map(|h| {
        resp.headers()
            .get(*h)
            .and_then(|v| v.to_str().ok())
            .and_then(parse_sha256_etag)
    }))
}

/// Streaming sha256 of a file on disk, as lowercase hex.
fn sha256_hex_of_file(path: &Path) -> Result<String, HubError> {
    let io_err = |e: std::io::Error| HubError::Io {
        path: path.to_path_buf(),
        reason: e.to_string(),
    };
    let mut f = std::fs::File::open(path).map_err(io_err)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; CHUNK_BYTES];
    let mut hashed = 0u64;
    loop {
        let n = f.read(&mut buf).map_err(io_err)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        hashed += n as u64;
        assert!(
            hashed <= MAX_FILE_BYTES,
            "{path:?}: exceeds the file bound while hashing"
        );
    }
    Ok(hex64(&hasher.finalize()))
}

/// Feed the already-downloaded `.part` prefix into the stream hasher so a
/// resumed download still verifies as one whole file.
fn hash_part_prefix(part: &Path, resume_from: u64, hasher: &mut Sha256) -> Result<(), HubError> {
    assert!(resume_from > 0, "no prefix to hash");
    let mut f = std::fs::File::open(part).map_err(|e| HubError::Io {
        path: part.to_path_buf(),
        reason: e.to_string(),
    })?;
    let mut remaining = resume_from;
    let mut buf = vec![0u8; CHUNK_BYTES];
    while remaining > 0 {
        let want = remaining.min(CHUNK_BYTES as u64) as usize;
        let n = f.read(&mut buf[..want]).map_err(|e| HubError::Io {
            path: part.to_path_buf(),
            reason: e.to_string(),
        })?;
        // The prefix length came from this file's own metadata an instant ago.
        assert!(n > 0, "{part:?}: prefix ended {remaining} bytes early");
        hasher.update(&buf[..n]);
        remaining -= n as u64;
    }
    Ok(())
}

/// Fetch `url` into `dest`, streaming through `<dest>.part` and resuming any
/// earlier partial download. No-op when `dest` already exists (cache-first).
/// `on_progress` fires after every chunk with cumulative counts. The stream
/// is verified against the server's announced sha256 when there is one
/// (Hub LFS files), else by length.
pub fn fetch_file(
    url: &str,
    dest: &Path,
    on_progress: impl FnMut(Progress<'_>),
) -> Result<(), HubError> {
    fetch_file_with(url, dest, FetchOptions::default(), on_progress)
}

/// [`fetch_file`] with explicit [`FetchOptions`]. With `verify_cached`, an
/// existing `dest` is re-hashed against the announced sha256; a mismatch
/// deletes the corrupt copy (and any stale `.part` that would poison a
/// resume) and re-fetches once — self-healing, never silent.
pub fn fetch_file_with(
    url: &str,
    dest: &Path,
    opts: FetchOptions,
    mut on_progress: impl FnMut(Progress<'_>),
) -> Result<(), HubError> {
    assert!(url.starts_with("https://"), "refusing non-https url: {url}");
    if dest.exists() {
        if !opts.verify_cached {
            return Ok(()); // cache hit — never re-fetch a completed file
        }
        let Some(expected) = announced_sha256(url)? else {
            return Ok(()); // nothing announced — nothing to re-verify against
        };
        if sha256_hex_of_file(dest)? == expected {
            return Ok(());
        }
        for stale in [dest.to_path_buf(), part_path(dest)] {
            if stale.exists() {
                std::fs::remove_file(&stale).map_err(|e| HubError::Io {
                    path: stale.clone(),
                    reason: e.to_string(),
                })?;
            }
        }
    }
    download(url, dest, &mut on_progress)
}

/// The streaming GET behind [`fetch_file_with`]: resume, hash, length-check,
/// then atomically rename `.part` → `dest`.
fn download(
    url: &str,
    dest: &Path,
    on_progress: &mut impl FnMut(Progress<'_>),
) -> Result<(), HubError> {
    debug_assert!(!dest.exists(), "download() requires a vacant destination");
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
    // One cheap HEAD up front: with an announced sha256 the whole stream
    // (resumed prefix included) is verified; without one, length still is.
    let expected_sha = announced_sha256(url)?;

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

    let mut hasher = expected_sha.as_ref().map(|_| Sha256::new());
    if resumed && let Some(h) = hasher.as_mut() {
        hash_part_prefix(&part, resume_from, h)?;
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
        if let Some(h) = hasher.as_mut() {
            h.update(&buf[..n]);
        }
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
    if let (Some(expected), Some(h)) = (expected_sha, hasher) {
        let computed = hex64(&h.finalize());
        if computed != expected {
            // A wrong-hash .part must never seed a resume — drop it.
            std::fs::remove_file(&part).map_err(|e| HubError::Io {
                path: part.clone(),
                reason: e.to_string(),
            })?;
            return Err(HubError::Corrupt {
                url: url.into(),
                expected,
                computed,
            });
        }
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
    on_progress: impl FnMut(Progress<'_>),
) -> Result<PathBuf, HubError> {
    fetch_model_with(
        repo,
        revision,
        dest_dir,
        FetchOptions::default(),
        on_progress,
    )
}

/// [`fetch_model`] with explicit [`FetchOptions`] (e.g. re-verify cached
/// files' sha256 before trusting them).
pub fn fetch_model_with(
    repo: &str,
    revision: &str,
    dest_dir: &Path,
    opts: FetchOptions,
    mut on_progress: impl FnMut(Progress<'_>),
) -> Result<PathBuf, HubError> {
    for file in ["config.json", "tokenizer.json"] {
        fetch_file_with(
            &hub_file_url(repo, revision, file),
            &dest_dir.join(file),
            opts,
            &mut on_progress,
        )?;
    }
    // Single-file first (the common case for the small-model tiers we target).
    let single = fetch_file_with(
        &hub_file_url(repo, revision, "model.safetensors"),
        &dest_dir.join("model.safetensors"),
        opts,
        &mut on_progress,
    );
    if single.is_ok() {
        return Ok(dest_dir.to_path_buf());
    }
    // Fall back to a sharded checkpoint; if there's no index either, report
    // the original single-file error (the more useful signal).
    let index_name = "model.safetensors.index.json";
    let index_dest = dest_dir.join(index_name);
    if fetch_file_with(
        &hub_file_url(repo, revision, index_name),
        &index_dest,
        opts,
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
        fetch_file_with(
            &hub_file_url(repo, revision, &shard),
            &dest_dir.join(&shard),
            opts,
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
    fn sha256_etag_parsing_accepts_only_content_hashes() {
        let sha = "a".repeat(64);
        // The Hub quotes LFS etags; weak etags carry a W/ prefix.
        assert_eq!(parse_sha256_etag(&format!("\"{sha}\"")), Some(sha.clone()));
        assert_eq!(
            parse_sha256_etag(&format!("W/\"{sha}\"")),
            Some(sha.clone())
        );
        assert_eq!(parse_sha256_etag(&sha.to_uppercase()), Some(sha));
        // Git-style 40-hex sha1, non-hex, and empty values are not sha256s.
        assert_eq!(parse_sha256_etag(&format!("\"{}\"", "b".repeat(40))), None);
        assert_eq!(parse_sha256_etag(&format!("\"{}\"", "z".repeat(64))), None);
        assert_eq!(parse_sha256_etag(""), None);
    }

    #[test]
    fn file_hash_matches_the_reference_vector() {
        // FIPS 180-2 test vector: sha256("abc").
        let dir = std::env::temp_dir().join("mummu-hub-test-sha");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("abc.txt");
        std::fs::write(&p, b"abc").unwrap();
        assert_eq!(
            sha256_hex_of_file(&p).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn prefix_plus_remainder_hash_equals_whole_file_hash() {
        // The resume path hashes the .part prefix, then the streamed tail;
        // together they must equal one pass over the whole file.
        let dir = std::env::temp_dir().join("mummu-hub-test-prefix");
        std::fs::create_dir_all(&dir).unwrap();
        let whole: Vec<u8> = (0u32..200_000).map(|i| (i % 251) as u8).collect();
        let p = dir.join("whole.bin");
        std::fs::write(&p, &whole).unwrap();
        let reference = sha256_hex_of_file(&p).unwrap();

        let split = whole.len() / 3;
        let part = dir.join("whole.bin.part");
        std::fs::write(&part, &whole[..split]).unwrap();
        let mut h = Sha256::new();
        hash_part_prefix(&part, split as u64, &mut h).unwrap();
        h.update(&whole[split..]);
        assert_eq!(hex64(&h.finalize()), reference);
        std::fs::remove_dir_all(&dir).ok();
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
