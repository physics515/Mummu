//! The **autotune cache**: where CubeCL persists its kernel picks, and how to
//! throw them away.
//!
//! CubeCL benchmarks several implementations of each kernel the first time it
//! sees one and writes the winner to disk, keyed by (device, kernel,
//! checksum). Later processes load those picks instead of re-tuning, which is
//! what makes a cold start bearable — but the cache has **no invalidation and
//! no re-tune trigger**, so a pick made while the machine was busy is
//! indistinguishable from a good one and is believed forever. Measured on
//! 2026-08-09 (see `bench/BASELINE.md`): a tune that happened during a
//! contended moment cost **21–27 % of f16 decode throughput** in every
//! subsequent process, while the f32 picks from the same moment were
//! unaffected — so the symptom is silent, partial, and permanent.
//!
//! This module is the repair: report where the cache lives and delete it, so a
//! consumer can offer a "re-tune GPU kernels" action instead of shipping a bad
//! tune to a user forever. It reads the same configuration CubeCL reads
//! (`[cubecl.autotune] cache` from the `cubecl.toml` / `burn.toml` discovered
//! by walking up from the process CWD), so the path is right by construction
//! rather than by convention.

use std::path::{Path, PathBuf};

use cubecl_runtime::config::{CubeClRuntimeConfig, RuntimeConfig};

/// Directory segment CubeCL appends to the configured cache root for autotune
/// data (`CacheOption::name("autotune")`). Load-bearing for safety: every path
/// this module deletes ends in this segment, so a misconfigured root can never
/// turn a clear into a wider delete.
const AUTOTUNE_SEGMENT: &str = "autotune";

/// Bound on a cache walk. The layout is
/// `<root>/autotune/<version>/<device>/<kernel>.json.log` — a few hundred
/// files on a normal machine. A runaway count means the root is pointing
/// somewhere it should not, and is an error rather than a long walk.
const MAX_CACHE_FILES: usize = 65_536;
/// Bound on recursion depth for the same reason (the real layout is 3 deep).
const MAX_CACHE_DEPTH: usize = 8;

/// What went wrong inspecting or clearing the cache.
#[derive(Debug, thiserror::Error)]
pub enum TuneError {
    /// The cache directory could not be read or removed.
    #[error("autotune cache i/o at {path}: {message}")]
    Io {
        /// The path being read or removed.
        path: PathBuf,
        /// The underlying OS error.
        message: String,
    },
    /// The cache tree is larger or deeper than any real autotune cache, which
    /// means the configured root is not what we think it is. Refused rather
    /// than walked (or deleted).
    #[error("autotune cache at {path} is implausible ({what}) — refusing to touch it")]
    Implausible {
        /// The configured cache directory.
        path: PathBuf,
        /// Which bound was exceeded.
        what: String,
    },
}

/// Where the autotune cache lives, and how much of it there is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuneCacheReport {
    /// The `<root>/autotune` directory CubeCL writes to.
    pub dir: PathBuf,
    /// Number of cache files found (0 when the cache does not exist yet).
    pub files: usize,
    /// Total bytes of those files.
    pub bytes: u64,
}

impl TuneCacheReport {
    /// Has anything been tuned and persisted yet?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.files == 0
    }
}

/// The directory CubeCL persists autotune picks to, per the configuration it
/// would itself discover.
///
/// **Reads the global config**, which initializes it if no one has yet — the
/// same one-shot singleton `RuntimeConfig::set` writes to. A consumer that
/// wants to `set` a custom config must do so *before* calling this (and before
/// building any backend), exactly as CubeCL requires.
#[must_use]
pub fn autotune_cache_dir() -> PathBuf {
    let dir = CubeClRuntimeConfig::get()
        .autotune
        .cache
        .root()
        .join(AUTOTUNE_SEGMENT);
    debug_assert!(
        dir.ends_with(AUTOTUNE_SEGMENT),
        "the cache dir must always end in the autotune segment"
    );
    dir
}

/// Measure the persisted cache without changing it. A missing directory is
/// not an error — it means nothing has been tuned yet.
pub fn autotune_cache_report() -> Result<TuneCacheReport, TuneError> {
    let dir = autotune_cache_dir();
    let (files, bytes) = measure(&dir, 0)?;
    Ok(TuneCacheReport { dir, files, bytes })
}

/// Delete the persisted autotune cache, returning what was removed.
///
/// The next process to run will re-tune from scratch and write fresh picks —
/// **the next one**, not this one: a running process has already loaded the
/// cache into memory and will keep using and re-writing it, so a consumer
/// should treat this as "re-tune on next launch" (or call it before building a
/// backend). Idempotent: clearing an absent cache reports zero and succeeds.
pub fn clear_autotune_cache() -> Result<TuneCacheReport, TuneError> {
    let report = autotune_cache_report()?;
    assert!(
        report.dir.ends_with(AUTOTUNE_SEGMENT),
        "refusing to remove a directory that is not the autotune cache: {:?}",
        report.dir
    );
    if !report.dir.exists() {
        debug_assert!(report.is_empty(), "an absent cache cannot hold files");
        return Ok(report);
    }
    std::fs::remove_dir_all(&report.dir).map_err(|e| TuneError::Io {
        path: report.dir.clone(),
        message: e.to_string(),
    })?;
    Ok(report)
}

/// Bounded recursive walk: `(file count, total bytes)` under `dir`.
fn measure(dir: &Path, depth: usize) -> Result<(usize, u64), TuneError> {
    if depth > MAX_CACHE_DEPTH {
        return Err(TuneError::Implausible {
            path: dir.to_path_buf(),
            what: format!("deeper than {MAX_CACHE_DEPTH} levels"),
        });
    }
    if !dir.is_dir() {
        return Ok((0, 0));
    }
    let entries = std::fs::read_dir(dir).map_err(|e| TuneError::Io {
        path: dir.to_path_buf(),
        message: e.to_string(),
    })?;

    let (mut files, mut bytes) = (0usize, 0u64);
    for entry in entries {
        let entry = entry.map_err(|e| TuneError::Io {
            path: dir.to_path_buf(),
            message: e.to_string(),
        })?;
        let path = entry.path();
        if path.is_dir() {
            let (f, b) = measure(&path, depth + 1)?;
            files += f;
            bytes += b;
        } else {
            files += 1;
            bytes += entry.metadata().map(|m| m.len()).unwrap_or(0);
        }
        if files > MAX_CACHE_FILES {
            return Err(TuneError::Implausible {
                path: dir.to_path_buf(),
                what: format!("more than {MAX_CACHE_FILES} files"),
            });
        }
    }
    Ok((files, bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cache_dir_always_ends_in_the_autotune_segment() {
        let dir = autotune_cache_dir();
        assert!(
            dir.ends_with(AUTOTUNE_SEGMENT),
            "cache dir {dir:?} must end in {AUTOTUNE_SEGMENT}"
        );
        // And it must be an absolute path — the roots CubeCL can resolve to
        // (CWD, the project target dir, the user config dir, or an explicit
        // file path) are all absolute in practice, and a relative one would
        // make "clear the cache" depend on the caller's CWD.
        assert!(dir.is_absolute(), "cache dir {dir:?} must be absolute");
    }

    #[test]
    fn measuring_an_absent_directory_reports_empty() {
        let missing = std::env::temp_dir().join("mummu-no-such-autotune-dir-9e3f");
        assert!(!missing.exists(), "fixture path must not exist");
        assert_eq!(
            measure(&missing, 0).expect("absent is not an error"),
            (0, 0)
        );
    }

    #[test]
    fn measuring_counts_files_and_bytes_across_nested_dirs() {
        let root = std::env::temp_dir().join("mummu-tune-measure-a41c");
        let nested = root.join("0.10.0").join("device-4-0");
        std::fs::create_dir_all(&nested).expect("fixture dirs");
        std::fs::write(nested.join("matmul.json.log"), b"12345").expect("fixture file");
        std::fs::write(nested.join("reduce.json.log"), b"678").expect("fixture file");

        let (files, bytes) = measure(&root, 0).expect("walks");
        assert_eq!(files, 2, "both nested files counted");
        assert_eq!(bytes, 8, "byte totals summed across directories");

        std::fs::remove_dir_all(&root).expect("fixture cleanup");
    }

    #[test]
    fn measuring_refuses_an_implausibly_deep_tree() {
        // Depth is checked before the directory is read, so a synthetic path
        // is enough — no need to build a 9-level fixture.
        let deep = std::env::temp_dir().join("mummu-tune-depth");
        let err = measure(&deep, MAX_CACHE_DEPTH + 1).expect_err("too deep is an error");
        assert!(
            matches!(err, TuneError::Implausible { .. }),
            "expected Implausible, got {err:?}"
        );
    }

    #[test]
    fn clearing_an_absent_cache_is_a_successful_no_op() {
        // `clear` on a machine that has never tuned must not error; the real
        // dir may or may not exist here, so assert on the shape of the result.
        let before = autotune_cache_report().expect("report");
        assert!(before.dir.ends_with(AUTOTUNE_SEGMENT));
        if !before.dir.exists() {
            let cleared = clear_autotune_cache().expect("clearing nothing succeeds");
            assert!(cleared.is_empty(), "nothing to clear reports empty");
        }
    }
}
