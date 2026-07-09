//! Model-cache disk accounting (P8): report how much space each cached model
//! takes and validate a removal target so a consumer's settings UI can
//! reclaim disk without ever escaping the cache dir. App-agnostic and free of
//! async/UI types; ported from laurelane's unit-tested implementation.

use std::path::{Component, Path, PathBuf};

/// One cached model directory and its size on disk.
#[derive(serde::Serialize, Debug, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ModelDisk {
    /// Cache subdir name, e.g. `"qwen2.5-1.5b"`.
    pub name: String,
    pub bytes: u64,
}

/// A full disk report for a model cache dir.
#[derive(serde::Serialize, Debug, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct DiskReport {
    /// Cached models, largest first.
    pub models: Vec<ModelDisk>,
    pub total_bytes: u64,
    /// The cache dir being reported — the "your models live here" line.
    pub location: String,
}

/// Recursively sum the sizes of every regular file under `path`. Best-effort:
/// unreadable entries are skipped and symlinked dirs are not followed (only
/// real directories recurse), so a broken link can't send it off the rails.
#[must_use]
pub fn dir_size(path: &Path) -> u64 {
    let mut total = 0;
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    for entry in entries.flatten() {
        let Ok(ft) = entry.file_type() else { continue };
        if ft.is_dir() {
            total += dir_size(&entry.path());
        } else if ft.is_file()
            && let Ok(meta) = entry.metadata()
        {
            total += meta.len();
        }
    }
    total
}

/// Immediate subdirs of `models_dir`, each with its recursive size, sorted
/// largest-first (ties broken by name). A missing dir is an empty list, not
/// an error — a fresh install has no downloaded models yet.
#[must_use]
pub fn cached_models(models_dir: &Path) -> Vec<ModelDisk> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(models_dir) else {
        return out;
    };
    for entry in entries.flatten() {
        if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            let name = entry.file_name().to_string_lossy().into_owned();
            out.push(ModelDisk {
                name,
                bytes: dir_size(&entry.path()),
            });
        }
    }
    out.sort_by(|a, b| b.bytes.cmp(&a.bytes).then_with(|| a.name.cmp(&b.name)));
    out
}

/// A full report: cached models + their combined size + the cache location.
#[must_use]
pub fn report(models_dir: &Path) -> DiskReport {
    let models = cached_models(models_dir);
    let total_bytes = models.iter().map(|m| m.bytes).sum();
    debug_assert!(
        models.windows(2).all(|w| w[0].bytes >= w[1].bytes),
        "cached_models must sort largest-first"
    );
    DiskReport {
        models,
        total_bytes,
        location: models_dir.to_string_lossy().into_owned(),
    }
}

/// Is `name` a safe single cache-subdir component? Rejects empty, `.`/`..`,
/// anything containing a path separator, and any rooted/prefixed path — so a
/// removal can never climb out of the cache dir.
#[must_use]
pub fn is_safe_component(name: &str) -> bool {
    if name.is_empty() || name == "." || name == ".." {
        return false;
    }
    if name.contains('/') || name.contains('\\') {
        return false;
    }
    let mut parts = Path::new(name).components();
    matches!(parts.next(), Some(Component::Normal(_))) && parts.next().is_none()
}

/// Resolve the on-disk path to remove for cache subdir `name`, or an error if
/// `name` is unsafe or absent. The returned path is always a direct child of
/// `models_dir`.
pub fn resolve_removal(models_dir: &Path, name: &str) -> Result<PathBuf, String> {
    if !is_safe_component(name) {
        return Err(format!("unsafe model name: {name:?}"));
    }
    let target = models_dir.join(name);
    if !target.is_dir() {
        return Err(format!("no cached model named {name:?}"));
    }
    debug_assert!(
        target.starts_with(models_dir),
        "removal must stay inside the cache"
    );
    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    /// A unique temp dir for one test, wiped fresh. Clock-free (uses the
    /// test's own tag), so parallel tests never collide.
    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("mummu_manage_test_{tag}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write(path: &Path, bytes: usize) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, vec![b'x'; bytes]).unwrap();
    }

    #[test]
    fn dir_size_sums_nested_files() {
        let root = scratch("dirsize");
        write(&root.join("a.bin"), 100);
        write(&root.join("sub/b.bin"), 250);
        write(&root.join("sub/deep/c.bin"), 50);
        assert_eq!(dir_size(&root), 400);
        // A missing dir is zero, not a panic.
        assert_eq!(dir_size(&root.join("nope")), 0);
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn cached_models_lists_subdirs_largest_first_ignoring_loose_files() {
        let root = scratch("cached");
        write(&root.join("qwen2.5-1.5b/model.safetensors"), 3000);
        write(&root.join("all-minilm/model.safetensors"), 90);
        write(&root.join("lfm2.5/model.safetensors"), 500);
        write(&root.join("loose.txt"), 10_000); // not a dir → ignored
        let got = cached_models(&root);
        assert_eq!(
            got,
            vec![
                ModelDisk {
                    name: "qwen2.5-1.5b".into(),
                    bytes: 3000
                },
                ModelDisk {
                    name: "lfm2.5".into(),
                    bytes: 500
                },
                ModelDisk {
                    name: "all-minilm".into(),
                    bytes: 90
                },
            ],
        );
        assert_eq!(report(&root).total_bytes, 3590);
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn cached_models_of_missing_dir_is_empty() {
        let missing = std::env::temp_dir().join("mummu_manage_test_absent_xyz");
        let _ = fs::remove_dir_all(&missing);
        assert!(cached_models(&missing).is_empty());
        assert_eq!(report(&missing).total_bytes, 0);
    }

    #[test]
    fn is_safe_component_accepts_names_and_rejects_traversal() {
        assert!(is_safe_component("qwen2.5-1.5b"));
        assert!(is_safe_component("all-minilm"));
        for bad in [
            "",
            ".",
            "..",
            "a/b",
            "a\\b",
            "/etc",
            "..\\..\\x",
            "C:\\x",
            "sub/",
        ] {
            assert!(!is_safe_component(bad), "should reject {bad:?}");
        }
    }

    #[test]
    fn resolve_removal_rejects_unsafe_and_absent_but_finds_present() {
        let root = scratch("removal");
        write(&root.join("qwen2.5-0.5b/model.safetensors"), 10);
        assert_eq!(
            resolve_removal(&root, "qwen2.5-0.5b").unwrap(),
            root.join("qwen2.5-0.5b")
        );
        assert!(resolve_removal(&root, "not-there").is_err());
        assert!(resolve_removal(&root, "..").is_err());
        assert!(resolve_removal(&root, "../models").is_err());
        fs::remove_dir_all(&root).unwrap();
    }
}
