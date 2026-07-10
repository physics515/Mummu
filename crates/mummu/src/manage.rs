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

/// The settings-UI-facing management surface (P8): one object owning the
/// models root that composes the catalog ([`crate::registry`]), downloads
/// ([`crate::hub`], with per-chunk progress), disk accounting, and safe
/// removal. Active-model *switching* is the consumer's `ModelSlot` keyed by
/// [`ModelManager::model_dir`].
pub struct ModelManager {
    root: PathBuf,
    catalog: Vec<crate::registry::ModelSpec>,
}

impl ModelManager {
    /// Manage `root` with the built-in catalog.
    #[must_use]
    pub fn new(root: PathBuf) -> Self {
        Self::with_catalog(root, crate::registry::catalog())
    }

    /// Manage `root` with an app-supplied catalog (all specs must validate).
    #[must_use]
    pub fn with_catalog(root: PathBuf, catalog: Vec<crate::registry::ModelSpec>) -> Self {
        assert!(
            !root.as_os_str().is_empty(),
            "models root must be non-empty"
        );
        assert!(!catalog.is_empty(), "catalog must not be empty");
        for spec in &catalog {
            if let Err(e) = spec.validate() {
                panic!("invalid catalog spec: {e}");
            }
        }
        Self { root, catalog }
    }

    #[must_use]
    pub fn catalog(&self) -> &[crate::registry::ModelSpec] {
        &self.catalog
    }

    /// The dir a catalog model lives in (whether or not it's installed yet).
    pub fn model_dir(&self, name: &str) -> Result<PathBuf, String> {
        self.spec(name).map(|s| s.dir(&self.root))
    }

    /// Is every required artifact of `name` on disk? (config + tokenizer +
    /// single-file weights or a shard index.)
    pub fn is_installed(&self, name: &str) -> Result<bool, String> {
        let dir = self.model_dir(name)?;
        let weights = dir.join("model.safetensors").is_file()
            || dir.join("model.safetensors.index.json").is_file();
        Ok(weights && dir.join("config.json").is_file() && dir.join("tokenizer.json").is_file())
    }

    /// Download `name` from its spec (resumable, cache-first), reporting
    /// progress per chunk. Returns the model dir ready for `load_from_dir`.
    pub fn install(
        &self,
        name: &str,
        on_progress: impl FnMut(crate::hub::Progress<'_>),
    ) -> Result<PathBuf, String> {
        let spec = self.spec(name)?;
        spec.fetch(&self.root, on_progress)
            .map_err(|e| e.to_string())
    }

    /// Remove `name`'s files from disk (traversal-safe). The caller drops any
    /// live `ModelSlot` first — removal only touches the disk.
    pub fn remove(&self, name: &str) -> Result<(), String> {
        let target = resolve_removal(&self.root, name)?;
        std::fs::remove_dir_all(&target).map_err(|e| format!("remove {name:?}: {e}"))
    }

    /// Disk usage for everything under the root, largest first.
    #[must_use]
    pub fn disk_report(&self) -> DiskReport {
        report(&self.root)
    }

    fn spec(&self, name: &str) -> Result<&crate::registry::ModelSpec, String> {
        self.catalog.iter().find(|s| s.name == name).ok_or_else(|| {
            let known: Vec<&str> = self.catalog.iter().map(|s| s.name.as_str()).collect();
            format!("unknown model {name:?}; catalog has {known:?}")
        })
    }
}

#[cfg(test)]
mod manager_tests {
    use super::*;

    fn temp_root(tag: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("mummu-manager-{tag}"));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    fn fake_install(root: &Path, name: &str) {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        for f in ["config.json", "tokenizer.json", "model.safetensors"] {
            std::fs::write(dir.join(f), b"{}").unwrap();
        }
    }

    #[test]
    fn install_state_and_report_reflect_disk() {
        let root = temp_root("state");
        let mgr = ModelManager::new(root.clone());
        assert_eq!(mgr.is_installed("all-minilm-l6-v2"), Ok(false));

        fake_install(&root, "all-minilm-l6-v2");
        assert_eq!(mgr.is_installed("all-minilm-l6-v2"), Ok(true));

        let rep = mgr.disk_report();
        assert_eq!(rep.models.len(), 1);
        assert_eq!(rep.models[0].name, "all-minilm-l6-v2");
        assert!(rep.total_bytes > 0);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn remove_deletes_only_the_named_model() {
        let root = temp_root("remove");
        let mgr = ModelManager::new(root.clone());
        fake_install(&root, "all-minilm-l6-v2");
        fake_install(&root, "qwen2.5-0.5b-instruct");

        mgr.remove("all-minilm-l6-v2").unwrap();
        assert_eq!(mgr.is_installed("all-minilm-l6-v2"), Ok(false));
        assert_eq!(mgr.is_installed("qwen2.5-0.5b-instruct"), Ok(true));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn unknown_names_fail_loudly_with_the_catalog() {
        let root = temp_root("unknown");
        let mgr = ModelManager::new(root.clone());
        let err = mgr.model_dir("nope").unwrap_err();
        assert!(err.contains("nope") && err.contains("all-minilm-l6-v2"));
        assert!(mgr.remove("nope").is_err());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    #[should_panic(expected = "catalog must not be empty")]
    fn empty_catalog_is_rejected() {
        let _ = ModelManager::with_catalog(PathBuf::from("x"), vec![]);
    }
}
