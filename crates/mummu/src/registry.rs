//! The model registry: declarative [`ModelSpec`]s and a small built-in
//! catalog of known-good models. Adding a model to Mummu is a manifest entry
//! here (or an app-supplied spec), not new code — the spec names the source
//! repo, the architecture that loads it, and the files it needs; `fetch`
//! hands it to the P3 downloader.

use std::path::{Path, PathBuf};

use crate::hub::{self, HubError, Progress};

/// Which from-scratch implementation loads this checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Architecture {
    /// `models::qwen2` — Qwen2 / Qwen2.5 decoder tiers.
    Qwen2,
    /// `models::lfm2` — LFM2 / LFM2.5 hybrid conv+attention.
    Lfm2,
    /// `models::minilm` — all-MiniLM BERT sentence embedder.
    MiniLm,
}

/// A declarative model manifest entry.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelSpec {
    /// Short cache-dir-safe name, e.g. `qwen2.5-1.5b-instruct`.
    pub name: String,
    /// HuggingFace repo id (`owner/name`).
    pub repo: String,
    /// Git revision (tag, branch, or commit) — pin for reproducibility.
    pub revision: String,
    pub architecture: Architecture,
    /// Rough on-disk size, for settings UIs and fit checks (0 = unknown).
    pub disk_bytes_estimate: u64,
}

impl ModelSpec {
    /// Sanity for manifest entries (also the deserialization gate).
    pub fn validate(&self) -> Result<(), String> {
        if self.name.is_empty()
            || !self
                .name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        {
            return Err(format!("bad spec name {:?}", self.name));
        }
        if !self.repo.contains('/') {
            return Err(format!("repo must be owner/name, got {:?}", self.repo));
        }
        if self.revision.is_empty() {
            return Err("revision must be non-empty (pin something)".into());
        }
        Ok(())
    }

    /// The cache directory this model lives in under `models_root`.
    #[must_use]
    pub fn dir(&self, models_root: &Path) -> PathBuf {
        models_root.join(&self.name)
    }

    /// Download this model into `models_root` (resumable, cache-first; see
    /// [`hub::fetch_model`]) and return its directory, ready for the
    /// architecture's `load_from_dir`.
    pub fn fetch(
        &self,
        models_root: &Path,
        on_progress: impl FnMut(Progress<'_>),
    ) -> Result<PathBuf, HubError> {
        assert!(self.validate().is_ok(), "fetch of an invalid spec");
        hub::fetch_model(
            &self.repo,
            &self.revision,
            &self.dir(models_root),
            on_progress,
        )
    }
}

/// The built-in catalog: the models Mummu has ported and parity-verified (or
/// is actively gating — see the ROADMAP P2 checklist for each one's status).
#[must_use]
pub fn catalog() -> Vec<ModelSpec> {
    let entries = vec![
        ModelSpec {
            name: "qwen2.5-1.5b-instruct".into(),
            repo: "Qwen/Qwen2.5-1.5B-Instruct".into(),
            revision: "main".into(),
            architecture: Architecture::Qwen2,
            disk_bytes_estimate: 3_100_000_000,
        },
        ModelSpec {
            name: "qwen2.5-0.5b-instruct".into(),
            repo: "Qwen/Qwen2.5-0.5B-Instruct".into(),
            revision: "main".into(),
            architecture: Architecture::Qwen2,
            disk_bytes_estimate: 1_000_000_000,
        },
        ModelSpec {
            name: "lfm2.5-1.2b".into(),
            repo: "LiquidAI/LFM2.5-1.2B-Instruct".into(),
            revision: "main".into(),
            architecture: Architecture::Lfm2,
            disk_bytes_estimate: 2_400_000_000,
        },
        ModelSpec {
            name: "all-minilm-l6-v2".into(),
            repo: "sentence-transformers/all-MiniLM-L6-v2".into(),
            revision: "main".into(),
            architecture: Architecture::MiniLm,
            disk_bytes_estimate: 91_000_000,
        },
    ];
    debug_assert!(
        entries.iter().all(|s| s.validate().is_ok()),
        "built-in catalog must validate"
    );
    entries
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_catalog_validates_and_names_are_unique() {
        let cat = catalog();
        assert!(cat.len() >= 3);
        for spec in &cat {
            spec.validate().unwrap_or_else(|e| panic!("{e}"));
        }
        let mut names: Vec<&str> = cat.iter().map(|s| s.name.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), cat.len(), "duplicate names in the catalog");
    }

    #[test]
    fn spec_dir_is_rooted_and_named() {
        let spec = &catalog()[0];
        let dir = spec.dir(Path::new("root/models"));
        assert_eq!(dir, Path::new("root/models").join(&spec.name));
    }

    #[test]
    fn traversal_names_are_rejected() {
        let mut spec = catalog()[0].clone();
        spec.name = "../escape".into();
        assert!(spec.validate().is_err(), "path traversal must not validate");
    }

    #[test]
    fn bare_repo_is_rejected() {
        let mut spec = catalog()[0].clone();
        spec.repo = "qwen".into();
        assert!(spec.validate().is_err());
    }

    #[test]
    fn specs_round_trip_through_json() {
        let spec = &catalog()[0];
        let json = serde_json::to_string(spec).unwrap();
        let back: ModelSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, spec.name);
        assert_eq!(back.architecture, spec.architecture);
    }
}
