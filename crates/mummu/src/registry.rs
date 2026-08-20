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
    /// `models::qwen3` — Qwen3 dense decoder (per-head q/k norm, no qkv bias,
    /// decoupled head_dim); the function-calling tier (4B / 9B).
    Qwen3,
    /// `models::lfm2` — LFM2 / LFM2.5 hybrid conv+attention.
    Lfm2,
    /// `models::minilm` — all-MiniLM BERT sentence embedder.
    MiniLm,
    /// `models::olmoe` — OLMoE sparse mixture-of-experts decoder (the zoo's
    /// first MoE). Imports from GGUF (experts pre-fused) or from HF
    /// safetensors (experts fused on import).
    Olmoe,
}

/// How the checkpoint's weights are stored — which fetch + load path a spec
/// takes.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum WeightFormat {
    /// `config.json` + `tokenizer.json` + `model.safetensors` (or a shard
    /// index); fetched by [`hub::fetch_model`], loaded by `load_from_dir`.
    #[default]
    Safetensors,
    /// One self-contained `.gguf` file in the repo (config + tokenizer +
    /// weights in the metadata); loaded by the architecture's
    /// `load_from_gguf` + [`crate::tokenizer::tokenizer_from_gguf`].
    Gguf {
        /// The file name inside the repo, e.g. `qwen2.5-1.5b-instruct-q4_k_m.gguf`.
        file: String,
    },
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
    /// Weight storage (absent in older manifests = safetensors).
    #[serde(default)]
    pub format: WeightFormat,
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
        if let WeightFormat::Gguf { file } = &self.format {
            let safe = !file.is_empty()
                && !file.contains("..")
                && !file.starts_with('/')
                && file.ends_with(".gguf");
            if !safe {
                return Err(format!("bad gguf file name {file:?}"));
            }
        }
        Ok(())
    }

    /// The cache directory this model lives in under `models_root`.
    #[must_use]
    pub fn dir(&self, models_root: &Path) -> PathBuf {
        models_root.join(&self.name)
    }

    /// For a GGUF spec: the local path of the model file after [`Self::fetch`].
    #[must_use]
    pub fn gguf_path(&self, models_root: &Path) -> Option<PathBuf> {
        match &self.format {
            WeightFormat::Gguf { file } => Some(self.dir(models_root).join(file)),
            WeightFormat::Safetensors => None,
        }
    }

    /// Download this model into `models_root` (resumable, cache-first; see
    /// [`hub::fetch_model`] / [`hub::fetch_file`]) and return its directory,
    /// ready for the architecture's `load_from_dir` / `load_from_gguf`.
    pub fn fetch(
        &self,
        models_root: &Path,
        on_progress: impl FnMut(Progress<'_>),
    ) -> Result<PathBuf, HubError> {
        assert!(self.validate().is_ok(), "fetch of an invalid spec");
        let dir = self.dir(models_root);
        match &self.format {
            WeightFormat::Safetensors => {
                hub::fetch_model(&self.repo, &self.revision, &dir, on_progress)
            }
            WeightFormat::Gguf { file } => {
                let url = hub::hub_file_url(&self.repo, &self.revision, file);
                std::fs::create_dir_all(&dir).map_err(|e| HubError::Io {
                    path: dir.clone(),
                    reason: e.to_string(),
                })?;
                hub::fetch_file(&url, &dir.join(file), on_progress)?;
                Ok(dir)
            }
        }
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
            format: WeightFormat::Safetensors,
            disk_bytes_estimate: 3_100_000_000,
        },
        ModelSpec {
            name: "qwen2.5-0.5b-instruct".into(),
            repo: "Qwen/Qwen2.5-0.5B-Instruct".into(),
            revision: "main".into(),
            architecture: Architecture::Qwen2,
            format: WeightFormat::Safetensors,
            disk_bytes_estimate: 1_000_000_000,
        },
        ModelSpec {
            name: "lfm2.5-1.2b".into(),
            repo: "LiquidAI/LFM2.5-1.2B-Instruct".into(),
            revision: "main".into(),
            architecture: Architecture::Lfm2,
            format: WeightFormat::Safetensors,
            disk_bytes_estimate: 2_400_000_000,
        },
        ModelSpec {
            name: "lfm2.5-230m".into(),
            repo: "LiquidAI/LFM2.5-230M".into(),
            revision: "main".into(),
            architecture: Architecture::Lfm2,
            format: WeightFormat::Safetensors,
            disk_bytes_estimate: 500_000_000,
        },
        ModelSpec {
            name: "all-minilm-l6-v2".into(),
            repo: "sentence-transformers/all-MiniLM-L6-v2".into(),
            revision: "main".into(),
            architecture: Architecture::MiniLm,
            format: WeightFormat::Safetensors,
            disk_bytes_estimate: 91_000_000,
        },
        // Single-file GGUF variants — quarter the download, same model
        // (proven vs the bf16 safetensors builds in tests/real_gguf.rs).
        ModelSpec {
            name: "qwen2.5-1.5b-instruct-q4km".into(),
            repo: "Qwen/Qwen2.5-1.5B-Instruct-GGUF".into(),
            revision: "main".into(),
            architecture: Architecture::Qwen2,
            format: WeightFormat::Gguf {
                file: "qwen2.5-1.5b-instruct-q4_k_m.gguf".into(),
            },
            disk_bytes_estimate: 1_120_000_000,
        },
        ModelSpec {
            name: "lfm2.5-1.2b-q4km".into(),
            repo: "LiquidAI/LFM2.5-1.2B-Instruct-GGUF".into(),
            revision: "main".into(),
            architecture: Architecture::Lfm2,
            format: WeightFormat::Gguf {
                file: "LFM2.5-1.2B-Instruct-Q4_K_M.gguf".into(),
            },
            disk_bytes_estimate: 731_000_000,
        },
        // Qwen3 dense — the local function-calling tier. 0.6B is the fast
        // parity-validation / CPU tier; 4B is the BFCL sweet spot.
        ModelSpec {
            name: "qwen3-0.6b".into(),
            repo: "Qwen/Qwen3-0.6B".into(),
            revision: "main".into(),
            architecture: Architecture::Qwen3,
            format: WeightFormat::Safetensors,
            disk_bytes_estimate: 1_500_000_000,
        },
        ModelSpec {
            name: "qwen3-0.6b-q4km".into(),
            repo: "unsloth/Qwen3-0.6B-GGUF".into(),
            revision: "main".into(),
            architecture: Architecture::Qwen3,
            format: WeightFormat::Gguf {
                file: "Qwen3-0.6B-Q4_K_M.gguf".into(),
            },
            disk_bytes_estimate: 484_000_000,
        },
        ModelSpec {
            name: "qwen3-4b".into(),
            repo: "Qwen/Qwen3-4B".into(),
            revision: "main".into(),
            architecture: Architecture::Qwen3,
            format: WeightFormat::Safetensors,
            disk_bytes_estimate: 8_100_000_000,
        },
        ModelSpec {
            name: "qwen3-4b-q4km".into(),
            repo: "Qwen/Qwen3-4B-GGUF".into(),
            revision: "main".into(),
            architecture: Architecture::Qwen3,
            format: WeightFormat::Gguf {
                file: "Qwen3-4B-Q4_K_M.gguf".into(),
            },
            disk_bytes_estimate: 2_500_000_000,
        },
        // The zoo's first MoE: 64 experts, 8 active per token (1B active /
        // 7B total). Resident-everything first cut — ~28 GB dequantized to
        // f32, sized for the CPU backend (128 GB reference machine).
        ModelSpec {
            name: "olmoe-1b-7b-0125-instruct-q4km".into(),
            repo: "allenai/OLMoE-1B-7B-0125-Instruct-GGUF".into(),
            revision: "main".into(),
            architecture: Architecture::Olmoe,
            format: WeightFormat::Gguf {
                file: "OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf".into(),
            },
            disk_bytes_estimate: 4_210_000_000,
        },
        // The same MoE from its HF source: 3 bf16 safetensors shards + an
        // index, with the 64 experts stored separately. `load_from_dir` fuses
        // them into the `[experts, out, in]` banks the module holds.
        ModelSpec {
            name: "olmoe-1b-7b-0125-instruct".into(),
            repo: "allenai/OLMoE-1B-7B-0125-Instruct".into(),
            revision: "main".into(),
            architecture: Architecture::Olmoe,
            format: WeightFormat::Safetensors,
            disk_bytes_estimate: 13_800_000_000,
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

    #[test]
    fn gguf_specs_round_trip_and_old_manifests_default_to_safetensors() {
        let gguf = catalog()
            .into_iter()
            .find(|s| matches!(s.format, WeightFormat::Gguf { .. }))
            .expect("catalog has a gguf entry");
        let json = serde_json::to_string(&gguf).unwrap();
        let back: ModelSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.format, gguf.format);
        let file = match &gguf.format {
            WeightFormat::Gguf { file } => file.clone(),
            WeightFormat::Safetensors => unreachable!(),
        };
        assert_eq!(
            gguf.gguf_path(Path::new("root")),
            Some(Path::new("root").join(&gguf.name).join(file))
        );

        // A manifest written before `format` existed still deserializes.
        let old = r#"{"name":"m","repo":"a/b","revision":"main",
                      "architecture":"Qwen2","disk_bytes_estimate":1}"#;
        let back: ModelSpec = serde_json::from_str(old).unwrap();
        assert_eq!(back.format, WeightFormat::Safetensors);
        assert_eq!(back.gguf_path(Path::new("root")), None);
    }

    #[test]
    fn bad_gguf_file_names_are_rejected() {
        let mut spec = catalog()[0].clone();
        for bad in ["", "../up.gguf", "/abs.gguf", "weights.bin"] {
            spec.format = WeightFormat::Gguf { file: bad.into() };
            assert!(spec.validate().is_err(), "{bad:?} must not validate");
        }
        spec.format = WeightFormat::Gguf {
            file: "ok-model.q4_k_m.gguf".into(),
        };
        assert!(spec.validate().is_ok());
    }
}
