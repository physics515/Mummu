//! Weight import: safetensors → Burn modules, checked and loud.
//!
//! The pieces every model load shares (P3): a dtype-cast adapter (HF ships
//! bf16, which wgpu can't ingest directly), and a checked-load wrapper that
//! **fails on missing or errored params** instead of silently zero-initing —
//! a partial load is a quietly broken model.

use std::path::{Path, PathBuf};
use std::rc::Rc;

use burn::module::Module;
use burn::store::{ModuleAdapter, ModuleSnapshot, SafetensorsStore, TensorSnapshot};
use burn::tensor::DType;

/// Everything that can go wrong turning files on disk into a loaded model.
#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    #[error("required file missing: {0}")]
    MissingFile(PathBuf),
    #[error("parse {file}: {reason}")]
    Parse { file: PathBuf, reason: String },
    #[error("load weights ({file}): {reason}")]
    Load { file: PathBuf, reason: String },
    #[error(
        "weight load incomplete ({file}): {applied} applied, {missing} missing, {errors} errors\n{report}"
    )]
    Incomplete {
        file: PathBuf,
        applied: usize,
        missing: usize,
        errors: usize,
        report: String,
    },
}

/// Cast bf16/f16/f64 float tensors to a target dtype on load. HF checkpoints
/// are commonly stored in **bf16**; `burn-store` keeps the source dtype, which
/// the wgpu backend can't ingest. Mirrors Burn's `HalfPrecisionAdapter` but
/// covers bf16 → f32/f16 too. Non-float tensors pass through untouched.
#[derive(Clone)]
pub struct CastFloatAdapter {
    target: DType,
}

impl CastFloatAdapter {
    #[must_use]
    pub fn new(target: DType) -> Self {
        assert!(
            matches!(target, DType::F16 | DType::F32 | DType::F64 | DType::BF16),
            "CastFloatAdapter: target must be a float dtype, got {target:?}"
        );
        Self { target }
    }
}

impl ModuleAdapter for CastFloatAdapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        let is_float = matches!(
            snapshot.dtype,
            DType::BF16 | DType::F16 | DType::F32 | DType::F64
        );
        if !is_float || snapshot.dtype == self.target {
            return snapshot.clone();
        }
        let target = self.target;
        let data_fn = snapshot.clone_data_fn();
        let cast = Rc::new(move || Ok(data_fn()?.convert_dtype(target)));
        TensorSnapshot::from_closure(
            cast,
            target,
            snapshot.shape.clone(),
            snapshot.path_stack.clone().unwrap_or_default(),
            snapshot.container_stack.clone().unwrap_or_default(),
            snapshot.tensor_id.unwrap_or_default(),
        )
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

/// Load `store` into `module`, refusing partial results: any missing param or
/// per-tensor error is an [`ImportError::Incomplete`] carrying the store's own
/// readable report. Unused checkpoint tensors are *allowed* (e.g. BERT's
/// intentionally-skipped `pooler.*`) — callers that care inspect the report.
pub fn load_checked<M: Module<B> + ModuleSnapshot<B>, B: burn::tensor::backend::Backend>(
    module: &mut M,
    store: &mut SafetensorsStore,
    weights_path: &Path,
) -> Result<(), ImportError> {
    let report = module.load_from(store).map_err(|e| ImportError::Load {
        file: weights_path.to_path_buf(),
        reason: e.to_string(),
    })?;
    if !report.errors.is_empty() || !report.missing.is_empty() {
        return Err(ImportError::Incomplete {
            file: weights_path.to_path_buf(),
            applied: report.applied.len(),
            missing: report.missing.len(),
            errors: report.errors.len(),
            report: format!("{report}"),
        });
    }
    debug_assert!(
        !report.applied.is_empty(),
        "load_checked: a successful load must have applied at least one tensor"
    );
    Ok(())
}

/// `dir/file`, or [`ImportError::MissingFile`] if absent.
pub fn required_file(dir: &Path, file: &str) -> Result<PathBuf, ImportError> {
    assert!(!file.is_empty(), "required_file: empty file name");
    let path = dir.join(file);
    if path.is_file() {
        Ok(path)
    } else {
        Err(ImportError::MissingFile(path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_file_finds_present_and_rejects_absent() {
        let dir = std::env::temp_dir().join("mummu_import_test");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        assert!(required_file(&dir, "config.json").is_ok());
        let err = required_file(&dir, "nope.bin").unwrap_err();
        assert!(matches!(err, ImportError::MissingFile(_)));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    #[should_panic(expected = "float dtype")]
    fn cast_adapter_rejects_non_float_target() {
        let _ = CastFloatAdapter::new(DType::I32);
    }
}
