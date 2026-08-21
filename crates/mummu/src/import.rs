//! Weight import: checkpoint files → Burn modules, checked and loud.
//!
//! The pieces every model load shares (P3): a dtype-cast adapter (HF ships
//! bf16, which wgpu can't ingest directly), a weights-file picker
//! (**safetensors** preferred, the PyTorch state dict `pytorch_model.bin` as
//! the fallback for models never re-shipped as safetensors), and a
//! checked-load wrapper that **fails on missing or errored params** instead
//! of silently zero-initing — a partial load is a quietly broken model.

use std::path::{Path, PathBuf};
use std::rc::Rc;

use burn::module::Module;
use burn::store::{
    ModuleAdapter, ModuleSnapshot, ModuleStore, PyTorchToBurnAdapter, SafetensorsStore,
    TensorSnapshot,
};
use burn::tensor::DType;

use crate::gguf::{GgufFile, GgufMap, GgufTensorInfo};

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
    /// A checkpoint's own metadata files disagree with each other (e.g.
    /// `tokenizer_config.json` names an EOS `config.json` does not, or a
    /// chat-template whose tool-call convention is not the one this family's
    /// byte-verified renderer speaks). A repackaging bug that a checked *weight*
    /// load cannot see; surfaced loudly at load rather than mis-stopping or
    /// mis-templating at generate time.
    #[error("inconsistent metadata ({file}): {reason}")]
    Inconsistent { file: PathBuf, reason: String },
}

/// Why a freshly-imported model failed its post-load **sanity smoke** (one
/// forward, checked for liveness). These are the silent-broken-import failure
/// modes a checked *load* cannot see — the weights all applied, but the model
/// does not actually compute.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum SanityError {
    /// Logits contain NaN or ±Inf — the classic signature of a wrong dtype
    /// (e.g. an f16 overflow that should have been an f32 island) or corrupt
    /// weight bytes that still deserialized to the right shape.
    #[error(
        "{count} non-finite logit(s) (first at index {first_index}) — bad dtype or corrupt weights"
    )]
    NonFinite { count: usize, first_index: usize },
    /// The logits width does not equal the expected vocabulary — a config /
    /// tokenizer / checkpoint mismatch (the model and its tokenizer disagree
    /// on how many tokens exist).
    #[error("logits width {logits_len} != expected vocab {expected} — config/tokenizer mismatch")]
    WrongVocab { logits_len: usize, expected: usize },
    /// Every logit is (near-)identical — a dead forward: zero-initialized
    /// weights, or an all-masked/zeroed activation path that a checked load
    /// still reports as fully applied.
    #[error("degenerate logits (spread {spread:e} < {threshold:e}) — dead/zero-init forward")]
    Degenerate { spread: f32, threshold: f32 },
}

/// The healthy result of a [`logit_sanity`] smoke: the winning token and how
/// sharply the distribution favors it (a coarse confidence signal for logs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SanitySmoke {
    /// argmax token id.
    pub top_id: u32,
    /// The winning logit.
    pub top_logit: f32,
    /// max − min over the logits (the dynamic range; large for a live model).
    pub spread: f32,
}

/// Logits below this spread (max − min) are treated as a dead/uniform forward.
/// A live LM's first-token logits span tens; a zero-init forward spans ~0. The
/// gap is enormous, so a tiny threshold never false-positives a real model.
const DEGENERATE_SPREAD: f32 = 1e-4;

/// Post-load **sanity smoke** over one forward's logits: finite, the expected
/// width, and not degenerate. This is *liveness*, not parity — an arbitrary
/// user import has no reference to compare against (catalog models get the P7
/// parity gates); this only proves the model actually computes rather than
/// silently returning garbage a checked load reported as fully applied.
pub fn logit_sanity(logits: &[f32], expected_vocab: usize) -> Result<SanitySmoke, SanityError> {
    assert!(
        expected_vocab > 0,
        "logit_sanity: expected_vocab must be positive"
    );
    if logits.len() != expected_vocab {
        return Err(SanityError::WrongVocab {
            logits_len: logits.len(),
            expected: expected_vocab,
        });
    }
    assert!(
        !logits.is_empty(),
        "logit_sanity: width matched a positive vocab"
    );

    if let Some((first_index, _)) = logits.iter().enumerate().find(|(_, l)| !l.is_finite()) {
        let count = logits.iter().filter(|l| !l.is_finite()).count();
        return Err(SanityError::NonFinite { count, first_index });
    }

    // All finite: min/max define the spread and the argmax.
    let (mut min, mut max, mut top_id) = (f32::INFINITY, f32::NEG_INFINITY, 0usize);
    for (i, &l) in logits.iter().enumerate() {
        if l < min {
            min = l;
        }
        if l > max {
            (max, top_id) = (l, i);
        }
    }
    let spread = max - min;
    if spread < DEGENERATE_SPREAD {
        return Err(SanityError::Degenerate {
            spread,
            threshold: DEGENERATE_SPREAD,
        });
    }
    debug_assert!(
        spread > 0.0 && max.is_finite(),
        "healthy smoke has positive finite spread"
    );
    Ok(SanitySmoke {
        top_id: top_id as u32,
        top_logit: max,
        spread,
    })
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

/// Load `store` (any format: safetensors, PyTorch state dict, …) into
/// `module`, refusing partial results: any missing param or per-tensor error
/// is an [`ImportError::Incomplete`] carrying the store's own readable
/// report. Unused checkpoint tensors are *allowed* (e.g. BERT's
/// intentionally-skipped `pooler.*`) — callers that care inspect the report.
pub fn load_checked<M, B, S>(
    module: &mut M,
    store: &mut S,
    weights_path: &Path,
) -> Result<(), ImportError>
where
    M: Module<B> + ModuleSnapshot<B>,
    B: burn::tensor::backend::Backend,
    S: ModuleStore,
{
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

/// Largest f32 payload [`DequantSink::Auto`] will let a dequant hold in RAM.
///
/// The in-memory sink's peak is ~2x the payload (the blob, plus the model
/// being built from it), so this ceiling is really a ~1 GiB peak — small
/// enough to be free on any machine that could run the model at all. Above
/// it, [`DequantSink::Scratch`] costs one f32 write plus an mmap read-back
/// and halves the peak; measured on Qwen2.5-1.5B (6.2 GB of f32), that trade
/// is inside the run-to-run noise of the load it is part of, so there is no
/// reason to keep paying the doubled peak for anything of real size.
pub const MAX_IN_MEMORY_DEQUANT_BYTES: u64 = 512 << 20;

/// Where a GGUF dequant's f32 payload lands on its way into a
/// [`SafetensorsStore`].
///
/// The two sinks are byte-identical by construction — `gguf::dequant_into` is
/// ONE function behind both, pinned by a unit test — so this is a resource
/// trade, never a correctness one. [`Self::Memory`] peaks at ~2x the payload
/// and needs nothing; [`Self::Scratch`] peaks at ~1x (the payload becomes
/// file-backed mmap) and needs payload-sized free disk beside the weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DequantSink {
    /// Hold the whole f32 payload in RAM and hand it to the store as bytes.
    Memory,
    /// Stream it to a scratch file beside the weights; `burn-store` mmaps
    /// that back and materializes tensors lazily, so the payload is never
    /// charged to commit.
    Scratch,
    /// Pick by payload size: [`Self::Memory`] only up to
    /// [`MAX_IN_MEMORY_DEQUANT_BYTES`], [`Self::Scratch`] above it.
    Auto,
}

impl DequantSink {
    /// Resolve [`Self::Auto`] against a payload size; the other two answer
    /// for themselves. Never returns `Auto`.
    #[must_use]
    pub fn resolve(self, total_f32_bytes: u64) -> Self {
        let picked = match self {
            Self::Auto if total_f32_bytes > MAX_IN_MEMORY_DEQUANT_BYTES => Self::Scratch,
            Self::Auto => Self::Memory,
            explicit => explicit,
        };
        debug_assert!(picked != Self::Auto, "resolve must decide");
        debug_assert!(
            picked != Self::Memory
                || total_f32_bytes <= MAX_IN_MEMORY_DEQUANT_BYTES
                || self == Self::Memory,
            "Auto never picks Memory above the ceiling"
        );
        picked
    }
}

/// Owns a scratch file for the life of one load.
///
/// The file is as large as the weights it carries (~28 GB dequantizing
/// OLMoE-1B-7B's Q4_K_M), so leaving one behind on a failed load would
/// quietly fill the disk over a few retries. `Drop` removes it on every exit
/// path, success or `?` — and because the store reads it lazily, the guard
/// must outlive `load_checked`, which holding it as a local does.
#[derive(Debug)]
pub struct ScratchFile {
    path: PathBuf,
}

impl ScratchFile {
    /// Placed beside the weights, so the scratch write lands on the same
    /// volume as the model rather than on a small system temp drive.
    ///
    /// The name carries a process-unique counter as well as the pid: two
    /// concurrent loads in one process must not choose the same file and
    /// interleave their writes into it.
    pub fn new(dir: &Path) -> Result<Self, ImportError> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        assert!(!dir.as_os_str().is_empty(), "scratch dir must be named");
        let path = dir.join(format!(
            "mummu-scratch-{}-{}.safetensors.tmp",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        // A leftover from a killed process must never be mistaken for ours.
        if path.exists() {
            std::fs::remove_file(&path).map_err(|e| ImportError::Parse {
                file: path.clone(),
                reason: format!("could not clear a stale scratch file: {e}"),
            })?;
        }
        Ok(Self { path })
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        debug_assert!(!self.path.as_os_str().is_empty(), "scratch path is named");
        &self.path
    }
}

impl Drop for ScratchFile {
    fn drop(&mut self) {
        // Best effort by construction: a Drop that can fail has nowhere to
        // report to, and a stranded scratch file is not worth a panic.
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Dequantize `f` to f32 safetensors and return the store every GGUF loader
/// then adds its own key remaps to, plus the scratch guard (if any) that must
/// stay alive until the load finishes.
///
/// This is the one place the four GGUF ports share: the sink choice, the
/// adapter chain, and — the part worth centralizing — taking the target float
/// dtype from the TYPE (`B::FloatElem`) rather than from a probe tensor.
/// Unspecified-dtype tensor creation follows the per-DEVICE default policy,
/// which another backend alias sharing the device (`Gpu` vs `GpuF16`) may have
/// flipped in this process.
pub fn gguf_store<B: burn::tensor::backend::Backend>(
    f: &GgufFile,
    map: &dyn Fn(&GgufTensorInfo) -> Option<GgufMap>,
    sink: DequantSink,
) -> Result<(SafetensorsStore, Option<ScratchFile>), ImportError> {
    let parse = |reason: String| ImportError::Parse {
        file: f.path.clone(),
        reason,
    };
    let total_f32_bytes: u64 = f.tensors.iter().map(|t| t.element_count() * 4).sum();
    let target_float = <B::FloatElem as burn::tensor::Element>::dtype();
    let adapter = PyTorchToBurnAdapter.chain(CastFloatAdapter::new(target_float));

    match sink.resolve(total_f32_bytes) {
        DequantSink::Memory => {
            let blob = f
                .dequant_to_safetensors(map)
                .map_err(|e| parse(e.to_string()))?;
            assert!(blob.len() > 8, "a parsed GGUF yields a non-empty blob");
            let store = SafetensorsStore::from_bytes(Some(blob))
                .with_from_adapter(adapter)
                .allow_partial(true);
            Ok((store, None))
        }
        DequantSink::Scratch => {
            // Beside the gguf, so the scratch write lands on the volume the
            // weights already live on.
            let scratch = ScratchFile::new(f.path.parent().unwrap_or(Path::new(".")))?;
            let bytes = f
                .dequant_to_safetensors_file(map, scratch.path())
                .map_err(|e| parse(e.to_string()))?;
            assert!(bytes > 0, "a parsed GGUF yields a non-empty payload");
            let store = SafetensorsStore::from_file(scratch.path().to_path_buf())
                .with_from_adapter(adapter)
                .allow_partial(true);
            Ok((store, Some(scratch)))
        }
        DequantSink::Auto => unreachable!("resolve never returns Auto"),
    }
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

/// The weights checkpoint found in a model dir.
#[derive(Debug, Clone)]
pub enum WeightsFile {
    /// `model.safetensors` — the primary format.
    Safetensors(PathBuf),
    /// `pytorch_model.bin` — the PyTorch state dict older checkpoints ship.
    PytorchBin(PathBuf),
}

/// Pick the weights file in `dir`: `model.safetensors` when present, else
/// `pytorch_model.bin`. Reports the *safetensors* name when neither exists
/// (it's the file a fresh download would produce).
pub fn weights_file(dir: &Path) -> Result<WeightsFile, ImportError> {
    let safetensors = dir.join("model.safetensors");
    if safetensors.is_file() {
        return Ok(WeightsFile::Safetensors(safetensors));
    }
    let bin = dir.join("pytorch_model.bin");
    if bin.is_file() {
        return Ok(WeightsFile::PytorchBin(bin));
    }
    Err(ImportError::MissingFile(safetensors))
}

#[cfg(test)]
mod tests {
    use super::{DequantSink, MAX_IN_MEMORY_DEQUANT_BYTES, ScratchFile};

    #[test]
    fn auto_picks_the_scratch_sink_exactly_above_the_ceiling() {
        // The boundary is the whole point: `Auto` must never leave a payload
        // in RAM whose 2x peak is what this ceiling exists to cap.
        assert_eq!(DequantSink::Auto.resolve(0), DequantSink::Memory);
        assert_eq!(
            DequantSink::Auto.resolve(MAX_IN_MEMORY_DEQUANT_BYTES),
            DequantSink::Memory
        );
        assert_eq!(
            DequantSink::Auto.resolve(MAX_IN_MEMORY_DEQUANT_BYTES + 1),
            DequantSink::Scratch
        );
        // An explicit choice is honoured in BOTH directions, including the
        // one `Auto` would never make — the A/B that set the ceiling needs it.
        assert_eq!(DequantSink::Memory.resolve(u64::MAX), DequantSink::Memory);
        assert_eq!(DequantSink::Scratch.resolve(0), DequantSink::Scratch);
    }

    #[test]
    fn scratch_files_are_unique_per_instance_and_deleted_on_drop() {
        let dir = std::env::temp_dir().join("mummu-scratch-tests");
        std::fs::create_dir_all(&dir).expect("temp dir");

        let a = ScratchFile::new(&dir).expect("names a scratch file");
        let b = ScratchFile::new(&dir).expect("names a second one");
        assert_ne!(
            a.path(),
            b.path(),
            "two live loads must not share one scratch file"
        );

        let path = a.path().to_path_buf();
        std::fs::write(&path, b"payload").expect("writes");
        assert!(path.is_file());
        drop(a);
        assert!(!path.exists(), "Drop removes the scratch file");

        // And a stale file at the same name is cleared, not adopted: write
        // one at b's path, then re-create a guard for it.
        let stale = b.path().to_path_buf();
        std::fs::write(&stale, b"stale").expect("writes");
        drop(b);
        assert!(!stale.exists());
    }

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

    #[test]
    fn logit_sanity_accepts_a_live_distribution() {
        let logits = [0.1, 2.5, -1.0, 9.0, 0.3];
        let smoke = logit_sanity(&logits, 5).expect("live logits pass");
        assert_eq!(smoke.top_id, 3);
        assert_eq!(smoke.top_logit, 9.0);
        assert!(
            (smoke.spread - 10.0).abs() < 1e-6,
            "spread = max-min = 9-(-1)"
        );
    }

    #[test]
    fn logit_sanity_catches_non_finite() {
        let nan = [1.0, f32::NAN, 3.0, 2.0];
        assert_eq!(
            logit_sanity(&nan, 4),
            Err(SanityError::NonFinite {
                count: 1,
                first_index: 1
            })
        );
        let inf = [f32::INFINITY, 0.0, f32::NEG_INFINITY];
        assert_eq!(
            logit_sanity(&inf, 3),
            Err(SanityError::NonFinite {
                count: 2,
                first_index: 0
            })
        );
    }

    #[test]
    fn logit_sanity_catches_wrong_vocab() {
        // The finiteness check must not run before the width check — a short
        // vector is a mismatch, reported as such (not an index panic).
        assert_eq!(
            logit_sanity(&[1.0, 2.0, 3.0], 4),
            Err(SanityError::WrongVocab {
                logits_len: 3,
                expected: 4
            })
        );
    }

    #[test]
    fn logit_sanity_catches_degenerate_forward() {
        // Zero-init / dead forward: all (near-)equal.
        let flat = [0.5f32; 8];
        assert!(matches!(
            logit_sanity(&flat, 8),
            Err(SanityError::Degenerate { .. })
        ));
        // A spread just above the threshold is live.
        let mut live = [0.0f32; 8];
        live[7] = 1e-3;
        assert!(logit_sanity(&live, 8).is_ok());
    }

    #[test]
    #[should_panic(expected = "expected_vocab must be positive")]
    fn logit_sanity_rejects_zero_vocab() {
        let _ = logit_sanity(&[], 0);
    }

    #[test]
    fn weights_file_prefers_safetensors_falls_back_to_pytorch() {
        let dir = std::env::temp_dir().join("mummu_weights_file_test");
        std::fs::create_dir_all(&dir).unwrap();
        // Neither present: missing, reported by the safetensors name.
        assert!(matches!(
            weights_file(&dir),
            Err(ImportError::MissingFile(p)) if p.ends_with("model.safetensors")
        ));
        // Only the state dict: the PyTorch path.
        std::fs::write(dir.join("pytorch_model.bin"), b"pt").unwrap();
        assert!(matches!(weights_file(&dir), Ok(WeightsFile::PytorchBin(_))));
        // Both: safetensors wins.
        std::fs::write(dir.join("model.safetensors"), b"st").unwrap();
        assert!(matches!(
            weights_file(&dir),
            Ok(WeightsFile::Safetensors(_))
        ));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
