//! f16 on-GPU validation (the P6 precision milestone): load real checkpoints
//! on the `GpuF16` backend and prove the claims — no shader-compile crash,
//! materially lower VRAM than f32, coherent greedy output. Ignored by
//! default; run with
//!
//! ```text
//! MUMMU_QWEN2_DIR=path/to/qwen2.5-1.5b MUMMU_QWEN3_DIR=path/to/qwen3-0.6b \
//!   cargo test -p mummu --release --test real_f16 -- --ignored --nocapture
//! ```
//!
//! Every `GpuF16` leg lives in THIS binary, isolated from the f32 suites:
//! Burn resolves unspecified-dtype tensor creation against a per-DEVICE
//! default policy, so a `GpuF16` client and a `Gpu` client sharing the
//! device inside one process can flip each other's ambient float dtype
//! (observed 2026-07-23: an f32 GGUF leg read back F16 logits after an f16
//! test ran first). Separate test binaries = separate processes = isolation.

use std::path::PathBuf;

use mummu::backend::{GpuF16, inventory};
use mummu::models::CausalLm;
use mummu::models::{qwen2, qwen3};
use tokenizers::Tokenizer;

fn qwen2_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("MUMMU_QWEN2_DIR")?);
    dir.is_dir().then_some(dir)
}

fn qwen3_dir() -> Option<PathBuf> {
    std::env::var_os("MUMMU_QWEN3_DIR")
        .map(PathBuf::from)
        .filter(|d| d.join("model.safetensors").is_file())
}

#[test]
#[ignore = "needs multi-GB local weights (MUMMU_QWEN2_DIR) + a SHADER_F16 GPU"]
fn qwen2_decodes_coherently_in_f16_on_gpu() {
    let Some(dir) = qwen2_dir() else {
        panic!("set MUMMU_QWEN2_DIR to a dir with config.json/tokenizer.json/model.safetensors");
    };
    assert!(
        inventory().any_shader_f16(),
        "no adapter advertises SHADER_F16 — cannot validate f16 here"
    );

    let raw = mummu::chat::ChatMl::qwen2().render(&[
        mummu::chat::Turn::system("You are a concise assistant."),
        mummu::chat::Turn::user("What is 2+2? Answer in one short sentence."),
    ]);
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json loads");
    let prompt = tok
        .encode(raw.as_str(), false)
        .expect("encodes")
        .get_ids()
        .to_vec();

    // Claim 1 (no crash): building the backend + loading casts bf16 -> f16 via
    // CastFloatAdapter (load_from_dir targets the backend float dtype).
    let device = burn::tensor::Device::<GpuF16>::default();
    let loaded = qwen2::load_from_dir::<GpuF16>(&dir, &device).expect("f16 weights load checked");

    // Claim 3 (coherent output): greedy still answers arithmetic.
    let start = std::time::Instant::now();
    let ids = loaded
        .greedy_generate(&prompt, 32, &device)
        .expect("f16 greedy decode");
    let secs = start.elapsed().as_secs_f64();
    assert!(!ids.is_empty(), "f16 decode produced no tokens before EOS");
    let text = tok.decode(&ids, true).expect("decode");
    eprintln!(
        "[real_f16] {} tokens in {secs:.2}s (incl. prefill): {text:?} ids={:?}",
        ids.len(),
        &ids[..ids.len().min(12)]
    );
    assert!(
        text.contains('4'),
        "expected the f16 answer to mention 4, got: {text:?}"
    );
    // Claim 2 (VRAM) is measured outside the process (nvidia-smi peak while
    // this test runs) — recorded in bench/BASELINE.md.
}

/// f16 leg for the Qwen3 dense arch — bf16 weights cast to f16 on load
/// (`CastFloatAdapter`), and its per-head q/k RMSNorm + decoupled head_dim
/// ride the SAME f32-softmax attention island Qwen2/LFM2 use, so the q·kᵀ
/// scores never overflow f16. Proves the dtype path (P3) and the f16
/// precision milestone (P6) cover the new architecture, not just Qwen2.
/// (Moved here from `real_qwen3.rs` for the process isolation the module
/// docs describe.)
#[test]
#[ignore = "needs the Qwen3 safetensors dir (MUMMU_QWEN3_DIR) + a SHADER_F16 GPU"]
fn real_qwen3_decodes_coherently_in_f16() {
    let dir = qwen3_dir().expect("set MUMMU_QWEN3_DIR to a Qwen3 safetensors dir");
    assert!(
        inventory().any_shader_f16(),
        "no adapter advertises SHADER_F16 — cannot validate f16 here"
    );
    let device = burn::tensor::Device::<GpuF16>::default();

    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let prompt_text = mummu::chat::ChatMl::qwen2().render(&[
        mummu::chat::Turn::system("You are a concise assistant. Do not think, answer directly."),
        mummu::chat::Turn::user("What is 2+2? Answer in one short sentence."),
    ]);
    let prompt = tok
        .encode(prompt_text, true)
        .expect("prompt encodes")
        .get_ids()
        .to_vec();

    // bf16 -> f16 on load; the build must not NaN through the qk-norm + softmax.
    let model = qwen3::load_from_dir::<GpuF16>(&dir, &device).expect("f16 weights load checked");
    let smoke = model
        .sanity_check(&prompt, model.config.vocab_size, &device)
        .expect("f16 forward is finite and non-degenerate (no overflow to NaN)");
    eprintln!(
        "[real_f16/qwen3] sanity smoke: top_id {} · spread {:.3}",
        smoke.top_id, smoke.spread
    );

    let ids = model
        .greedy_generate(&prompt, 48, &device)
        .expect("f16 decode");
    let text = tok.decode(&ids, true).expect("ids decode");
    eprintln!("[real_f16/qwen3] greedy: {text:?}");
    assert!(
        text.contains('4'),
        "expected the f16 answer to mention 4: {text:?}"
    );
}
