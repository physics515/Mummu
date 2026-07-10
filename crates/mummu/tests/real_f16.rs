//! f16 on-GPU validation (the P6 precision milestone): load Qwen2.5 on the
//! `GpuF16` backend and prove the three claims — no shader-compile crash,
//! materially lower VRAM than f32, coherent greedy output. Ignored by
//! default; run with
//!
//! ```text
//! MUMMU_QWEN2_DIR=path/to/qwen2.5-1.5b cargo test -p mummu --release --test real_f16 -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mummu::backend::{GpuF16, inventory};
use mummu::models::CausalLm;
use mummu::models::qwen2;
use tokenizers::Tokenizer;

fn qwen2_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("MUMMU_QWEN2_DIR")?);
    dir.is_dir().then_some(dir)
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
