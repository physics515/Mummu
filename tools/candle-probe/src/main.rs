//! Same-weights Candle reference for the P7 parity gate.
//!
//! Usage: `candle-probe <model-dir> [k]` — loads `config.json` /
//! `tokenizer.json` / `model.safetensors` from `<model-dir>` (a Qwen2-family
//! checkpoint), runs ONE forward over the fixed ChatML prompt below on CPU in
//! f32, and prints a JSON object with the prompt, the top-k ids, and their
//! logits. Redirect the output into
//! `crates/mummu/tests/fixtures/<model>_first_logits.json` to refresh the
//! committed fixture that `tests/parity_qwen2.rs` compares Burn against.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::{Config, ModelForCausalLM};

/// Must stay identical to `PROMPT`/`chatml` in `crates/mummu/tests/parity_qwen2.rs`;
/// the fixture carries the rendered prompt so the test can verify it drifted nowhere.
const PROMPT: &str = "List the first five prime numbers.";

fn chatml(user: &str) -> String {
    format!(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    )
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().ok_or("usage: candle-probe <model-dir> [k]")?);
    let k: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(5);
    assert!(k >= 1 && k <= 64, "k out of range: {k}");
    assert!(dir.is_dir(), "not a directory: {}", dir.display());

    let device = Device::Cpu;
    let config: Config = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &device)?
    };
    let mut model = ModelForCausalLM::new(&config, vb)?;

    let raw = chatml(PROMPT);
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))?;
    let ids = tokenizer.encode(raw.as_str(), false)?.get_ids().to_vec();
    assert!(!ids.is_empty(), "prompt tokenized to nothing");

    let input = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;
    // seqlen_offset 0: full prefill; the model returns last-position logits
    // (batch and seq dims of 1) — flatten to the bare vocab vector.
    let logits = model.forward(&input, 0)?.flatten_all()?.to_vec1::<f32>()?;
    assert!(logits.len() > k, "vocab smaller than k");

    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
    let top: Vec<serde_json::Value> = indexed[..k]
        .iter()
        .map(|&(id, logit)| serde_json::json!({ "id": id, "logit": logit }))
        .collect();

    let out = serde_json::json!({
        "reference": "candle-0.9.1 cpu f32",
        "prompt_raw": raw,
        "prompt_ids": ids,
        "top_k": top,
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
