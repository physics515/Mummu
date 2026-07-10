//! LFM2.5 parity: greedy sequence vs a local Ollama `lfm2.5` reference (the
//! same reference laurelane validated against). Ignored by default; run with
//!
//! ```text
//! MUMMU_LFM2_DIR=path/to/lfm2.5-1.2b cargo test -p mummu --release --test parity_lfm2 -- --ignored --nocapture
//! ```
//!
//! Needs a running Ollama whose `lfm2.5` tag is the SAME weights (the 1.2B
//! dense instruct). As of 2026-07-09 `lfm2.5:latest` resolves to the 8.5B
//! **MoE** at Q4_K_M with thinking — a different model, so this test fails
//! against it by design (fail loudly, never compare across weights). This is
//! the greedy leg of the P7 parity gate; the top-k-logits leg needs a
//! reference that exposes logits (Candle) and stays a P7 to-do.

use std::path::PathBuf;
use std::process::Command;

use mummu::backend::Gpu;
use mummu::models::lfm2;
use tokenizers::Tokenizer;

const PROMPT: &str = "List the first five prime numbers.";
const MAX_TOKENS: usize = 24;

fn lfm2_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("MUMMU_LFM2_DIR")?);
    dir.is_dir().then_some(dir)
}

/// The LFM2.5 ChatML wrapping (BOS + user turn + assistant open), rendered by
/// the library's own template (the same shape laurelane validated vs Ollama).
fn chatml(user: &str) -> String {
    mummu::chat::ChatMl::lfm2().render(&[mummu::chat::Turn::user(user)])
}

/// Greedy reference from Ollama in raw mode (identical prompt text,
/// temperature 0), via curl so the test needs no HTTP dependency.
fn ollama_greedy(raw_prompt: &str, max_tokens: usize) -> Result<String, String> {
    let body = serde_json::json!({
        "model": "lfm2.5",
        "prompt": raw_prompt,
        "raw": true,
        "stream": false,
        "options": { "temperature": 0.0, "num_predict": max_tokens }
    });
    let out = Command::new("curl")
        .args([
            "-s",
            "-X",
            "POST",
            "http://localhost:11434/api/generate",
            "-d",
        ])
        .arg(body.to_string())
        .output()
        .map_err(|e| format!("curl spawn: {e}"))?;
    if !out.status.success() {
        return Err(format!("curl exit {:?}", out.status.code()));
    }
    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).map_err(|e| format!("ollama response parse: {e}"))?;
    v.get("response")
        .and_then(|r| r.as_str())
        .map(str::to_string)
        .ok_or_else(|| format!("no response field in: {v}"))
}

#[test]
#[ignore = "needs local LFM2.5 weights (MUMMU_LFM2_DIR) + a running Ollama with lfm2.5"]
fn lfm2_greedy_sequence_matches_ollama_reference() {
    let Some(dir) = lfm2_dir() else {
        panic!("set MUMMU_LFM2_DIR to a dir with config.json/tokenizer.json/model.safetensors");
    };
    let raw = chatml(PROMPT);

    // Reference first: skip cleanly (not red) when Ollama isn't up.
    let reference = match ollama_greedy(&raw, MAX_TOKENS) {
        Ok(r) => r,
        Err(e) => panic!("Ollama reference unavailable: {e}"),
    };

    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json loads");
    let prompt_ids = tok
        .encode(raw.as_str(), false)
        .expect("encodes")
        .get_ids()
        .to_vec();

    let device = burn::tensor::Device::<Gpu>::default();
    let loaded = lfm2::load_from_dir::<Gpu>(&dir, &device).expect("weights load checked");
    let ids = loaded
        .greedy_generate(&prompt_ids, MAX_TOKENS, &device)
        .expect("greedy decode");
    let ours = tok.decode(&ids, true).expect("decode");

    eprintln!("[parity/lfm2] ours     ({} tokens): {ours:?}", ids.len());
    eprintln!("[parity/lfm2] ollama              : {reference:?}");

    // Exact-match the greedy prefix over the shorter of the two (EOS/num_predict
    // may truncate one side earlier than the other).
    let n = ours.trim().len().min(reference.trim().len());
    assert!(n >= 8, "outputs too short to compare: {n} chars");
    assert_eq!(
        &ours.trim()[..n],
        &reference.trim()[..n],
        "greedy sequences diverge"
    );
}
