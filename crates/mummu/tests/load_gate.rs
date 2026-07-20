//! The checkpoint-metadata consistency gate is wired into the real
//! `load_from_dir` entry point — and fires *before* weight loading.
//!
//! Unit tests in `tok_config` prove the gate's logic in isolation; this proves
//! the loaders actually *call* it. The gate runs right after `config.json` is
//! parsed and before any weight bytes are read, so these tests need neither a
//! GPU nor real weights: a zero-byte `model.safetensors` satisfies the loader's
//! existence check, and deliberately-mismatched metadata makes the gate reject
//! the load with [`ImportError::Inconsistent`] before the empty weights are ever
//! touched.
//!
//! Coverage spans both halves of the gate: the tokenizer-free
//! `tokenizer_config.json` ↔ `config.json` checks (EOS agreement, tool-call
//! convention) and the tokenizer-opening id cross-check (every added-token id in
//! `tokenizer_config.json` must equal the id the sibling `tokenizer.json`
//! assigns that content).

use std::path::Path;

use mummu::backend::Cpu;
use mummu::import::ImportError;
use mummu::models::qwen3;
use tokenizers::Tokenizer;
use tokenizers::models::bpe::{BPE, Vocab};

/// A minimal but valid Qwen3 `config.json` (tiny dims — the model is never
/// built in the negative case; the gate rejects first). `eos_token_id` is 5.
const MIN_QWEN3_CONFIG: &str = r#"{
    "vocab_size": 10,
    "hidden_size": 4,
    "intermediate_size": 8,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 2,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_word_embeddings": true,
    "eos_token_id": 5
}"#;

/// Lay down a checkpoint dir: `config.json`, a zero-byte `model.safetensors`
/// (present, so the loader's `required_file` passes), and the given
/// `tokenizer_config.json` body.
fn make_dir(name: &str, tok_config: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp checkpoint dir");
    std::fs::write(dir.join("config.json"), MIN_QWEN3_CONFIG).expect("write config.json");
    std::fs::write(dir.join("model.safetensors"), b"").expect("write empty weights");
    std::fs::write(dir.join("tokenizer_config.json"), tok_config).expect("write tokenizer_config");
    dir
}

#[test]
fn load_from_dir_rejects_an_eos_mismatch_before_reading_weights() {
    // tokenizer_config resolves EOS to id 9; config.json says 5 → contradiction.
    let dir = make_dir(
        "mummu_load_gate_eos_mismatch",
        r#"{
            "eos_token": "<|end|>",
            "added_tokens_decoder": {"9": {"content": "<|end|>", "special": true}}
        }"#,
    );
    let device = burn::tensor::Device::<Cpu>::default();
    match qwen3::load_from_dir::<Cpu>(&dir, &device) {
        Err(ImportError::Inconsistent { reason, .. }) => {
            assert!(
                reason.contains('9'),
                "reason names the stray EOS id: {reason}"
            );
        }
        Err(other) => panic!("expected Inconsistent, got {other:?}"),
        Ok(_) => panic!("expected the gate to reject, but the load succeeded"),
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn load_from_dir_rejects_a_foreign_tool_call_convention() {
    // EOS agrees (id 5), but the template speaks LFM's bracket convention while
    // the Qwen3 loader's renderer is Hermes → the gate rejects.
    let dir = make_dir(
        "mummu_load_gate_convention_mismatch",
        r#"{
            "eos_token": "<|end|>",
            "chat_template": "…<|tool_call_start|>[f(x=1)]<|tool_call_end|>…",
            "added_tokens_decoder": {"5": {"content": "<|end|>", "special": true}}
        }"#,
    );
    let device = burn::tensor::Device::<Cpu>::default();
    match qwen3::load_from_dir::<Cpu>(&dir, &device) {
        Err(ImportError::Inconsistent { .. }) => {}
        Err(other) => panic!("expected Inconsistent for a foreign convention, got {other:?}"),
        Ok(_) => panic!("expected the gate to reject, but the load succeeded"),
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn load_from_dir_passes_the_gate_when_metadata_agrees() {
    // EOS agrees (id 5), Hermes template matches the loader's renderer → the
    // gate passes. The load then fails *later* on the empty weights, which is
    // exactly the point: the error is NOT an Inconsistent, proving the gate let
    // an agreeing checkpoint through.
    let dir = make_dir(
        "mummu_load_gate_agrees",
        r#"{
            "eos_token": "<|end|>",
            "chat_template": "…<tools>{{s}}</tools>…<tool_call>\n{j}\n</tool_call>…",
            "added_tokens_decoder": {"5": {"content": "<|end|>", "special": true}}
        }"#,
    );
    let device = burn::tensor::Device::<Cpu>::default();
    match qwen3::load_from_dir::<Cpu>(&dir, &device) {
        // The gate passed; the load then fails on the empty weights — any error
        // that is NOT Inconsistent proves the agreeing checkpoint cleared it.
        Err(ImportError::Inconsistent { .. }) => {
            panic!("an agreeing checkpoint must clear the gate, but it was rejected")
        }
        Err(
            ImportError::Load { .. } | ImportError::Incomplete { .. } | ImportError::Parse { .. },
        ) => {}
        Err(other) => panic!("expected a downstream weight error, got {other:?}"),
        Ok(_) => panic!("empty weights must fail the load after the gate"),
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Write a minimal but real `tokenizer.json` into `dir` whose model vocab maps
/// each `(token, id)` in `entries` — enough for `Tokenizer::token_to_id` (what
/// the gate's id cross-check calls) to resolve them. Loaded back by the real
/// `tokenizers` crate the loader uses, so this exercises the true path.
fn write_tokenizer_json(dir: &Path, entries: &[(&str, u32)]) {
    // A BPE model with an explicit vocab and no merges: `token_to_id` (what the
    // gate calls) resolves each entry straight from the vocab. Mirrors the
    // production `tokenizer_from_gguf` builder so it uses the same crate types.
    let mut vocab: Vocab = Vocab::default();
    vocab.insert("<unk>".to_string(), 0);
    for (token, id) in entries {
        vocab.insert((*token).to_string(), *id);
    }
    let model = BPE::builder()
        .vocab_and_merges(vocab, Vec::new())
        .unk_token("<unk>".to_string())
        .build()
        .expect("bpe model builds");
    let tok = Tokenizer::new(model);
    tok.save(dir.join("tokenizer.json"), false)
        .expect("write tokenizer.json");
}

#[test]
fn load_from_dir_rejects_an_added_token_id_mismatch_before_reading_weights() {
    // EOS agrees (id 5) and there is no chat template, so the tokenizer-free
    // checks pass — the *id cross-check* is what must fire. tokenizer_config
    // declares <|extra|> at id 6, but the real tokenizer.json puts it at 999.
    let dir = make_dir(
        "mummu_load_gate_added_id_mismatch",
        r#"{
            "eos_token": "<|end|>",
            "added_tokens_decoder": {
                "5": {"content": "<|end|>", "special": true},
                "6": {"content": "<|extra|>", "special": true}
            }
        }"#,
    );
    write_tokenizer_json(&dir, &[("<|end|>", 5), ("<|extra|>", 999)]);
    let device = burn::tensor::Device::<Cpu>::default();
    match qwen3::load_from_dir::<Cpu>(&dir, &device) {
        Err(ImportError::Inconsistent { reason, file }) => {
            assert!(
                file.ends_with("tokenizer.json"),
                "the id cross-check names tokenizer.json: {}",
                file.display()
            );
            assert!(
                reason.contains("<|extra|>") && reason.contains("999"),
                "reason names the disagreeing token and the tokenizer's id: {reason}"
            );
        }
        Err(other) => panic!("expected Inconsistent from the id cross-check, got {other:?}"),
        Ok(_) => panic!("expected the gate to reject, but the load succeeded"),
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn load_from_dir_passes_the_gate_when_tokenizer_ids_agree() {
    // Same declared ids, but now the real tokenizer.json agrees on both — the id
    // cross-check clears and the load proceeds to fail on the empty weights (an
    // error that is NOT Inconsistent proves the agreeing tokenizer let it past).
    let dir = make_dir(
        "mummu_load_gate_added_id_agrees",
        r#"{
            "eos_token": "<|end|>",
            "added_tokens_decoder": {
                "5": {"content": "<|end|>", "special": true},
                "6": {"content": "<|extra|>", "special": true}
            }
        }"#,
    );
    write_tokenizer_json(&dir, &[("<|end|>", 5), ("<|extra|>", 6)]);
    let device = burn::tensor::Device::<Cpu>::default();
    match qwen3::load_from_dir::<Cpu>(&dir, &device) {
        Err(ImportError::Inconsistent { reason, .. }) => {
            panic!("agreeing tokenizer ids must clear the gate, but it was rejected: {reason}")
        }
        Err(
            ImportError::Load { .. } | ImportError::Incomplete { .. } | ImportError::Parse { .. },
        ) => {}
        Err(other) => panic!("expected a downstream weight error, got {other:?}"),
        Ok(_) => panic!("empty weights must fail the load after the gate"),
    }
    std::fs::remove_dir_all(&dir).ok();
}
