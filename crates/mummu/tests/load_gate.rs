//! The `tokenizer_config.json` consistency gate is wired into the real
//! `load_from_dir` entry point — and fires *before* weight loading.
//!
//! Unit tests in `tok_config` prove the gate's logic in isolation; this proves
//! the loaders actually *call* it. The gate runs right after `config.json` is
//! parsed and before any weight bytes are read, so these tests need neither a
//! GPU nor real weights: a zero-byte `model.safetensors` satisfies the loader's
//! existence check, and a deliberately-mismatched `tokenizer_config.json` makes
//! the gate reject the load with [`ImportError::Inconsistent`] before the empty
//! weights are ever touched.

use mummu::backend::Cpu;
use mummu::import::ImportError;
use mummu::models::qwen3;

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
