//! Real-file proof for the `tokenizer_config.json` importer: parse an actual
//! checkpoint's config and cross-check the imported special-token ids against
//! the sibling `tokenizer.json` that the `tokenizers` crate loads. This is the
//! runtime check a synthetic unit test can't make — that the config's declared
//! ids agree with the real tokenizer's vocab. Ignored by default; run with:
//!
//! ```text
//! MUMMU_QWEN3_DIR=path/to/qwen3-0.6b \
//!   cargo test -p mummu --test real_tokenizer_config -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mummu::tok_config::{TokenizerConfig, ToolCallConvention};
use tokenizers::Tokenizer;

fn dir() -> Option<PathBuf> {
    std::env::var_os("MUMMU_QWEN3_DIR")
        .map(PathBuf::from)
        .filter(|d| d.join("tokenizer_config.json").is_file())
}

#[test]
#[ignore = "needs the local Qwen3 checkpoint dir (MUMMU_QWEN3_DIR)"]
fn qwen3_config_special_ids_agree_with_the_tokenizer() {
    let dir = dir().expect("set MUMMU_QWEN3_DIR to a Qwen3 checkpoint dir");
    let cfg = TokenizerConfig::from_dir(&dir).expect("tokenizer_config.json parses");

    // Qwen3 declares ChatML's <|im_end|> as EOS and <|endoftext|> as PAD, with
    // no BOS prepending — imported straight from the config.
    let eos = cfg.eos_token.as_ref().expect("eos declared");
    assert_eq!(eos.content, "<|im_end|>", "Qwen3 EOS is <|im_end|>");
    assert_eq!(
        cfg.eos_id(),
        Some(151_645),
        "EOS id from added_tokens_decoder"
    );
    let pad = cfg.pad_token.as_ref().expect("pad declared");
    assert_eq!(pad.content, "<|endoftext|>");
    assert_eq!(cfg.pad_id(), Some(151_643));
    assert!(!cfg.add_bos_token, "Qwen3 does not prepend BOS");

    // The embedded chat template is imported and speaks ChatML — the same
    // markers our byte-verified chat::ChatMl::qwen2() renderer emits.
    let template = cfg.chat_template.as_deref().expect("chat template present");
    assert!(template.contains("<|im_start|>"), "ChatML start marker");
    assert!(template.contains("<|im_end|>"), "ChatML end marker");

    // Qwen3 ships a Hermes tool-calling template — detected from its markers,
    // so an app can pick chat's Hermes render style without hardcoding it.
    assert_eq!(
        cfg.tool_call_convention(),
        Some(ToolCallConvention::Hermes),
        "Qwen3 template is Hermes (<tool_call>/<tools>)"
    );

    // THE cross-check: every resolved special id must equal what the real
    // tokenizer.json assigns that content. Config and tokenizer agree.
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json loads");
    for st in [eos, pad] {
        let via_tokenizer = tok.token_to_id(&st.content);
        assert_eq!(
            via_tokenizer, st.id,
            "config id for {:?} ({:?}) must match the tokenizer's id ({:?})",
            st.content, st.id, via_tokenizer
        );
    }

    // And the library validator agrees on every added-token id (this is the
    // same cross-check, now a first-class `tok_config` function a loader calls).
    cfg.check_ids_against(|t| tok.token_to_id(t))
        .expect("every added-token id agrees with tokenizer.json");

    // config.json ↔ tokenizer_config.json EOS agreement: the id config.json's
    // eos_token_id names must be the id tokenizer_config's eos_token resolves to.
    let config_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).expect("config.json"))
            .expect("config.json parses");
    let config_eos: Vec<u32> = match &config_json["eos_token_id"] {
        serde_json::Value::Number(n) => vec![n.as_u64().unwrap() as u32],
        serde_json::Value::Array(a) => a
            .iter()
            .filter_map(|v| v.as_u64().map(|x| x as u32))
            .collect(),
        _ => Vec::new(),
    };
    cfg.check_eos_agrees(&config_eos)
        .expect("config.json eos_token_id agrees with tokenizer_config.json eos_token");

    println!(
        "qwen3 tokenizer_config: eos={:?}({:?}) pad={:?}({:?}) added={} chat_template={}B — all ids agree",
        eos.content,
        cfg.eos_id(),
        pad.content,
        cfg.pad_id(),
        cfg.added_tokens.len(),
        template.len(),
    );
}
