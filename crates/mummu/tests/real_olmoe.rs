//! Real-weights validation of the OLMoE-1B-7B port (P2's first MoE): the
//! registry spec fetches the official allenai Q4_K_M GGUF, the ONE file
//! yields config + tokenizer + weights, and the model loads and greedy-
//! decodes coherently on the **CPU** backend — the resident-everything first
//! cut dequantizes ~7B params to ~28 GB of f32, which is CPU-RAM territory
//! (the 128 GB reference machine), not 16 GB-card territory. Ignored by
//! default; run with
//!
//! ```text
//! MUMMU_HUB_DEST=path/to/models-cache \
//! MUMMU_OLMOE_TOK_JSON=path/to/olmoe/tokenizer.json \
//!   cargo test -p mummu --release --test real_olmoe -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mummu::backend::Cpu;
use mummu::gguf::{GgufFile, GgufValue};
use mummu::models::CausalLm;
use mummu::models::olmoe;

fn hub_dest() -> PathBuf {
    std::env::var_os("MUMMU_HUB_DEST")
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set MUMMU_HUB_DEST to the models dir for the ~4.2 GB download"))
}

fn fetch_olmoe() -> PathBuf {
    let dest = hub_dest();
    let spec = mummu::registry::catalog()
        .into_iter()
        .find(|s| s.name == "olmoe-1b-7b-0125-instruct-q4km")
        .expect("the OLMoE GGUF is in the catalog");
    spec.fetch(&dest, |_| {}).expect("registry fetch");
    let path = spec.gguf_path(&dest).expect("gguf specs have a file path");
    assert!(path.is_file(), "downloaded file exists at {path:?}");
    path
}

/// The GGUF-built tokenizer must byte-match the checkpoint's own HF
/// `tokenizer.json` (fetched separately — the GGUF repo does not ship it).
#[test]
#[ignore = "needs network (MUMMU_HUB_DEST) + the HF tokenizer.json (MUMMU_OLMOE_TOK_JSON)"]
fn olmoe_gguf_tokenizer_matches_the_hf_tokenizer() {
    let Some(tok_json) = std::env::var_os("MUMMU_OLMOE_TOK_JSON").map(PathBuf::from) else {
        panic!("set MUMMU_OLMOE_TOK_JSON to the checkpoint's tokenizer.json");
    };
    let path = fetch_olmoe();
    let f = GgufFile::open(&path).expect("valid GGUF");
    let ours = mummu::tokenizer::tokenizer_from_gguf(&f).expect("tokenizer from metadata");
    let hf = tokenizers::Tokenizer::from_file(&tok_json).expect("hf tokenizer.json loads");

    // The battery the other GGUF tokenizer gates use: template-ish specials,
    // unicode/CJK/emoji, whitespace runs (OLMoE has dedicated multi-space
    // added tokens), contractions (case-sensitive in this family), empty.
    let battery = [
        "<|user|>\nWhat is 2 + 2?\n<|assistant|>\n",
        "The quick brown fox jumps over the lazy dog.",
        "I'll say it's DONE — they're not.",
        "code:\n    if x  <  3   {\n        return;\n    }\n",
        "日本語のテキストと emoji 🦉🌲 mixed in.",
        "naïve café résumé ™ ½ ﬁligree",
        "|||EMAIL_ADDRESS||| and |||IP_ADDRESS||| specials",
        "",
    ];
    for prompt in battery {
        let a = ours.encode(prompt, false).expect("ours encodes");
        let b = hf.encode(prompt, false).expect("hf encodes");
        assert_eq!(
            a.get_ids(),
            b.get_ids(),
            "ids diverge on {prompt:?}:\n ours {:?}\n hf   {:?}",
            a.get_tokens(),
            b.get_tokens()
        );
        let ra = ours.decode(a.get_ids(), false).expect("ours decodes");
        let rb = hf.decode(b.get_ids(), false).expect("hf decodes");
        assert_eq!(ra, rb, "decode round-trips diverge on {prompt:?}");
    }
    eprintln!(
        "[real_olmoe] tokenizer byte-identical to tokenizer.json on {} prompts",
        battery.len()
    );
}

/// The end-to-end proof: the ONE .gguf file loads (checked, every tensor
/// mapped) and greedy-decodes a correct answer through the MoE stack on the
/// CPU backend.
#[test]
#[ignore = "needs network (MUMMU_HUB_DEST; ~4.2 GB), ~30 GB free COMMIT and ~28 GB \
            of scratch disk beside the gguf (CPU backend)"]
fn olmoe_gguf_loads_and_decodes_on_cpu() {
    let path = fetch_olmoe();
    let f = GgufFile::open(&path).expect("valid GGUF");
    assert_eq!(f.architecture(), Some("olmoe"), "wrong architecture");
    let cfg = olmoe::OlmoeConfig::from_gguf(&f).expect("config from metadata");
    assert_eq!(cfg.num_experts, 64, "1B-7B has 64 experts");
    assert_eq!(cfg.num_experts_per_tok, 8, "1B-7B routes top-8");
    let tok = mummu::tokenizer::tokenizer_from_gguf(&f).expect("tokenizer from metadata");
    // The zephyr-style template from the model card, BOS first.
    let bos = f
        .get("tokenizer.ggml.bos_token_id")
        .and_then(GgufValue::as_u64)
        .and_then(|v| {
            f.get("tokenizer.ggml.tokens")
                .and_then(GgufValue::as_array)
                .and_then(|t| t.get(usize::try_from(v).ok()?))
                .and_then(GgufValue::as_str)
                .map(String::from)
        })
        .unwrap_or_default();
    drop(f);

    let raw =
        format!("{bos}<|user|>\nWhat is 2 + 2? Answer in one short sentence.\n<|assistant|>\n");
    let prompt = tok
        .encode(raw.as_str(), false)
        .expect("encodes")
        .get_ids()
        .to_vec();
    assert!(prompt.len() >= 8, "rendered prompt suspiciously short");

    let device = burn::tensor::Device::<Cpu>::default();
    let start = std::time::Instant::now();
    let loaded = olmoe::load_from_gguf::<Cpu>(&path, &device).expect("checked load");
    eprintln!(
        "[real_olmoe] loaded {} layers x {} experts in {:.1}s",
        loaded.config.num_hidden_layers,
        loaded.config.num_experts,
        start.elapsed().as_secs_f64()
    );

    // Liveness first: finite, vocab-wide, non-degenerate logits.
    let smoke = loaded
        .sanity_check(&prompt, loaded.config.vocab_size, &device)
        .expect("sanity smoke");
    eprintln!("[real_olmoe] sanity: {smoke:?}");

    let start = std::time::Instant::now();
    let ids = loaded
        .greedy_generate(&prompt, 24, &device)
        .expect("greedy decode");
    let secs = start.elapsed().as_secs_f64();
    assert!(!ids.is_empty(), "decode produced no tokens before EOS");
    let text = tok.decode(&ids, true).expect("decode");
    eprintln!(
        "[real_olmoe] {} tokens in {secs:.1}s ({:.2} s/token): {text:?}",
        ids.len(),
        secs / ids.len() as f64
    );
    assert!(
        text.contains('4') || text.to_lowercase().contains("four"),
        "expected the answer to mention 4, got: {text:?}"
    );
}
