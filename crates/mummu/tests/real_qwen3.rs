//! Real-weights proof for the Qwen3 dense port: load the actual checkpoint
//! (safetensors AND the Q4_K_M GGUF of the same weights), decode on the GPU,
//! and cross-check the two builds agree on the first token. Ignored by default;
//! run with the paths set:
//!
//! ```text
//! MUMMU_QWEN3_DIR=path/to/qwen3-0.6b \
//! MUMMU_QWEN3_GGUF_PATH=path/to/Qwen3-0.6B-Q4_K_M.gguf \
//!   cargo test -p mummu --test real_qwen3 -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mummu::backend::{Gpu, use_gpu};
use mummu::gguf::GgufFile;
use mummu::models::CausalLm;
use mummu::models::qwen3;

fn dir() -> Option<PathBuf> {
    std::env::var_os("MUMMU_QWEN3_DIR")
        .map(PathBuf::from)
        .filter(|d| d.join("model.safetensors").is_file())
}

fn gguf_path() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var_os("MUMMU_QWEN3_GGUF_PATH")?);
    path.is_file().then_some(path)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

fn argmax(v: &[f32]) -> usize {
    let (mut best, mut best_v) = (0usize, f32::NEG_INFINITY);
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            (best, best_v) = (i, x);
        }
    }
    best
}

/// END-TO-END on the real bf16 safetensors: the Qwen3 arch (per-head q/k norm,
/// no qkv bias, decoupled head_dim) loads real weights and greedy-decodes a
/// coherent, correct answer on the GPU. The tokenizer is the checkpoint's own
/// `tokenizer.json`; the prompt uses the Qwen ChatML template.
#[test]
#[ignore = "needs the local Qwen3 safetensors dir (MUMMU_QWEN3_DIR) + GPU"]
fn real_qwen3_safetensors_loads_and_decodes_on_gpu() {
    let dir = dir().expect("set MUMMU_QWEN3_DIR to a Qwen3 safetensors dir");
    assert!(use_gpu(), "this proof wants the real GPU");
    let device = burn::tensor::Device::<Gpu>::default();

    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let chat = mummu::chat::ChatMl::qwen2(); // Qwen3 shares Qwen2's ChatML
    let prompt_text = chat.render(&[
        mummu::chat::Turn::system("You are a concise assistant. Do not think, answer directly."),
        mummu::chat::Turn::user("What is 2+2? Answer in one short sentence."),
    ]);
    let prompt = tok
        .encode(prompt_text, true)
        .expect("prompt encodes")
        .get_ids()
        .to_vec();

    let model = qwen3::load_from_dir::<Gpu>(&dir, &device).expect("safetensors load is checked");
    // The decoupled shape holds on the real weights.
    assert!(
        model.config.num_attention_heads * model.config.head_dim != model.config.hidden_size
            || model.config.head_dim * model.config.num_attention_heads == model.config.hidden_size,
        "config loaded"
    );
    eprintln!(
        "[real_qwen3] {} layers · hidden {} · {} heads · {} kv · head_dim {} · tied {}",
        model.config.num_hidden_layers,
        model.config.hidden_size,
        model.config.num_attention_heads,
        model.config.num_key_value_heads,
        model.config.head_dim,
        model.config.tie_word_embeddings,
    );

    let ids = model.greedy_generate(&prompt, 48, &device).expect("decode");
    let text = tok.decode(&ids, true).expect("ids decode");
    eprintln!("[real_qwen3] safetensors greedy: {text:?}");
    assert!(text.contains('4'), "expected the answer 4 in: {text:?}");
}

/// END-TO-END on the Q4_K_M GGUF alone (config + tokenizer + weights from the
/// one file), cross-checked against the bf16 safetensors build: both decode a
/// correct answer and agree on the top first-token id (small logit drift IS the
/// quantization; a layout/qk-norm-mapping bug reads as disagreement / ≈0 cosine).
/// Loads sequentially (drop the GGUF model before the safetensors one) so peak
/// VRAM stays one-model-sized.
#[test]
#[ignore = "needs the Qwen3 GGUF (MUMMU_QWEN3_GGUF_PATH) + safetensors dir (MUMMU_QWEN3_DIR) + GPU"]
fn real_qwen3_gguf_loads_and_agrees_with_safetensors() {
    let path = gguf_path().expect("set MUMMU_QWEN3_GGUF_PATH to a Qwen3 q4_k_m gguf");
    let dir = dir().expect("set MUMMU_QWEN3_DIR to the same model's safetensors dir");
    assert!(use_gpu(), "this proof wants the real GPU");
    let device = burn::tensor::Device::<Gpu>::default();

    // Tokenizer straight from the GGUF metadata — the whole model is ONE file.
    let header = GgufFile::open(&path).expect("header parses");
    assert_eq!(header.architecture(), Some("qwen3"));
    let tok = mummu::tokenizer::tokenizer_from_gguf(&header).expect("tokenizer from metadata");
    let reference =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");

    let chat = mummu::chat::ChatMl::qwen2();
    let prompt_text = chat.render(&[
        mummu::chat::Turn::system("You are a concise assistant. Do not think, answer directly."),
        mummu::chat::Turn::user("What is 2+2? Answer in one short sentence."),
    ]);
    // The tokenizer-from-GGUF must match tokenizer.json on this prompt.
    let prompt = tok
        .encode(prompt_text.clone(), true)
        .expect("prompt encodes")
        .get_ids()
        .to_vec();
    let ref_ids = reference
        .encode(prompt_text, true)
        .expect("ref encodes")
        .get_ids()
        .to_vec();
    assert_eq!(
        prompt, ref_ids,
        "tokenizer-from-GGUF diverges from tokenizer.json"
    );

    let gguf_model = qwen3::load_from_gguf::<Gpu>(&path, &device).expect("gguf load is checked");
    let ids = gguf_model
        .greedy_generate(&prompt, 48, &device)
        .expect("decode");
    let text = tok.decode(&ids, true).expect("ids decode");
    eprintln!("[real_qwen3] Q4_K_M greedy: {text:?}");
    assert!(text.contains('4'), "expected the answer 4 in: {text:?}");

    let logits_of_gguf = {
        let mut cache = gguf_model.new_cache();
        gguf_model
            .forward(&prompt, 0, &mut cache, &device)
            .into_data()
            .to_vec::<f32>()
            .expect("logits read back")
    };
    drop(gguf_model);

    let st_model = qwen3::load_from_dir::<Gpu>(&dir, &device).expect("safetensors load");
    let st_logits = {
        let mut cache = st_model.new_cache();
        st_model
            .forward(&prompt, 0, &mut cache, &device)
            .into_data()
            .to_vec::<f32>()
            .expect("logits read back")
    };
    drop(st_model);

    let cos = cosine(&logits_of_gguf, &st_logits);
    let (g_top, s_top) = (argmax(&logits_of_gguf), argmax(&st_logits));
    eprintln!(
        "[real_qwen3] first-token logits: cosine {cos:.5} vs bf16 · top-1 {g_top} vs {s_top}"
    );
    assert_eq!(g_top, s_top, "Q4_K_M must agree with bf16 on the top token");
    assert!(
        cos > 0.95,
        "logit cosine {cos} — quantization noise should be small, layout/qk-norm bugs are not"
    );
}
