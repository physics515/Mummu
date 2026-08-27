//! Real-file GGUF header proof: parse an actual llama.cpp-quantized model
//! (Qwen2.5-1.5B-Instruct Q4_K_M) and check the header describes the model
//! we know. Ignored by default; run with
//!
//! ```text
//! MUMMU_GGUF_PATH=path/to/qwen2.5-1.5b-instruct-q4_k_m.gguf cargo test -p mummu --test real_gguf -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mummu::attn_config::{RopeScaling, sliding_window_from_gguf};
use mummu::backend::use_gpu;
use mummu::gguf::{GgmlType, GgufFile, GgufValue};
use mummu::models::CausalLm;
use mummu::models::qwen2;

fn gguf_path() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var_os("MUMMU_GGUF_PATH")?);
    path.is_file().then_some(path)
}

#[test]
#[ignore = "needs a local GGUF model file (MUMMU_GGUF_PATH)"]
fn real_qwen2_gguf_header_parses_and_describes_the_model() {
    let Some(path) = gguf_path() else {
        panic!("set MUMMU_GGUF_PATH to a local .gguf model file");
    };
    let f = GgufFile::open(&path).expect("header parses");

    eprintln!(
        "[real_gguf] v{} · {} kvs · {} tensors · align {} · data at {}",
        f.version,
        f.metadata.len(),
        f.tensors.len(),
        f.alignment,
        f.data_offset
    );
    assert_eq!(f.architecture(), Some("qwen2"));

    // The tokenizer vocab must be a real array of strings.
    let tokens = f
        .get("tokenizer.ggml.tokens")
        .and_then(GgufValue::as_array)
        .expect("vocab array present");
    assert!(
        tokens.len() > 100_000,
        "Qwen vocab is ~152k: {}",
        tokens.len()
    );

    // The embedding tensor exists with the config's dims (ggml order:
    // fastest-varying first — [hidden, vocab]).
    let embd = f.tensor("token_embd.weight").expect("embedding present");
    assert_eq!(embd.dims[0], 1536, "Qwen2.5-1.5B hidden size");
    assert_eq!(embd.dims[1] as usize, tokens.len(), "vocab rows");

    // Every tensor: known dtype (the parser guarantees it), aligned offset,
    // whole blocks; and at least one K-quant tensor is actually present.
    for t in &f.tensors {
        assert!(t.offset.is_multiple_of(f.alignment), "{}", t.name);
        assert!(t.byte_len() > 0, "{}", t.name);
    }
    let kquants = f
        .tensors
        .iter()
        .filter(|t| matches!(t.dtype, GgmlType::Q4_K | GgmlType::Q6_K))
        .count();
    assert!(kquants > 0, "a q4_k_m file carries K-quant tensors");
    let payload_bytes: u64 = f.tensors.iter().map(|t| t.byte_len()).sum();
    eprintln!(
        "[real_gguf] token_embd {:?} {:?} · {} K-quant tensors · payload ~{:.2} GiB",
        embd.dtype,
        embd.dims,
        kquants,
        payload_bytes as f64 / f64::from(1u32 << 30)
    );
}

/// The attention-shaping half of the same header: a llama.cpp GGUF declares
/// `<arch>.rope.scaling.*` and `<arch>.attention.sliding_window` only for
/// models that use them, so the zoo's own files must parse to "no scaling, no
/// window" — and `Qwen2Config::from_gguf` must accept them. Header-only, so it
/// needs no GPU and no dequantization.
#[test]
#[ignore = "needs a local GGUF model file (MUMMU_GGUF_PATH)"]
fn real_qwen2_gguf_header_declares_no_rope_scaling_and_no_window() {
    let Some(path) = gguf_path() else {
        panic!("set MUMMU_GGUF_PATH to a local .gguf model file");
    };
    let f = GgufFile::open(&path).expect("header parses");

    let scaling = RopeScaling::from_gguf(&f, "qwen2");
    let window = sliding_window_from_gguf(&f, "qwen2");
    eprintln!("[real_gguf] rope.scaling = {scaling:?} · attention.sliding_window = {window:?}");
    assert!(
        scaling.is_none(),
        "an unscaled checkpoint must declare no rope.scaling.* keys, got {scaling:?}"
    );
    assert!(
        window.is_none(),
        "Qwen2.5 GGUFs declare no window, got {window:?}"
    );

    // And the whole config still parses through the new validation.
    let cfg = qwen2::Qwen2Config::from_gguf(&f).expect("gguf config parses");
    assert!(cfg.rope_scaling.is_none() && cfg.sliding_window.is_none());
    assert_eq!(cfg.hidden_size, 1536, "Qwen2.5-1.5B hidden size");
    eprintln!(
        "[real_gguf] context_length = {:?} · theta {}",
        cfg.max_position_embeddings, cfg.rope_theta
    );
}

/// Minimal safetensors reader for the cross-check: header JSON + raw bf16
/// tensor bytes widened to f32.
fn safetensors_bf16_f32(path: &std::path::Path, name: &str) -> Vec<f32> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path).expect("safetensors opens");
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes).expect("header length");
    let header_len = u64::from_le_bytes(len_bytes);
    let mut header = vec![0u8; usize::try_from(header_len).expect("sane header")];
    file.read_exact(&mut header).expect("header json");
    let header: serde_json::Value = serde_json::from_slice(&header).expect("header parses");
    let entry = header.get(name).unwrap_or_else(|| panic!("{name} present"));
    assert_eq!(entry["dtype"], "BF16", "cross-check expects bf16 weights");
    let start = entry["data_offsets"][0].as_u64().expect("start offset");
    let end = entry["data_offsets"][1].as_u64().expect("end offset");
    file.seek(SeekFrom::Start(8 + header_len + start))
        .expect("seek to tensor");
    let mut raw = vec![0u8; usize::try_from(end - start).expect("sane tensor")];
    file.read_exact(&mut raw).expect("tensor bytes");
    let (pairs, rest) = raw.as_chunks::<2>();
    assert!(rest.is_empty(), "bf16 tensor byte length must be even");
    pairs
        .iter()
        .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
        .collect()
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

/// Dequantization proof against the model's TRUE weights: the same
/// checkpoint exists here as bf16 safetensors and as a Q4_K_M GGUF, so
/// - the GGUF's F32 norm tensors must equal the bf16 originals EXACTLY
///   (bf16 → f32 widening is lossless and that is how llama.cpp converts);
/// - a dequantized Q4_K embedding row must land within quantization error
///   of the original (cosine ≈ 1; garbage layout decode would be ≈ 0).
#[test]
#[ignore = "needs the local GGUF (MUMMU_GGUF_PATH) + safetensors (MUMMU_QWEN2_DIR) of the same model"]
fn real_qwen2_gguf_dequant_matches_the_true_weights() {
    let Some(path) = gguf_path() else {
        panic!("set MUMMU_GGUF_PATH to the qwen2.5-1.5b-instruct q4_k_m gguf");
    };
    let st_dir = std::env::var_os("MUMMU_QWEN2_DIR").map(PathBuf::from);
    let Some(st) = st_dir.filter(|d| d.join("model.safetensors").is_file()) else {
        panic!("set MUMMU_QWEN2_DIR to the same model's safetensors dir");
    };
    let st = st.join("model.safetensors");
    let f = GgufFile::open(&path).expect("header parses");

    // Leg 1: F32 norm — exact.
    let ours = f
        .read_tensor_f32("output_norm.weight")
        .expect("norm dequantizes");
    let reference = safetensors_bf16_f32(&st, "model.norm.weight");
    assert_eq!(ours.len(), reference.len(), "same norm size");
    let exact = ours
        .iter()
        .zip(&reference)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    assert!(exact, "GGUF F32 norm must be the bf16 weights, bit-exact");
    eprintln!(
        "[real_gguf] output_norm.weight: {} f32 values bit-exact vs safetensors",
        ours.len()
    );

    // Leg 2: Q4_K embedding rows — within quantization error of the truth.
    let embd = f.read_tensor_f32("token_embd.weight").expect("dequantizes");
    let truth = safetensors_bf16_f32(&st, "model.embed_tokens.weight");
    assert_eq!(embd.len(), truth.len(), "same embedding size");
    let hidden = 1536;
    for row in [9707usize, 100_000] {
        let ours_row = &embd[row * hidden..(row + 1) * hidden];
        let true_row = &truth[row * hidden..(row + 1) * hidden];
        let cos = cosine(ours_row, true_row);
        eprintln!("[real_gguf] Q4_K embd row {row}: cosine {cos:.5} vs bf16 truth");
        assert!(
            cos > 0.97,
            "row {row}: cosine {cos} — layout decode is wrong"
        );
    }
}

/// The tokenizer rebuilt from GGUF metadata must be **byte-identical** to the
/// checkpoint's own `tokenizer.json` — same ids for every prompt shape we
/// throw at it (ChatML with specials, unicode, numbers, whitespace runs), and
/// the same decoded text back.
#[test]
#[ignore = "needs the local GGUF (MUMMU_GGUF_PATH) + safetensors dir (MUMMU_QWEN2_DIR)"]
fn real_qwen2_gguf_tokenizer_matches_the_hf_tokenizer() {
    let Some(path) = gguf_path() else {
        panic!("set MUMMU_GGUF_PATH to the qwen2.5-1.5b-instruct q4_k_m gguf");
    };
    let dir = std::env::var_os("MUMMU_QWEN2_DIR")
        .map(PathBuf::from)
        .filter(|d| d.join("tokenizer.json").is_file())
        .expect("set MUMMU_QWEN2_DIR to the same model's safetensors dir (tokenizer.json)");

    let f = GgufFile::open(&path).expect("header parses");
    let ours = mummu::tokenizer::tokenizer_from_gguf(&f).expect("tokenizer builds from metadata");
    let reference =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");

    let chat = mummu::chat::ChatMl::qwen2();
    let battery = [
        chat.render(&[
            mummu::chat::Turn::system("You are a concise assistant."),
            mummu::chat::Turn::user("What is 2+2? Answer in one short sentence."),
        ]),
        "The quick brown fox jumps over the lazy dog.".into(),
        "héllo wörld — 世界 · Ω ≠ ω · 🦀🔥".into(),
        "  leading spaces, trailing  \n\nnewlines\r\nand\ttabs ".into(),
        "1234567890 3.14159 1e-6 0x7F".into(),
        "<|im_start|>user\nplain<|im_end|><|endoftext|>".into(),
        "don't they're we've I'll it's CAN'T".into(),
        String::new(),
    ];
    for text in &battery {
        let a = ours.encode(text.as_str(), true).expect("ours encodes");
        let b = reference.encode(text.as_str(), true).expect("ref encodes");
        assert_eq!(
            a.get_ids(),
            b.get_ids(),
            "token ids diverge on {text:?}: {:?} vs {:?}",
            a.get_tokens(),
            b.get_tokens()
        );
        let da = ours.decode(a.get_ids(), false).expect("ours decodes");
        let db = reference.decode(b.get_ids(), false).expect("ref decodes");
        assert_eq!(da, db, "decoded text diverges on {text:?}");
    }
    eprintln!(
        "[real_gguf] tokenizer-from-GGUF: {} prompts byte-identical to tokenizer.json",
        battery.len()
    );
}

// ---- LFM2.5 (the hybrid conv+attention architecture) ----------------------

fn lfm2_gguf_path() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var_os("MUMMU_LFM2_GGUF_PATH")?);
    path.is_file().then_some(path)
}

fn lfm2_dir() -> Option<PathBuf> {
    std::env::var_os("MUMMU_LFM2_DIR")
        .map(PathBuf::from)
        .filter(|d| d.join("model.safetensors").is_file())
}

/// LFM2.5 mapping proof against the TRUE weights: every F32 tensor in the
/// GGUF (norms, per-head q/k norms, and the depthwise conv kernels — the
/// reshape special-case) must equal the bf16 safetensors bit-exactly.
#[test]
#[ignore = "needs the local LFM2.5 GGUF (MUMMU_LFM2_GGUF_PATH) + safetensors (MUMMU_LFM2_DIR)"]
fn real_lfm2_gguf_f32_tensors_match_the_true_weights() {
    let Some(path) = lfm2_gguf_path() else {
        panic!("set MUMMU_LFM2_GGUF_PATH to the lfm2.5-1.2b q4_k_m gguf");
    };
    let Some(dir) = lfm2_dir() else {
        panic!("set MUMMU_LFM2_DIR to the same model's safetensors dir");
    };
    let st = dir.join("model.safetensors");
    let f = GgufFile::open(&path).expect("header parses");

    // (gguf name, safetensors name) — covers the plain rename, the per-head
    // norms, and the conv-kernel reshape (same bytes, unsqueezed shape).
    let pairs = [
        ("token_embd_norm.weight", "model.embedding_norm.weight"),
        (
            "blk.0.attn_norm.weight",
            "model.layers.0.operator_norm.weight",
        ),
        (
            "blk.2.attn_q_norm.weight",
            "model.layers.2.self_attn.q_layernorm.weight",
        ),
        (
            "blk.0.shortconv.conv.weight",
            "model.layers.0.conv.conv.weight",
        ),
        (
            "blk.15.shortconv.conv.weight",
            "model.layers.15.conv.conv.weight",
        ),
    ];
    for (ours_name, ref_name) in pairs {
        let ours = f.read_tensor_f32(ours_name).expect("dequantizes");
        let reference = safetensors_bf16_f32(&st, ref_name);
        assert_eq!(ours.len(), reference.len(), "{ours_name}: same size");
        let exact = ours
            .iter()
            .zip(&reference)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(exact, "{ours_name} must be bit-exact vs {ref_name}");
    }
    eprintln!(
        "[real_gguf/lfm2] {} F32 tensors bit-exact vs safetensors (incl. conv kernels)",
        pairs.len()
    );
}

/// The LFM2.5 tokenizer rebuilt from GGUF metadata must match the
/// checkpoint's `tokenizer.json` — with AND without the BOS-adding
/// post-processor (LFM2 sets `add_bos_token`, unlike Qwen2).
#[test]
#[ignore = "needs the local LFM2.5 GGUF (MUMMU_LFM2_GGUF_PATH) + safetensors dir (MUMMU_LFM2_DIR)"]
fn real_lfm2_gguf_tokenizer_matches_the_hf_tokenizer() {
    let Some(path) = lfm2_gguf_path() else {
        panic!("set MUMMU_LFM2_GGUF_PATH to the lfm2.5-1.2b q4_k_m gguf");
    };
    let dir = lfm2_dir().expect("set MUMMU_LFM2_DIR to the safetensors dir");
    let f = GgufFile::open(&path).expect("header parses");
    let ours = mummu::tokenizer::tokenizer_from_gguf(&f).expect("tokenizer builds");
    let reference =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");

    let chat = mummu::chat::ChatMl::lfm2();
    let battery = [
        chat.render(&[
            mummu::chat::Turn::system("You are a concise assistant."),
            mummu::chat::Turn::user("What is 2+2? Answer in one short sentence."),
        ]),
        "The quick brown fox jumps over the lazy dog.".into(),
        "héllo wörld — 世界 · Ω ≠ ω · 🦀🔥".into(),
        "1234567890 3.14159 1e-6 0x7F".into(), // digits split in ≤3 groups
        "  leading spaces, trailing  \n\nnewlines\r\nand\ttabs ".into(),
        "don't they're we've I'll it's CAN'T".into(),
    ];
    for text in &battery {
        for add_special in [true, false] {
            let a = ours.encode(text.as_str(), add_special).expect("encodes");
            let b = reference
                .encode(text.as_str(), add_special)
                .expect("encodes");
            assert_eq!(
                a.get_ids(),
                b.get_ids(),
                "ids diverge (add_special={add_special}) on {text:?}"
            );
        }
    }
    eprintln!(
        "[real_gguf/lfm2] tokenizer-from-GGUF: {} prompts × 2 modes byte-identical",
        battery.len()
    );
}

/// END-TO-END for the hybrid: the LFM2.5 Q4_K_M GGUF alone becomes a running
/// model on the GPU (conv layers, attention layers, per-head norms — all
/// mapped from llama.cpp naming), greedy-decodes a correct answer, and its
/// first-token logits agree with the bf16 safetensors build.
#[tokio::test]
#[ignore = "needs the local LFM2.5 GGUF (MUMMU_LFM2_GGUF_PATH) + safetensors dir (MUMMU_LFM2_DIR) + GPU"]
async fn real_lfm2_gguf_loads_and_decodes_on_gpu() {
    use mummu::models::lfm2;
    let Some(path) = lfm2_gguf_path() else {
        panic!("set MUMMU_LFM2_GGUF_PATH to the lfm2.5-1.2b q4_k_m gguf");
    };
    let dir = lfm2_dir().expect("set MUMMU_LFM2_DIR to the safetensors dir");
    assert!(use_gpu(), "this proof wants the real GPU");
    let device = mummu::backend::gpu_device();

    let header = GgufFile::open(&path).expect("header parses");
    let tok = mummu::tokenizer::tokenizer_from_gguf(&header).expect("tokenizer from metadata");
    let chat = mummu::chat::ChatMl::lfm2();
    let prompt_text = chat.render(&[
        mummu::chat::Turn::system("You are a concise assistant."),
        mummu::chat::Turn::user("What is 2+2? Answer in one short sentence."),
    ]);
    // The rendered template already carries <|startoftext|>; no auto-BOS.
    let prompt = tok
        .encode(prompt_text, false)
        .expect("prompt encodes")
        .get_ids()
        .to_vec();

    let gguf_model = lfm2::load_from_gguf(&path, &device).expect("gguf load is checked");
    assert_eq!(gguf_model.config.num_hidden_layers, 16);
    assert_eq!(
        gguf_model
            .config
            .layer_types
            .iter()
            .filter(|t| *t == "conv")
            .count(),
        10,
        "the 1.2B is 10 conv + 6 attention"
    );
    let ids = gguf_model
        .greedy_generate(&prompt, 32, &device)
        .await
        .expect("decode");
    let text = tok.decode(&ids, true).expect("ids decode");
    eprintln!("[real_gguf/lfm2] Q4_K_M greedy: {text:?}");
    assert!(text.contains('4'), "expected the answer 4 in: {text:?}");

    let logits_of = |m: &lfm2::LoadedLfm2| -> Vec<f32> {
        let mut cache = m.new_cache();
        m.forward(&prompt, 0, &mut cache, &device)
            .into_data()
            .to_vec::<f32>()
            .expect("logits read back")
    };
    let gguf_logits = logits_of(&gguf_model);
    drop(gguf_model);
    let st_model = lfm2::load_from_dir(&dir, &device).expect("safetensors load");
    let st_logits = logits_of(&st_model);
    drop(st_model);

    let argmax = |v: &[f32]| -> usize {
        let (mut best, mut best_v) = (0usize, f32::NEG_INFINITY);
        for (i, &x) in v.iter().enumerate() {
            if x > best_v {
                (best, best_v) = (i, x);
            }
        }
        best
    };
    let cos = cosine(&gguf_logits, &st_logits);
    let (g_top, s_top) = (argmax(&gguf_logits), argmax(&st_logits));
    eprintln!(
        "[real_gguf/lfm2] first-token logits: cosine {cos:.5} vs bf16 · top-1 {g_top} vs {s_top}"
    );
    assert_eq!(g_top, s_top, "Q4_K_M must agree with bf16 on the top token");
    assert!(
        cos > 0.95,
        "logit cosine {cos} — quantization noise should be small, layout bugs are not"
    );
}

/// END-TO-END: the Q4_K_M GGUF alone (config + weights from the one file)
/// becomes a running model on the GPU — greedy-decodes a correct answer, and
/// its first-token logits agree with the bf16 safetensors build of the same
/// checkpoint (top-1 identical, high cosine; small drift IS the quantization).
/// The models load sequentially — the second only after the first is dropped
/// — so peak VRAM stays one-model-sized.
#[tokio::test]
#[ignore = "needs the local GGUF (MUMMU_GGUF_PATH) + safetensors dir (MUMMU_QWEN2_DIR) + GPU"]
async fn real_qwen2_gguf_loads_and_decodes_on_gpu() {
    let Some(path) = gguf_path() else {
        panic!("set MUMMU_GGUF_PATH to the qwen2.5-1.5b-instruct q4_k_m gguf");
    };
    let dir = std::env::var_os("MUMMU_QWEN2_DIR")
        .map(PathBuf::from)
        .filter(|d| d.join("tokenizer.json").is_file())
        .expect("set MUMMU_QWEN2_DIR to the same model's safetensors dir (tokenizer.json)");
    assert!(use_gpu(), "this proof wants the real GPU");
    let device = mummu::backend::gpu_device();

    // The tokenizer comes from the GGUF itself — the whole model is ONE file
    // (byte-verified against tokenizer.json by the sibling test).
    let header = GgufFile::open(&path).expect("header parses");
    let tok = mummu::tokenizer::tokenizer_from_gguf(&header).expect("tokenizer from metadata");
    let chat = mummu::chat::ChatMl::qwen2();
    let prompt_text = chat.render(&[
        mummu::chat::Turn::system("You are a concise assistant."),
        mummu::chat::Turn::user("What is 2+2? Answer in one short sentence."),
    ]);
    let prompt = tok
        .encode(prompt_text, true)
        .expect("prompt encodes")
        .get_ids()
        .to_vec();

    // Leg 1: the GGUF-loaded model decodes a coherent, correct answer.
    let gguf_model = qwen2::load_from_gguf(&path, &device).expect("gguf load is checked");
    assert_eq!(gguf_model.config.vocab_size, 151_936);
    assert_eq!(gguf_model.config.num_hidden_layers, 28);
    let ids = gguf_model
        .greedy_generate(&prompt, 32, &device)
        .await
        .expect("decode");
    let text = tok.decode(&ids, true).expect("ids decode");
    eprintln!("[real_gguf] Q4_K_M greedy: {text:?}");
    assert!(text.contains('4'), "expected the answer 4 in: {text:?}");

    let logits_of = |m: &qwen2::LoadedQwen2| -> Vec<f32> {
        let mut cache = m.new_cache();
        m.forward(&prompt, 0, &mut cache, &device)
            .into_data()
            .to_vec::<f32>()
            .expect("logits read back")
    };
    let argmax = |v: &[f32]| -> usize {
        let (mut best, mut best_v) = (0usize, f32::NEG_INFINITY);
        for (i, &x) in v.iter().enumerate() {
            if x > best_v {
                (best, best_v) = (i, x);
            }
        }
        best
    };

    // Leg 2: first-token logits vs the bf16 safetensors build — sequential
    // loads (drop first) keep peak VRAM at one model.
    let gguf_logits = logits_of(&gguf_model);
    drop(gguf_model);
    let st_model = qwen2::load_from_dir(&dir, &device).expect("safetensors load");
    let st_logits = logits_of(&st_model);
    drop(st_model);

    let top5 = |v: &[f32]| -> Vec<usize> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]));
        idx.truncate(5);
        idx
    };
    let cos = cosine(&gguf_logits, &st_logits);
    let (g_top, s_top) = (argmax(&gguf_logits), argmax(&st_logits));
    let (g5, s5) = (top5(&gguf_logits), top5(&st_logits));
    let overlap = g5.iter().filter(|id| s5.contains(id)).count();
    eprintln!(
        "[real_gguf] first-token logits: cosine {cos:.5} vs bf16 · top-1 {g_top} vs {s_top} · top-5 overlap {overlap}/5 ({g5:?} vs {s5:?})"
    );
    assert_eq!(g_top, s_top, "Q4_K_M must agree with bf16 on the top token");
    assert!(overlap >= 4, "top-5 sets diverge: {g5:?} vs {s5:?}");
    // Measured 0.977 on this checkpoint: 28 layers of Q4_K_M drift (plus the
    // bf16 reference's own rounding). A layout/scale decode bug reads ≈ 0.
    assert!(
        cos > 0.95,
        "logit cosine {cos} — quantization noise should be small, layout bugs are not"
    );
}
