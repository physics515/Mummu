//! Real-file GGUF header proof: parse an actual llama.cpp-quantized model
//! (Qwen2.5-1.5B-Instruct Q4_K_M) and check the header describes the model
//! we know. Ignored by default; run with
//!
//! ```text
//! MUMMU_GGUF_PATH=path/to/qwen2.5-1.5b-instruct-q4_k_m.gguf cargo test -p mummu --test real_gguf -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mummu::backend::{Gpu, use_gpu};
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
    raw.chunks_exact(2)
        .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
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

/// END-TO-END: the Q4_K_M GGUF alone (config + weights from the one file)
/// becomes a running model on the GPU — greedy-decodes a correct answer, and
/// its first-token logits agree with the bf16 safetensors build of the same
/// checkpoint (top-1 identical, high cosine; small drift IS the quantization).
/// The models load sequentially — the second only after the first is dropped
/// — so peak VRAM stays one-model-sized.
#[test]
#[ignore = "needs the local GGUF (MUMMU_GGUF_PATH) + safetensors dir (MUMMU_QWEN2_DIR) + GPU"]
fn real_qwen2_gguf_loads_and_decodes_on_gpu() {
    let Some(path) = gguf_path() else {
        panic!("set MUMMU_GGUF_PATH to the qwen2.5-1.5b-instruct q4_k_m gguf");
    };
    let dir = std::env::var_os("MUMMU_QWEN2_DIR")
        .map(PathBuf::from)
        .filter(|d| d.join("tokenizer.json").is_file())
        .expect("set MUMMU_QWEN2_DIR to the same model's safetensors dir (tokenizer.json)");
    assert!(use_gpu(), "this proof wants the real GPU");
    let device = burn::tensor::Device::<Gpu>::default();

    // Tokenizer from the sibling checkpoint — tokenizer-from-GGUF-metadata
    // is the next P3 slice.
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
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
    let gguf_model = qwen2::load_from_gguf::<Gpu>(&path, &device).expect("gguf load is checked");
    assert_eq!(gguf_model.config.vocab_size, 151_936);
    assert_eq!(gguf_model.config.num_hidden_layers, 28);
    let ids = gguf_model
        .greedy_generate(&prompt, 32, &device)
        .expect("decode");
    let text = tok.decode(&ids, true).expect("ids decode");
    eprintln!("[real_gguf] Q4_K_M greedy: {text:?}");
    assert!(text.contains('4'), "expected the answer 4 in: {text:?}");

    let logits_of = |m: &qwen2::LoadedQwen2<Gpu>| -> Vec<f32> {
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
    let st_model = qwen2::load_from_dir::<Gpu>(&dir, &device).expect("safetensors load");
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
