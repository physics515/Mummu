//! Real-file GGUF header proof: parse an actual llama.cpp-quantized model
//! (Qwen2.5-1.5B-Instruct Q4_K_M) and check the header describes the model
//! we know. Ignored by default; run with
//!
//! ```text
//! MUMMU_GGUF_PATH=path/to/qwen2.5-1.5b-instruct-q4_k_m.gguf cargo test -p mummu --test real_gguf -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mummu::gguf::{GgmlType, GgufFile, GgufValue};

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
