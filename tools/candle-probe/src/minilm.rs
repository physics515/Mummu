//! Same-weights Candle reference for the MiniLM leg of the P7 parity gate.
//!
//! Usage: `minilm-probe <model-dir>` — loads the all-MiniLM checkpoint from
//! `<model-dir>` (`config.json` / `tokenizer.json` / `model.safetensors`),
//! embeds the fixed sentence below on CPU in f32 (masked-mean-pool +
//! L2-normalize, exactly Mummu's pipeline), and prints the sentence, token
//! ids, and the full embedding vector as JSON. Redirect into
//! `crates/mummu/tests/fixtures/minilm_embedding.json` to refresh the
//! committed fixture that `tests/real_minilm.rs` compares Burn against.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};

/// Must stay identical to the sentence in the parity test in
/// `crates/mummu/tests/real_minilm.rs`; the fixture carries it for a drift check.
const SENTENCE: &str = "The quick brown fox jumps over the lazy dog.";

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().ok_or("usage: minilm-probe <model-dir>")?);
    assert!(dir.is_dir(), "not a directory: {}", dir.display());

    let device = Device::Cpu;
    let config: Config = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &device)?
    };
    let model = BertModel::load(vb, &config)?;

    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))?;
    let enc = tokenizer.encode(SENTENCE, true)?;
    let ids = enc.get_ids().to_vec();
    let mask: Vec<u32> = enc.get_attention_mask().to_vec();
    assert!(!ids.is_empty(), "sentence tokenized to nothing");
    assert_eq!(ids.len(), mask.len(), "ids/mask length mismatch");

    let t = ids.len();
    let input = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;
    let type_ids = input.zeros_like()?;
    let attention = Tensor::new(mask.as_slice(), &device)?.unsqueeze(0)?;
    let hidden = model.forward(&input, &type_ids, Some(&attention))?; // [1, t, h]

    // Masked mean pool + L2 normalize — the sentence-transformers recipe and
    // exactly what `LoadedMiniLm::embed_ids` does.
    let mask_f = attention.to_dtype(DType::F32)?.reshape((1, t, 1))?;
    let summed = hidden.broadcast_mul(&mask_f)?.sum(1)?; // [1, h]
    let count = mask_f.sum(1)?; // [1, 1]
    let mean = summed.broadcast_div(&count)?;
    let norm = mean.sqr()?.sum_keepdim(1)?.sqrt()?;
    let embedding = mean.broadcast_div(&norm)?.flatten_all()?.to_vec1::<f32>()?;

    let out = serde_json::json!({
        "reference": "candle-0.9.1 cpu f32",
        "sentence": SENTENCE,
        "ids": ids,
        "embedding": embedding,
    });
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}
