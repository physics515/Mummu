//! Real-weights MiniLM smoke: load the actual all-MiniLM checkpoint on the
//! CPU backend and check the embedding space behaves semantically. Ignored by
//! default; run with
//!
//! ```text
//! MUMMU_MINILM_DIR=path/to/minilm cargo test -p mummu --test real_minilm -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mummu::backend::Cpu;
use mummu::models::minilm;
use tokenizers::Tokenizer;

fn minilm_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("MUMMU_MINILM_DIR")?);
    dir.is_dir().then_some(dir)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    // Embeddings are L2-normalized, so the dot product IS the cosine.
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[test]
#[ignore = "needs local MiniLM weights (MUMMU_MINILM_DIR)"]
fn minilm_embeds_similar_sentences_closer_than_dissimilar() {
    let Some(dir) = minilm_dir() else {
        panic!("set MUMMU_MINILM_DIR to a dir with config.json/tokenizer.json/model.safetensors");
    };
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json loads");
    let device = burn::tensor::Device::<Cpu>::default();
    let loaded = minilm::load_from_dir::<Cpu>(&dir, &device).expect("weights load checked");

    let embed = |text: &str| -> Vec<f32> {
        let enc = tok.encode(text, true).expect("encodes");
        let ids = enc.get_ids().to_vec();
        let mask: Vec<f32> = enc.get_attention_mask().iter().map(|&m| m as f32).collect();
        loaded.embed_ids(&ids, &mask, &device).expect("embeds")
    };

    let cat1 = embed("The cat sat on the mat.");
    let cat2 = embed("A feline rested on the rug.");
    let finance = embed("Quarterly revenue grew twelve percent year over year.");

    let sim = cosine(&cat1, &cat2);
    let dis1 = cosine(&cat1, &finance);
    let dis2 = cosine(&cat2, &finance);
    eprintln!("[real_minilm] cat~cat = {sim:.4}, cat~finance = {dis1:.4}/{dis2:.4}");

    assert!(sim > 0.5, "paraphrases should be similar, got {sim}");
    assert!(
        sim > dis1 + 0.2 && sim > dis2 + 0.2,
        "paraphrase similarity ({sim}) should clearly beat cross-topic ({dis1}, {dis2})"
    );

    // Unit norm on real weights too.
    let norm: f32 = cat1.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-3, "L2 norm should be 1, got {norm}");
}
