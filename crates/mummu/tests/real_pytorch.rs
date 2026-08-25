//! Real-weights PyTorch state-dict import proof: fetch MiniLM's original
//! `pytorch_model.bin` from the Hub, load it through `PytorchStore`, and
//! prove the embedding matches the safetensors-loaded model on the SAME
//! sentence — identical weights through two formats must agree. Ignored by
//! default (network, ~90 MB per format); run with
//!
//! ```text
//! MUMMU_HUB_DEST=some/tmp/dir cargo test -p mummu --release --test real_pytorch -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mummu::hub;
use mummu::models::minilm;
use tokenizers::Tokenizer;

const REPO: &str = "sentence-transformers/all-MiniLM-L6-v2";
const SENTENCE: &str = "Two formats, one set of weights.";

fn embed_from(dir: &std::path::Path) -> Vec<f32> {
    let device = mummu::backend::cpu_device();
    let loaded = minilm::load_from_dir(dir, &device).expect("checked load");
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer loads");
    let enc = tok.encode(SENTENCE, true).expect("encodes");
    let mask: Vec<f32> = enc.get_attention_mask().iter().map(|&m| m as f32).collect();
    loaded
        .embed_ids(enc.get_ids(), &mask, &device)
        .expect("embeds")
}

#[test]
#[ignore = "needs network (MUMMU_HUB_DEST names the download dir)"]
fn pytorch_bin_load_matches_safetensors_load() {
    let Some(dest) = std::env::var_os("MUMMU_HUB_DEST").map(PathBuf::from) else {
        panic!("set MUMMU_HUB_DEST to a scratch dir for the ~180 MB of downloads");
    };

    // Reference: the safetensors copy (the catalog dir other tests share).
    let st_dir = dest.join("all-minilm-l6-v2");
    hub::fetch_model(REPO, "main", &st_dir, |_| {}).expect("safetensors fetch");

    // Candidate: a dir holding ONLY the PyTorch state dict (+ config &
    // tokenizer), so weights_file() must take the .bin path.
    let pt_dir = dest.join("minilm-pt");
    for f in ["config.json", "tokenizer.json", "pytorch_model.bin"] {
        hub::fetch_file(&hub::hub_file_url(REPO, "main", f), &pt_dir.join(f), |_| {})
            .expect("pt fetch");
    }
    assert!(
        !pt_dir.join("model.safetensors").exists(),
        "candidate dir must not contain safetensors — the test would prove nothing"
    );

    let reference = embed_from(&st_dir);
    let candidate = embed_from(&pt_dir);
    assert_eq!(reference.len(), 384);
    assert_eq!(candidate.len(), 384);

    let max_abs_diff = reference
        .iter()
        .zip(&candidate)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let cosine: f32 = reference.iter().zip(&candidate).map(|(a, b)| a * b).sum();
    eprintln!(
        "[real_pytorch] safetensors vs pytorch_model.bin: cosine {cosine:.8}, max |Δ| {max_abs_diff:e}"
    );
    // Same f32 weights through two containers on one backend: numerically
    // identical up to load-order noise — far tighter than any semantic bound.
    assert!(
        max_abs_diff < 1e-6,
        "formats diverge: max |Δcomponent| = {max_abs_diff}"
    );
}
