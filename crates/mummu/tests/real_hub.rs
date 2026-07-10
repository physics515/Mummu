//! Real-network Hub download proof: fetch a small model from HuggingFace,
//! checked-load it, and run it. Ignored by default (network + ~90 MB); run with
//!
//! ```text
//! MUMMU_HUB_DEST=some/tmp/dir cargo test -p mummu --release --test real_hub -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mummu::backend::Cpu;
use mummu::hub;
use mummu::models::minilm;
use tokenizers::Tokenizer;

/// Small enough to download in seconds, real enough to prove the pipeline.
const REPO: &str = "sentence-transformers/all-MiniLM-L6-v2";

/// The MiniLM entry from the built-in catalog (also pins the repo above).
fn minilm_spec() -> mummu::registry::ModelSpec {
    mummu::registry::catalog()
        .into_iter()
        .find(|s| s.repo == REPO)
        .expect("MiniLM is in the built-in catalog")
}

#[test]
#[ignore = "needs network (MUMMU_HUB_DEST names the download dir)"]
fn hub_download_then_load_then_embed() {
    let Some(dest) = std::env::var_os("MUMMU_HUB_DEST").map(PathBuf::from) else {
        panic!("set MUMMU_HUB_DEST to a scratch dir for the ~90 MB download");
    };
    let mut events = 0u64;
    let mut last_total = None;
    // Spec-driven: the catalog entry names the repo/revision/dir.
    let dir = minilm_spec()
        .fetch(&dest, |p| {
            events += 1;
            last_total = p.total_bytes;
        })
        .expect("hub fetch");
    // Either the network streamed (progress fired) or everything was already
    // cached from a prior run (zero events) — both are correct; the load
    // below is the real proof either way.
    eprintln!(
        "[real_hub] fetched into {} ({events} progress events, total {last_total:?})",
        dir.display()
    );

    for f in ["config.json", "tokenizer.json", "model.safetensors"] {
        assert!(dir.join(f).is_file(), "{f} must exist after fetch");
    }

    // Checked load + a real embedding: the downloaded artifacts are usable.
    let device = burn::tensor::Device::<Cpu>::default();
    let loaded = minilm::load_from_dir::<Cpu>(&dir, &device).expect("checked load");
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer loads");
    let enc = tok
        .encode("Downloads that verify themselves.", true)
        .expect("encodes");
    let mask: Vec<f32> = enc.get_attention_mask().iter().map(|&m| m as f32).collect();
    let embedding = loaded
        .embed_ids(enc.get_ids(), &mask, &device)
        .expect("embeds");
    let norm: f32 = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
    eprintln!(
        "[real_hub] embedding dims {}, norm {norm:.6}",
        embedding.len()
    );
    assert_eq!(embedding.len(), 384);
    assert!((norm - 1.0).abs() < 1e-3, "L2 norm should be 1, got {norm}");
}

#[test]
#[ignore = "needs network (MUMMU_HUB_DEST names the download dir)"]
fn hub_resume_completes_a_partial_download_byte_identical() {
    let Some(dest) = std::env::var_os("MUMMU_HUB_DEST").map(PathBuf::from) else {
        panic!("set MUMMU_HUB_DEST to a scratch dir");
    };
    let dir = dest.join("resume-proof");
    let url = hub::hub_file_url(REPO, "main", "tokenizer.json");
    let full = dir.join("tokenizer.json");

    // Reference download (or cache hit from a prior run).
    hub::fetch_file(&url, &full, |_| {}).expect("reference fetch");
    let reference = std::fs::read(&full).expect("reference bytes");
    assert!(reference.len() > 200_000, "file too small to prove resume");

    // Simulate a killed download: keep only the first half as `.part`.
    std::fs::remove_file(&full).expect("drop completed file");
    let half = reference.len() / 2;
    std::fs::write(dir.join("tokenizer.json.part"), &reference[..half]).expect("seed part");

    let mut first_event_received = None;
    hub::fetch_file(&url, &full, |p| {
        first_event_received.get_or_insert(p.received_bytes);
    })
    .expect("resumed fetch");

    let resumed = std::fs::read(&full).expect("resumed bytes");
    assert_eq!(
        resumed, reference,
        "resumed download must be byte-identical to a full one"
    );
    // The first progress event must already include the resumed prefix —
    // proof the transfer continued rather than restarted.
    let first = first_event_received.expect("progress fired");
    assert!(
        first > half as u64,
        "first event at {first} bytes should sit past the {half}-byte seed"
    );
    eprintln!(
        "[real_hub] resume: seeded {half} bytes, first event at {first}, final {} bytes identical",
        resumed.len()
    );
}

/// One-shot helper the nightly uses to pull the CPU-tier Qwen into a cache
/// dir (also a second real proof of the sharded/single-file fetch path on a
/// 1 GB checkpoint).
#[test]
#[ignore = "needs network (MUMMU_HUB_DEST names the download dir); ~1 GB"]
fn hub_fetches_the_cpu_tier_qwen() {
    let Some(dest) = std::env::var_os("MUMMU_HUB_DEST").map(PathBuf::from) else {
        panic!("set MUMMU_HUB_DEST to a scratch dir for the ~1 GB download");
    };
    let spec = mummu::registry::catalog()
        .into_iter()
        .find(|s| s.name == "qwen2.5-0.5b-instruct")
        .expect("0.5B is in the catalog");
    let dir = spec.fetch(&dest, |_| {}).expect("hub fetch");
    for f in ["config.json", "tokenizer.json", "model.safetensors"] {
        assert!(dir.join(f).is_file(), "{f} must exist after fetch");
    }
    eprintln!("[real_hub] 0.5B fetched into {}", dir.display());
}
