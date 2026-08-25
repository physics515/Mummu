//! Real-network Hub download proof: fetch a small model from HuggingFace,
//! checked-load it, and run it. Ignored by default (network + ~90 MB); run with
//!
//! ```text
//! MUMMU_HUB_DEST=some/tmp/dir cargo test -p mummu --release --test real_hub -- --ignored --nocapture
//! ```

use std::path::PathBuf;

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
    let device = mummu::backend::cpu_device();
    let loaded = minilm::load_from_dir(&dir, &device).expect("checked load");
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

#[test]
#[ignore = "needs network (MUMMU_HUB_DEST names the download dir)"]
fn hub_sha256_verification_catches_and_heals_a_corrupt_cache() {
    let Some(dest) = std::env::var_os("MUMMU_HUB_DEST").map(PathBuf::from) else {
        panic!("set MUMMU_HUB_DEST to a scratch dir for the ~90 MB download");
    };
    let dir = dest.join("verify-proof");
    let url = hub::hub_file_url(REPO, "main", "model.safetensors");
    let weights = dir.join("model.safetensors");
    let verify = hub::FetchOptions {
        verify_cached: true,
    };

    // Fresh (or cached) download; the stream itself hash-verifies against the
    // Hub's X-Linked-ETag, so success here is already an integrity proof.
    hub::fetch_file(&url, &weights, |_| {}).expect("verified fetch");
    let healthy_len = weights.metadata().expect("weights exist").len();

    // A clean cache re-verifies without being touched.
    let modified_before = weights.metadata().unwrap().modified().unwrap();
    hub::fetch_file_with(&url, &weights, verify, |_| {}).expect("clean cache re-verifies");
    assert_eq!(
        weights.metadata().unwrap().modified().unwrap(),
        modified_before,
        "a matching cached file must not be rewritten"
    );

    // Corrupt one byte mid-file (length unchanged — only the hash can see it),
    // then watch verify_cached self-heal by re-downloading.
    let mut bytes = std::fs::read(&weights).expect("read weights");
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xFF;
    std::fs::write(&weights, &bytes).expect("plant corruption");
    let mut streamed = 0u64;
    hub::fetch_file_with(&url, &weights, verify, |p| streamed = p.received_bytes)
        .expect("corrupt cache heals");
    assert_eq!(
        weights.metadata().unwrap().len(),
        healthy_len,
        "healed file has the healthy length"
    );
    assert_eq!(
        streamed, healthy_len,
        "healing must have re-streamed the whole file"
    );
    eprintln!(
        "[real_hub] sha256 gate: clean cache untouched; flipped byte at {mid} was caught and healed ({streamed} bytes re-streamed)"
    );

    // Resumed LFS download: the .part prefix must fold into the stream hash
    // (a 206 + hash_part_prefix pass over a real X-Linked-ETag file).
    let healthy = std::fs::read(&weights).expect("healthy bytes");
    std::fs::remove_file(&weights).expect("drop completed file");
    let half = healthy.len() / 2;
    std::fs::write(dir.join("model.safetensors.part"), &healthy[..half]).expect("seed part");
    hub::fetch_file(&url, &weights, |_| {}).expect("resumed fetch hash-verifies");
    assert_eq!(
        weights.metadata().unwrap().len(),
        healthy_len,
        "resumed file has the healthy length"
    );
    eprintln!("[real_hub] sha256 gate: resume from {half} bytes re-verified the whole file");
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

/// Registry → the new CPU-tier hybrid: LFM2.5-230M (June 2026) is the same
/// `lfm2` architecture the zoo already covers, so the catalog entry alone
/// makes it installable — this proof fetches it, checked-loads it on the CPU
/// backend, and greedy-decodes real text end-to-end.
#[tokio::test]
#[ignore = "needs network (MUMMU_HUB_DEST names the download dir; ~470 MB)"]
async fn hub_fetches_and_runs_lfm2_230m_on_cpu() {
    use mummu::models::CausalLm;
    let Some(dest) = std::env::var_os("MUMMU_HUB_DEST").map(PathBuf::from) else {
        panic!("set MUMMU_HUB_DEST to a scratch dir for the ~470 MB download");
    };
    let spec = mummu::registry::catalog()
        .into_iter()
        .find(|s| s.name == "lfm2.5-230m")
        .expect("the 230M is in the catalog");
    let dir = spec.fetch(&dest, |_| {}).expect("hub fetch");
    for f in ["config.json", "tokenizer.json", "model.safetensors"] {
        assert!(dir.join(f).is_file(), "{f} must exist after fetch");
    }

    let device = mummu::backend::cpu_device();
    let loaded = mummu::models::lfm2::load_from_dir(&dir, &device).expect("checked load");
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer loads");
    let raw = mummu::chat::ChatMl::lfm2().render(&[mummu::chat::Turn::user(
        "What is 2 + 2? Answer in one short sentence.",
    )]);
    let ids = tok
        .encode(raw.as_str(), false)
        .expect("encodes")
        .get_ids()
        .to_vec();
    let out = loaded
        .greedy_generate(&ids, 16, &device)
        .await
        .expect("greedy decode");
    let text = tok.decode(&out, true).expect("decode");
    eprintln!("[real_hub/230m] {} tokens: {text:?}", out.len());
    assert!(!out.is_empty(), "decoded no tokens");
    assert!(
        text.contains('4'),
        "a 230M instruct model should answer 2 + 2 = 4, got: {text:?}"
    );
}

/// Registry → single-file GGUF install: the catalog's LFM2.5 Q4_K_M spec
/// downloads through `fetch` (one ~700 MB file, resumable/cache-first like
/// every hub fetch), lands where `gguf_path` says, parses as a valid GGUF of
/// the right architecture, and its metadata builds the tokenizer — the whole
/// app-facing "install a GGUF model" path in one proof.
#[test]
#[ignore = "needs network (MUMMU_HUB_DEST names the download dir; ~700 MB)"]
fn hub_gguf_spec_downloads_and_parses() {
    use mummu::registry::WeightFormat;
    let Some(dest) = std::env::var_os("MUMMU_HUB_DEST").map(PathBuf::from) else {
        panic!("set MUMMU_HUB_DEST to a scratch dir for the ~700 MB download");
    };
    let spec = mummu::registry::catalog()
        .into_iter()
        .find(|s| matches!(s.format, WeightFormat::Gguf { .. }) && s.name.starts_with("lfm2.5"))
        .expect("the catalog has the LFM2.5 GGUF entry");

    let mut events = 0u64;
    let dir = spec.fetch(&dest, |_| events += 1).expect("gguf fetch");
    let path = spec.gguf_path(&dest).expect("gguf specs have a file path");
    assert!(path.starts_with(&dir), "file lives in the spec's dir");
    assert!(path.is_file(), "downloaded file exists at {path:?}");

    let f = mummu::gguf::GgufFile::open(&path).expect("valid GGUF");
    assert_eq!(f.architecture(), Some("lfm2"));
    assert!(!f.tensors.is_empty());
    let tok = mummu::tokenizer::tokenizer_from_gguf(&f).expect("tokenizer from metadata");
    assert!(tok.token_to_id("<|im_end|>").is_some());
    eprintln!(
        "[real_hub/gguf] {} → {} tensors, tokenizer ok ({events} progress events)",
        spec.name,
        f.tensors.len()
    );
}

/// The install path must feed the import-validation gates.
///
/// `validate_checkpoint_dir` fails OPEN: with no sibling `tokenizer_config.json`
/// it returns `Ok(None)` and the EOS-agreement + added-token-id checks quietly
/// do not run. So a checkpoint installed through Mummu's own registry has to
/// ARRIVE with that file, or every one of those gates is decorative for exactly
/// the models Mummu installed itself. This fetches a catalog model into a
/// **clean** dir (no hand-populated fixtures) and asserts the gate found
/// something to check.
#[test]
#[ignore = "needs network (MUMMU_HUB_DEST names the download dir; ~90 MB)"]
fn a_registry_install_arrives_with_the_files_the_import_gates_read() {
    let Some(dest) = std::env::var_os("MUMMU_HUB_DEST").map(PathBuf::from) else {
        panic!("set MUMMU_HUB_DEST to a scratch dir for the ~90 MB download");
    };
    // A dir of our own, so nothing here can be satisfied by a fixture someone
    // populated by hand — the whole point of the check.
    let clean = dest.join("gate-feed-probe");
    if clean.exists() {
        std::fs::remove_dir_all(&clean).expect("clear the probe dir");
    }
    let dir = minilm_spec().fetch(&clean, |_| {}).expect("hub fetch");

    assert!(
        dir.join("tokenizer_config.json").is_file(),
        "the install must fetch tokenizer_config.json, or the import gates no-op"
    );

    // And the gate itself must now have something to validate. MiniLM declares
    // no eos in config.json, so pass an empty set and assert only that the
    // config was FOUND — `Ok(None)` here is the silent-no-op this test exists
    // to forbid.
    let cfg = mummu::tokenizer::validate_checkpoint_dir(&dir, &[], None)
        .expect("tokenizer_config.json parses and agrees with tokenizer.json");
    assert!(
        cfg.is_some(),
        "validate_checkpoint_dir returned Ok(None) — the gate silently no-opped"
    );

    std::fs::remove_dir_all(&clean).ok();
}
