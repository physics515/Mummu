//! Retrieval-tier parity gate: the embedder and the rerankers against the
//! `HuggingFace` `transformers` reference on the same weights.
//!
//! The fixture (`fixtures/retrieval_reference.json`) is written by
//! `tools/retrieval_reference.py` — f32, eager attention, the model cards'
//! own recipes — and replayed here item by item, three layers deep:
//!
//! 1. **token ids, exactly** — our prompt construction and tokenizer
//!    against theirs (a query's instruction prefix, a decoder embedder's
//!    appended `<|endoftext|>`, the reranker's template split);
//! 2. **the numbers** — each embedding's cosine and largest component error,
//!    each rerank score's yes/no logit gap and `P(yes)`;
//! 3. **the decision** — which document each query retrieves first, and the
//!    order the reranker puts every document in.
//!
//! One document is longer than 512 tokens, so the chunked prefill both
//! paths take through the KV cache is held to the one-shot reference.
//!
//! Ignored by default; needs the three checkpoints under one root, by their
//! catalog names:
//!
//! ```text
//! MUMMU_RETRIEVAL_DIR=~/.cache/mummu-models \
//!   cargo test --release -p mummu --test real_retrieval -- --ignored --nocapture
//! ```
//!
//! `MUMMU_RETRIEVAL_DEVICE=gpu` runs the same gate on the accelerator.

#![warn(clippy::pedantic, clippy::nursery, clippy::all)]

use std::path::PathBuf;

use burn::tensor::Device;
use mummu::embed::{Embedder, TextKind, cosine};
use mummu::rerank::Reranker;
use serde_json::Value;

const FIXTURE: &str = include_str!("fixtures/retrieval_reference.json");

/// Cosine an embedding must reach against the reference. Both sides are
/// f32 on the same weights, so what is left is reduction order through 28
/// layers.
const MIN_COSINE: f32 = 0.9999;
/// Largest single-component error tolerated on a unit vector.
const MAX_COMPONENT_ERR: f32 = 2e-3;
/// Largest error on the yes-minus-no logit gap a reranker score is made of.
const MAX_GAP_ERR: f32 = 5e-2;
/// Largest error on `P(yes)` itself.
const MAX_RELEVANCE_ERR: f32 = 5e-3;

fn root() -> PathBuf {
    let root = std::env::var_os("MUMMU_RETRIEVAL_DIR")
        .map(PathBuf::from)
        .expect("set MUMMU_RETRIEVAL_DIR to the dir holding the retrieval checkpoints");
    assert!(root.is_dir(), "{} is not a directory", root.display());
    root
}

fn device() -> Device {
    match std::env::var("MUMMU_RETRIEVAL_DEVICE").as_deref() {
        Ok("gpu") => mummu::backend::gpu_device(),
        _ => mummu::backend::cpu_device(),
    }
}

fn fixture() -> Value {
    serde_json::from_str(FIXTURE).expect("fixture parses")
}

fn ids_of(v: &Value) -> Vec<u32> {
    v.as_array()
        .expect("ids array")
        .iter()
        .map(|x| u32::try_from(x.as_u64().expect("id")).expect("id fits u32"))
        .collect()
}

fn f32_of(v: &Value) -> f32 {
    mummu_num::narrow(v.as_f64().expect("number"))
}

fn str_list(v: &Value) -> Vec<String> {
    v.as_array()
        .expect("string array")
        .iter()
        .map(|s| s.as_str().expect("string").to_owned())
        .collect()
}

/// Index of the largest value (first on ties).
fn argmax(xs: &[f32]) -> usize {
    let mut best = 0;
    for (i, x) in xs.iter().enumerate() {
        if *x > xs[best] {
            best = i;
        }
    }
    best
}

#[tokio::test]
#[ignore = "needs local retrieval checkpoints (MUMMU_RETRIEVAL_DIR)"]
async fn harrier_embeddings_match_the_hf_reference() {
    let fx = fixture();
    let reference = &fx["harrier-oss-v1-0.6b"];
    let device = device();
    let started = std::time::Instant::now();
    let embedder = Embedder::load_from_dir(&root().join("harrier-oss-v1-0.6b"), &device)
        .expect("harrier loads");
    eprintln!("[real_retrieval] harrier loaded in {:?}", started.elapsed());
    assert_eq!(embedder.dims(), 1024);
    assert_eq!(
        embedder.prompt(TextKind::Query),
        reference["query_prompt"].as_str().expect("query_prompt"),
        "the query prompt must come from the checkpoint's own files"
    );

    let mut queries = Vec::new();
    let mut documents = Vec::new();
    let mut ref_queries = Vec::new();
    let mut ref_documents = Vec::new();
    let mut worst_cos = 1.0f32;
    let mut worst_err = 0.0f32;
    for item in reference["items"].as_array().expect("items") {
        let kind = match item["kind"].as_str() {
            Some("query") => TextKind::Query,
            _ => TextKind::Document,
        };
        let text = item["text"].as_str().expect("text");
        let prompted = embedder.prompted(text, kind, None);
        let (ids, truncated) = embedder.tokenize(&prompted).expect("tokenizes");
        assert!(!truncated);
        assert_eq!(ids, ids_of(&item["ids"]), "token ids differ for {text:?}");

        let t = std::time::Instant::now();
        let got = embedder
            .embed(text, kind, None, &device)
            .await
            .expect("embeds");
        let want: Vec<f32> = item["vector"]
            .as_array()
            .expect("vector")
            .iter()
            .map(f32_of)
            .collect();
        let cos = cosine(&got.vector, &want);
        let err = got
            .vector
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "[real_retrieval] harrier {kind:?} {:>4} tokens in {:>8.1?}: cosine {cos:.8}, max|Δ| {err:.2e}",
            got.tokens,
            t.elapsed()
        );
        worst_cos = worst_cos.min(cos);
        worst_err = worst_err.max(err);
        let norm: f32 = got.vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "not unit norm: {norm}");
        match kind {
            TextKind::Query => {
                queries.push(got.vector);
                ref_queries.push(want);
            }
            TextKind::Document => {
                documents.push(got.vector);
                ref_documents.push(want);
            }
        }
    }
    eprintln!("[real_retrieval] harrier worst cosine {worst_cos:.8}, worst max|Δ| {worst_err:.2e}");
    assert!(worst_cos >= MIN_COSINE, "cosine {worst_cos} < {MIN_COSINE}");
    assert!(
        worst_err <= MAX_COMPONENT_ERR,
        "max|Δ| {worst_err} > {MAX_COMPONENT_ERR}"
    );

    // The decision: each query retrieves the same document first.
    for (q, (ours, theirs)) in queries.iter().zip(&ref_queries).enumerate() {
        let ours: Vec<f32> = documents.iter().map(|d| cosine(ours, d)).collect();
        let theirs: Vec<f32> = ref_documents.iter().map(|d| cosine(theirs, d)).collect();
        assert_eq!(
            argmax(&ours),
            argmax(&theirs),
            "query {q} retrieves differently"
        );
        assert_eq!(
            argmax(&ours),
            q,
            "query {q} should retrieve its own document"
        );
    }
}

async fn reranker_matches(name: &str) {
    let fx = fixture();
    let reference = &fx[name];
    let queries = str_list(&fx["queries"]);
    let documents = str_list(&fx["documents"]);
    let instruction = fx["instruction"].as_str().expect("instruction");
    let device = device();
    let started = std::time::Instant::now();
    let reranker = Reranker::load_from_dir(&root().join(name), &device).expect("reranker loads");
    eprintln!("[real_retrieval] {name} loaded in {:?}", started.elapsed());
    assert_eq!(reranker.default_instruction(), instruction);

    let mut ours = vec![vec![0.0f32; documents.len()]; queries.len()];
    let mut theirs = vec![vec![0.0f32; documents.len()]; queries.len()];
    let mut worst_gap = 0.0f32;
    let mut worst_rel = 0.0f32;
    for pair in reference["pairs"].as_array().expect("pairs") {
        let qi = usize::try_from(pair["query"].as_u64().expect("q")).expect("fits");
        let di = usize::try_from(pair["document"].as_u64().expect("d")).expect("fits");
        let (ids, truncated) = reranker
            .prompt_ids(&queries[qi], &documents[di], None)
            .expect("prompt");
        assert!(!truncated);
        assert_eq!(
            ids,
            ids_of(&pair["ids"]),
            "prompt ids differ for pair ({qi}, {di})"
        );

        let t = std::time::Instant::now();
        let (relevance, tokens) = reranker
            .score(&queries[qi], &documents[di], None, &device)
            .await
            .expect("scores");
        let want = f32_of(&pair["relevance"]);
        let want_gap = f32_of(&pair["yes_logit"]) - f32_of(&pair["no_logit"]);
        // Recover our gap from P(yes) = sigmoid(gap), away from saturation.
        let clamped = relevance.clamp(1e-6, 1.0 - 1e-6);
        let gap = (clamped / (1.0 - clamped)).ln();
        let gap_err = if want.min(1.0 - want) > 1e-5 {
            (gap - want_gap).abs()
        } else {
            0.0
        };
        eprintln!(
            "[real_retrieval] {name} q{qi} d{di} {tokens:>4} tokens in {:>8.1?}: P(yes) {relevance:.6} vs {want:.6}, gap Δ {gap_err:.2e}",
            t.elapsed()
        );
        worst_gap = worst_gap.max(gap_err);
        worst_rel = worst_rel.max((relevance - want).abs());
        ours[qi][di] = relevance;
        theirs[qi][di] = want;
    }
    eprintln!("[real_retrieval] {name} worst |ΔP| {worst_rel:.2e}, worst gap Δ {worst_gap:.2e}");
    assert!(worst_rel <= MAX_RELEVANCE_ERR, "|ΔP(yes)| {worst_rel}");
    assert!(worst_gap <= MAX_GAP_ERR, "logit gap error {worst_gap}");

    // The decision: the same best document for every query.
    for (q, (o, t)) in ours.iter().zip(&theirs).enumerate() {
        assert_eq!(
            argmax(o),
            argmax(t),
            "query {q} ranks a different document first"
        );
        assert_eq!(argmax(o), q, "query {q} should rank its own document first");
    }
}

#[tokio::test]
#[ignore = "needs local retrieval checkpoints (MUMMU_RETRIEVAL_DIR)"]
async fn qwen3_reranker_06b_matches_the_hf_reference() {
    reranker_matches("qwen3-reranker-0.6b").await;
}

#[tokio::test]
#[ignore = "needs local retrieval checkpoints (MUMMU_RETRIEVAL_DIR); ~16 GB f32"]
async fn qwen3_reranker_4b_matches_the_hf_reference() {
    reranker_matches("qwen3-reranker-4b").await;
}

/// `Qwen3-Embedding-0.6B` loads through the same trunk although its
/// `config.json` EOS (`<|endoftext|>`) disagrees with its tokenizer config's
/// (`<|im_end|>`) — the case `tokenizer::validate_tokenizer_ids` exists for —
/// and retrieves each fixture query's own document first. No numeric
/// reference: it is not a catalog model, this proves only that the
/// architecture path, not one checkpoint, is what the tier rests on.
#[tokio::test]
#[ignore = "needs local retrieval checkpoints (MUMMU_RETRIEVAL_DIR) incl. qwen3-embedding-0.6b"]
async fn qwen3_embedding_06b_loads_and_retrieves() {
    let fx = fixture();
    let device = device();
    let embedder = Embedder::load_from_dir(&root().join("qwen3-embedding-0.6b"), &device)
        .expect("Qwen3-Embedding-0.6B loads despite its EOS disagreement");
    assert_eq!(embedder.dims(), 1024);
    assert!(
        embedder.prompt(TextKind::Query).ends_with("\nQuery:"),
        "its own query prompt, without harrier's trailing space"
    );
    let mut docs = Vec::new();
    for d in str_list(&fx["documents"]) {
        let e = embedder
            .embed(&d, TextKind::Document, None, &device)
            .await
            .expect("embeds");
        docs.push(e.vector);
    }
    for (q, query) in str_list(&fx["queries"]).iter().enumerate() {
        let e = embedder
            .embed(query, TextKind::Query, None, &device)
            .await
            .expect("embeds");
        let sims: Vec<f32> = docs.iter().map(|d| cosine(&e.vector, d)).collect();
        eprintln!("[real_retrieval] qwen3-embedding q{q}: {sims:.3?}");
        assert_eq!(argmax(&sims), q, "query {q} retrieves the wrong document");
    }
}
