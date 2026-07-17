//! Quantized-reference parity leg (P3): llama.cpp itself running the **SAME
//! quantized GGUF file** our loader loads — the strict check that our
//! container parse + dequant feed the model exactly what ggml's compute path
//! sees. The bf16-build comparison in `real_gguf.rs` bounds quantization
//! drift; this leg removes the quantization variable entirely by putting the
//! identical Q4_K_M bytes on both sides.
//!
//! Everything on our side comes from the ONE .gguf (config, weights,
//! tokenizer); the reference is `llama-server` on the same file, raw
//! `/completion`, prompts as token-id arrays. Ignored by default; run with
//!
//! ```text
//! MUMMU_GGUF_PATH=path/to/qwen2.5-1.5b-instruct-q4_k_m.gguf \
//! MUMMU_LFM2_GGUF_PATH=path/to/lfm2.5-1.2b-instruct-q4_k_m.gguf \
//! MUMMU_LLAMA_SERVER=path/to/llama-server.exe \
//!   cargo test -p mummu --release --test parity_gguf -- --ignored --nocapture
//! ```

mod llama_ref;

use std::path::PathBuf;

use llama_ref::{LlamaServer, logprobs_at};
use mummu::backend::Gpu;
use mummu::gguf::GgufFile;
use mummu::models::CausalLm;
use mummu::models::{lfm2, qwen2, qwen3};

const PROMPT: &str = "List the first five prime numbers.";
const MAX_TOKENS: usize = 24;
const TOP_K: usize = 5;

/// Max |Δlogprob| over the top-k between our load (weights dequantized to f32
/// once, wgpu compute) and llama.cpp on the SAME Q4_K_M file (CPU kernels
/// that also quantize the *activations* to Q8_K per integer dot product — the
/// reference's own noise floor, absent from our f32 path and an order larger
/// than the BF16 leg's 1.5e-2 in `parity_lfm2.rs`). Measured on the dev GPU:
/// 2.66e-1 (Qwen2), 2.60e-1 (LFM2.5) — while the 23/24-token greedy sequences
/// are byte-identical and the top-3 ids match in order for both, so the drift
/// lives in the far tail. 7.5e-1 gives ~2.8x headroom; the greedy byte-match
/// is the primary assert.
const LOGPROB_ABS_TOLERANCE: f64 = 7.5e-1;

/// The reference's activation-quantization noise (see above) reshuffles
/// near-ties deep in the top-k (measured: LFM2.5 swaps ranks 4-5), so the
/// strict-order assert covers the top 3 and the rest is set overlap.
const STRICT_ORDER_K: usize = 3;
const MIN_SET_OVERLAP: usize = 4;

fn env_path(var: &str, what: &str) -> PathBuf {
    let p = std::env::var_os(var)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {var} to {what}"));
    assert!(p.is_file(), "{var} is not a file: {p:?}");
    p
}

fn next_port() -> u16 {
    // This binary's two tests may run concurrently: one port each, in a range
    // distinct from parity_lfm2's.
    static NEXT_PORT: std::sync::atomic::AtomicU16 = std::sync::atomic::AtomicU16::new(18481);
    NEXT_PORT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// One quantized-reference comparison: `load` builds our model from the GGUF,
/// `render` wraps the prompt in the model's chat template. Panics (test
/// style) on any divergence.
fn compare_against_llama_cpp<M, C>(
    tag: &str,
    gguf: &std::path::Path,
    load: impl FnOnce(&std::path::Path, &burn::tensor::Device<Gpu>) -> M,
    render: impl FnOnce(&str) -> String,
) where
    M: CausalLm<Gpu, Cache = C>,
{
    let exe =
        llama_ref::server_exe().expect("set MUMMU_LLAMA_SERVER to a llama.cpp llama-server binary");

    // Tokenizer from the same single file, like everything else on our side.
    let f = GgufFile::open(gguf).expect("gguf header parses");
    let tok = mummu::tokenizer::tokenizer_from_gguf(&f).expect("tokenizer from gguf metadata");
    drop(f);
    let raw = render(PROMPT);
    let ids = tok
        .encode(raw.as_str(), false)
        .expect("encodes")
        .get_ids()
        .to_vec();
    assert!(ids.len() >= 8, "rendered prompt suspiciously short");

    // Reference first: fail with the transport error, not a weights error.
    let server = LlamaServer::start(&exe, gguf, next_port()).expect("llama-server starts");
    let reference = server
        .greedy_completion(&ids, MAX_TOKENS, 10)
        .expect("reference completion");
    assert!(
        !reference.steps.is_empty(),
        "reference returned no logprob steps"
    );
    let ref_top: Vec<(u32, f64)> = reference.steps[0].iter().copied().take(TOP_K).collect();
    assert_eq!(
        ref_top.len(),
        TOP_K,
        "reference returned fewer than top-{TOP_K}"
    );

    let device = burn::tensor::Device::<Gpu>::default();
    let loaded = load(gguf, &device);
    let mut cache = loaded.new_cache();
    let logits = loaded
        .forward(&ids, 0, &mut cache, &device)
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .expect("logits readback");
    drop(cache);

    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
    let our_ids: Vec<u32> = indexed[..TOP_K].iter().map(|&(id, _)| id as u32).collect();
    let ref_ids: Vec<u32> = ref_top.iter().map(|&(id, _)| id).collect();
    let ours_lp = logprobs_at(&logits, &our_ids);
    let max_abs_diff = ours_lp
        .iter()
        .zip(ref_top.iter())
        .map(|(a, &(_, b))| (a - b).abs())
        .fold(0.0_f64, f64::max);

    let out_ids = loaded
        .greedy_generate(&ids, MAX_TOKENS, &device)
        .expect("greedy decode");
    let ours = tok.decode(&out_ids, true).expect("decode");

    eprintln!("[parity/gguf/{tag}] top-{TOP_K} ids ours: {our_ids:?}");
    eprintln!("[parity/gguf/{tag}] top-{TOP_K} ids ref : {ref_ids:?}");
    eprintln!("[parity/gguf/{tag}] max |Δlogprob| vs llama.cpp: {max_abs_diff:e}");
    eprintln!(
        "[parity/gguf/{tag}] ours      ({} tokens): {ours:?}",
        out_ids.len()
    );
    eprintln!(
        "[parity/gguf/{tag}] llama.cpp           : {:?}",
        reference.content
    );

    assert_eq!(
        &our_ids[..STRICT_ORDER_K],
        &ref_ids[..STRICT_ORDER_K],
        "top-{STRICT_ORDER_K} ids diverge from llama.cpp on the same quantized file"
    );
    let overlap = our_ids.iter().filter(|id| ref_ids.contains(id)).count();
    assert!(
        overlap >= MIN_SET_OVERLAP,
        "top-{TOP_K} sets overlap only {overlap} (need >= {MIN_SET_OVERLAP}): \
         ours {our_ids:?} vs ref {ref_ids:?}"
    );
    assert!(
        max_abs_diff <= LOGPROB_ABS_TOLERANCE,
        "logprobs diverge: max |Δ| = {max_abs_diff} > {LOGPROB_ABS_TOLERANCE}"
    );
    let (a, b) = (ours.trim(), reference.content.trim());
    let n = a.len().min(b.len());
    assert!(n >= 8, "outputs too short to compare: {n} chars");
    assert_eq!(&a[..n], &b[..n], "greedy sequences diverge");
}

#[test]
#[ignore = "needs the Qwen2.5 Q4_K_M GGUF (MUMMU_GGUF_PATH) + llama-server (MUMMU_LLAMA_SERVER)"]
fn qwen2_q4_gguf_matches_llama_cpp_on_the_same_file() {
    let gguf = env_path("MUMMU_GGUF_PATH", "the qwen2.5-1.5b-instruct q4_k_m gguf");
    compare_against_llama_cpp(
        "qwen2",
        &gguf,
        |p, d| qwen2::load_from_gguf::<Gpu>(p, d).expect("gguf load checked"),
        |user| {
            mummu::chat::ChatMl::qwen2().render(&[
                mummu::chat::Turn::system("You are a helpful assistant."),
                mummu::chat::Turn::user(user),
            ])
        },
    );
}

#[test]
#[ignore = "needs a Qwen3 Q4_K_M GGUF (MUMMU_QWEN3_GGUF_PATH) + llama-server (MUMMU_LLAMA_SERVER)"]
fn qwen3_q4_gguf_matches_llama_cpp_on_the_same_file() {
    // Qwen3 is a thinking model (emits <think>…) — the greedy leg still holds
    // byte-for-byte because both sides receive the IDENTICAL prompt-id array
    // and greedy-decode, so the reasoning tokens match too; no chat/template
    // stack runs on either side (the LFM2 raw-completion caveat applies).
    let gguf = env_path("MUMMU_QWEN3_GGUF_PATH", "a qwen3 q4_k_m gguf");
    compare_against_llama_cpp(
        "qwen3",
        &gguf,
        |p, d| qwen3::load_from_gguf::<Gpu>(p, d).expect("gguf load checked"),
        |user| {
            mummu::chat::ChatMl::qwen2().render(&[
                mummu::chat::Turn::system("You are a helpful assistant."),
                mummu::chat::Turn::user(user),
            ])
        },
    );
}

#[test]
#[ignore = "needs the LFM2.5 Q4_K_M GGUF (MUMMU_LFM2_GGUF_PATH) + llama-server (MUMMU_LLAMA_SERVER)"]
fn lfm2_q4_gguf_matches_llama_cpp_on_the_same_file() {
    let gguf = env_path(
        "MUMMU_LFM2_GGUF_PATH",
        "the lfm2.5-1.2b-instruct q4_k_m gguf",
    );
    compare_against_llama_cpp(
        "lfm2",
        &gguf,
        |p, d| lfm2::load_from_gguf::<Gpu>(p, d).expect("gguf load checked"),
        |user| mummu::chat::ChatMl::lfm2().render(&[mummu::chat::Turn::user(user)]),
    );
}
