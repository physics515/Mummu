//! The shared quantized-reference comparison used by every GGUF parity leg:
//! llama.cpp running the SAME .gguf file our loader loads, compared by top-k
//! first-forward ids and a byte-identical greedy sequence.
//!
//! Lives beside `llama_ref` rather than inside it because `parity_lfm2.rs`
//! drives the reference server directly (its BF16-vs-safetensors legs are
//! shaped differently) and only needs the transport half.

use mummu::gguf::GgufFile;
use mummu::models::CausalLm;

use crate::llama_ref::{LlamaServer, logprobs_at, server_exe};
/// The shared GGUF prompt for every quantized-reference leg.
pub const PROMPT: &str = "List the first five prime numbers.";
/// Greedy tokens compared byte-for-byte.
pub const MAX_TOKENS: usize = 24;
/// Top-k compared on the first forward.
pub const TOP_K: usize = 5;
/// The reference's activation-quantization noise reshuffles near-ties deep in
/// the top-k (measured: LFM2.5 swaps ranks 4-5), so the strict-order assert
/// covers the top 3 and the rest is set overlap.
pub const STRICT_ORDER_K: usize = 3;
/// Minimum top-k set overlap.
pub const MIN_SET_OVERLAP: usize = 4;

/// Next free port for a reference server; each leg gets its own so a binary's
/// tests can run concurrently.
pub fn next_port(base: u16) -> u16 {
    static NEXT: std::sync::atomic::AtomicU16 = std::sync::atomic::AtomicU16::new(0);
    base + NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// One quantized-reference comparison: `load` builds our model from the GGUF,
/// `render` wraps the prompt in the model's chat template, and `tolerance`
/// bounds |Δlogprob| over the top-k. The caller supplies the `device` — the
/// dense tiers compare on the GPU, OLMoE's ~28 GB f32 build only fits the host,
/// and the f16 leg configures an f16 GPU device from its own binary (device
/// dtype settings lock once per process). Panics (test style) on any divergence.
pub async fn compare_against_llama_cpp<M, C>(
    tag: &str,
    gguf: &std::path::Path,
    port: u16,
    tolerance: f64,
    device: &burn::tensor::Device,
    load: impl FnOnce(&std::path::Path, &burn::tensor::Device) -> M,
    render: impl FnOnce(&str) -> String,
) where
    M: CausalLm<Cache = C>,
{
    let exe = server_exe().expect("set MUMMU_LLAMA_SERVER to a llama.cpp llama-server binary");

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
    let server = LlamaServer::start(&exe, gguf, port).expect("llama-server starts");
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

    // `device` is the caller's; nothing is defaulted here any more.
    let loaded = load(gguf, device);
    let mut cache = loaded.new_cache();
    let logits = loaded
        .forward(&ids, 0, &mut cache, device)
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
        .greedy_generate(&ids, MAX_TOKENS, device)
        .await
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
        max_abs_diff <= tolerance,
        "logprobs diverge: max |Δ| = {max_abs_diff} > {tolerance}"
    );
    let (a, b) = (ours.trim(), reference.content.trim());
    let n = a.len().min(b.len());
    assert!(n >= 8, "outputs too short to compare: {n} chars");
    assert_eq!(&a[..n], &b[..n], "greedy sequences diverge");
}
