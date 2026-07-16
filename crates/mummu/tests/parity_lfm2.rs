//! LFM2.5 parity gate — both legs — vs a **same-weights llama.cpp reference**.
//!
//! No Candle port of LFM2 exists and the local Ollama `lfm2.5` tag resolves to
//! the 8.5B MoE (different weights), so the reference is llama.cpp itself
//! (`llama-server`, raw `/completion`, CPU) running LiquidAI's official
//! **BF16 GGUF** of LFM2.5-1.2B-Instruct — bit-identical weights to the bf16
//! safetensors this test loads on the GPU. Prompts travel as token-id arrays
//! rendered by our byte-verified `ChatMl::lfm2()`, never through llama.cpp's
//! template stack (see the P7 roadmap notes). Ignored by default; run with
//!
//! ```text
//! MUMMU_LFM2_DIR=path/to/lfm2.5-1.2b \
//! MUMMU_LFM2_BF16_GGUF=path/to/LFM2.5-1.2B-Instruct-BF16.gguf \
//! MUMMU_LLAMA_SERVER=path/to/llama-server.exe \
//!   cargo test -p mummu --release --test parity_lfm2 -- --ignored --nocapture
//! ```

mod llama_ref;

use std::path::PathBuf;

use llama_ref::{LlamaServer, logprobs_at};
use mummu::backend::Gpu;
use mummu::models::CausalLm;
use mummu::models::lfm2;
use tokenizers::Tokenizer;

const PROMPT: &str = "List the first five prime numbers.";
const MAX_TOKENS: usize = 24;
const TOP_K: usize = 5;

/// Max |Δlogprob| tolerated over the top-k between Burn (wgpu, weights upcast
/// bf16→f32 once) and llama.cpp (CPU, ggml BF16 kernels that round the
/// *activations* to bf16 per dot product — the reference's own noise floor,
/// an order ~2^-8 relative error our f32 path doesn't have). Measured on the
/// dev GPU (4070 Ti SUPER, Vulkan/SPIR-V): 1.49e-2 with top-5 ids and a
/// 24-token greedy sequence exactly identical. 5e-2 gives ~3x headroom;
/// the strict-order id match above is the primary assert.
const LOGPROB_ABS_TOLERANCE: f64 = 5.0e-2;

fn lfm2_dir() -> PathBuf {
    let dir = std::env::var_os("MUMMU_LFM2_DIR")
        .map(PathBuf::from)
        .expect("set MUMMU_LFM2_DIR to a dir with config.json/tokenizer.json/model.safetensors");
    assert!(dir.is_dir(), "MUMMU_LFM2_DIR is not a directory: {dir:?}");
    dir
}

fn reference_gguf() -> PathBuf {
    let p = std::env::var_os("MUMMU_LFM2_BF16_GGUF")
        .map(PathBuf::from)
        .expect("set MUMMU_LFM2_BF16_GGUF to LiquidAI's LFM2.5-1.2B-Instruct-BF16.gguf");
    assert!(p.is_file(), "MUMMU_LFM2_BF16_GGUF is not a file: {p:?}");
    p
}

fn server() -> (LlamaServer, u16) {
    let exe =
        llama_ref::server_exe().expect("set MUMMU_LLAMA_SERVER to a llama.cpp llama-server binary");
    // One port per test: the two legs run concurrently in one test binary.
    static NEXT_PORT: std::sync::atomic::AtomicU16 = std::sync::atomic::AtomicU16::new(18471);
    let port = NEXT_PORT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let s = LlamaServer::start(&exe, &reference_gguf(), port).expect("llama-server starts");
    (s, port)
}

/// The LFM2.5 ChatML wrapping (BOS + user turn + assistant open), rendered by
/// the library's own byte-verified template.
fn chatml(user: &str) -> String {
    mummu::chat::ChatMl::lfm2().render(&[mummu::chat::Turn::user(user)])
}

fn prompt_ids(dir: &std::path::Path) -> (Tokenizer, Vec<u32>) {
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json loads");
    let ids = tok
        .encode(chatml(PROMPT).as_str(), false)
        .expect("encodes")
        .get_ids()
        .to_vec();
    assert!(ids.len() >= 8, "rendered prompt suspiciously short");
    (tok, ids)
}

#[test]
#[ignore = "needs LFM2.5 weights (MUMMU_LFM2_DIR), the BF16 GGUF (MUMMU_LFM2_BF16_GGUF) + llama-server (MUMMU_LLAMA_SERVER)"]
fn lfm2_first_forward_top_k_matches_llama_cpp_reference() {
    let dir = lfm2_dir();
    let (_tok, ids) = prompt_ids(&dir);

    // Reference first: fail with the transport error, not a weights error,
    // when the server can't come up.
    let (server, _port) = server();
    let reference = server
        .greedy_completion(&ids, 1, 10)
        .expect("reference completion");
    assert_eq!(reference.steps.len(), 1, "asked for exactly one position");
    let ref_top: Vec<(u32, f64)> = reference.steps[0].iter().copied().take(TOP_K).collect();
    assert_eq!(
        ref_top.len(),
        TOP_K,
        "reference returned fewer than top-{TOP_K}"
    );

    let device = burn::tensor::Device::<Gpu>::default();
    let loaded = lfm2::load_from_dir::<Gpu>(&dir, &device).expect("weights load checked");
    let mut cache = loaded.new_cache();
    let logits = loaded
        .forward(&ids, 0, &mut cache, &device)
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .expect("logits readback");

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

    eprintln!("[parity/lfm2] top-{TOP_K} ids ours: {our_ids:?}");
    eprintln!("[parity/lfm2] top-{TOP_K} ids ref : {ref_ids:?}");
    eprintln!("[parity/lfm2] max |Δlogprob| vs llama.cpp: {max_abs_diff:e}");

    assert_eq!(
        our_ids, ref_ids,
        "top-{TOP_K} ids diverge from the llama.cpp reference"
    );
    assert!(
        max_abs_diff <= LOGPROB_ABS_TOLERANCE,
        "logprobs diverge: max |Δ| = {max_abs_diff} > {LOGPROB_ABS_TOLERANCE}"
    );
}

#[test]
#[ignore = "needs LFM2.5 weights (MUMMU_LFM2_DIR), the BF16 GGUF (MUMMU_LFM2_BF16_GGUF) + llama-server (MUMMU_LLAMA_SERVER)"]
fn lfm2_greedy_sequence_matches_llama_cpp_reference() {
    let dir = lfm2_dir();
    let (tok, ids) = prompt_ids(&dir);

    let (server, _port) = server();
    let reference = server
        .greedy_completion(&ids, MAX_TOKENS, 0)
        .expect("reference completion");

    let device = burn::tensor::Device::<Gpu>::default();
    let loaded = lfm2::load_from_dir::<Gpu>(&dir, &device).expect("weights load checked");
    let out_ids = loaded
        .greedy_generate(&ids, MAX_TOKENS, &device)
        .expect("greedy decode");
    let ours = tok.decode(&out_ids, true).expect("decode");

    eprintln!(
        "[parity/lfm2] ours      ({} tokens): {ours:?}",
        out_ids.len()
    );
    eprintln!(
        "[parity/lfm2] llama.cpp           : {:?}",
        reference.content
    );

    // Exact-match the greedy prefix over the shorter of the two (EOS or the
    // token budget may truncate one side earlier than the other).
    let (a, b) = (ours.trim(), reference.content.trim());
    let n = a.len().min(b.len());
    assert!(n >= 8, "outputs too short to compare: {n} chars");
    assert_eq!(&a[..n], &b[..n], "greedy sequences diverge");
}
