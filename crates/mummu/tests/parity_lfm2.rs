//! LFM2.5 parity gate — both legs, both size tiers — vs a **same-weights
//! llama.cpp reference**.
//!
//! No Candle port of LFM2 exists and the local Ollama `lfm2.5` tag resolves to
//! the 8.5B MoE (different weights), so the reference is llama.cpp itself
//! (`llama-server`, raw `/completion`, CPU) running LiquidAI's official
//! **BF16 GGUF** of the same checkpoint — bit-identical weights to the bf16
//! safetensors these tests load on the GPU. Prompts travel as token-id arrays
//! rendered by our byte-verified `ChatMl::lfm2()`, never through llama.cpp's
//! template stack (see the P7 roadmap notes). Ignored by default; run with
//!
//! ```text
//! MUMMU_LFM2_DIR=path/to/lfm2.5-1.2b \
//! MUMMU_LFM2_BF16_GGUF=path/to/LFM2.5-1.2B-Instruct-BF16.gguf \
//! MUMMU_LFM2_230M_DIR=path/to/lfm2.5-230m \
//! MUMMU_LFM2_230M_BF16_GGUF=path/to/LFM2.5-230M-BF16.gguf \
//! MUMMU_LLAMA_SERVER=path/to/llama-server.exe \
//!   cargo test -p mummu --release --test parity_lfm2 -- --ignored --nocapture
//! ```
//!
//! The 1.2B and 230M tiers are the same `lfm2` architecture — one hybrid
//! conv/attention loader covers both — so the legs below are written once and
//! parameterized by tier.

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
/// dev GPU (4070 Ti SUPER, Vulkan/SPIR-V): 1.49e-2 (1.2B) and 3.24e-2 (230M),
/// each with top-5 ids and a 24-token greedy sequence exactly identical. The
/// 230M drifts ~2x further — a narrower model (1024 vs 2048 hidden) spends
/// fewer accumulation steps hiding the reference's per-dot bf16 rounding — so
/// 5e-2 is the tighter tier's ~1.5x headroom, not the 1.2B's ~3x. Raise this
/// only with a measured reason; the strict-order id match is the primary
/// assert and is what actually catches a wrong port.
const LOGPROB_ABS_TOLERANCE: f64 = 5.0e-2;

/// One size tier of the same architecture: where its bf16 safetensors live and
/// which same-weights BF16 GGUF is its reference.
struct Tier {
    tag: &'static str,
    dir_env: &'static str,
    gguf_env: &'static str,
}

const LFM2_1_2B: Tier = Tier {
    tag: "lfm2/1.2b",
    dir_env: "MUMMU_LFM2_DIR",
    gguf_env: "MUMMU_LFM2_BF16_GGUF",
};

const LFM2_230M: Tier = Tier {
    tag: "lfm2/230m",
    dir_env: "MUMMU_LFM2_230M_DIR",
    gguf_env: "MUMMU_LFM2_230M_BF16_GGUF",
};

fn model_dir(tier: &Tier) -> PathBuf {
    let dir = std::env::var_os(tier.dir_env)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            panic!(
                "set {} to a dir with config.json/tokenizer.json/model.safetensors",
                tier.dir_env
            )
        });
    assert!(dir.is_dir(), "{} is not a directory: {dir:?}", tier.dir_env);
    dir
}

fn reference_gguf(tier: &Tier) -> PathBuf {
    let p = std::env::var_os(tier.gguf_env)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {} to LiquidAI's same-weights BF16 GGUF", tier.gguf_env));
    assert!(p.is_file(), "{} is not a file: {p:?}", tier.gguf_env);
    p
}

fn server(tier: &Tier) -> LlamaServer {
    let exe =
        llama_ref::server_exe().expect("set MUMMU_LLAMA_SERVER to a llama.cpp llama-server binary");
    // One port per test: the legs run concurrently in one test binary.
    static NEXT_PORT: std::sync::atomic::AtomicU16 = std::sync::atomic::AtomicU16::new(18471);
    let port = NEXT_PORT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    LlamaServer::start(&exe, &reference_gguf(tier), port).expect("llama-server starts")
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

/// Leg 1 — the first forward's top-k logits must match the reference's
/// top-k **in order**, within the reference's own bf16-activation noise.
fn first_forward_leg(tier: &Tier) {
    let dir = model_dir(tier);
    let (_tok, ids) = prompt_ids(&dir);
    let tag = tier.tag;

    // Reference first: fail with the transport error, not a weights error,
    // when the server can't come up.
    let server = server(tier);
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

    eprintln!("[parity/{tag}] top-{TOP_K} ids ours: {our_ids:?}");
    eprintln!("[parity/{tag}] top-{TOP_K} ids ref : {ref_ids:?}");
    eprintln!("[parity/{tag}] max |Δlogprob| vs llama.cpp: {max_abs_diff:e}");

    assert_eq!(
        our_ids, ref_ids,
        "top-{TOP_K} ids diverge from the llama.cpp reference"
    );
    assert!(
        max_abs_diff <= LOGPROB_ABS_TOLERANCE,
        "logprobs diverge: max |Δ| = {max_abs_diff} > {LOGPROB_ABS_TOLERANCE}"
    );
}

/// Leg 2 — a short greedy sequence must match the reference token-for-token.
fn greedy_leg(tier: &Tier) {
    let dir = model_dir(tier);
    let (tok, ids) = prompt_ids(&dir);
    let tag = tier.tag;

    let server = server(tier);
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
        "[parity/{tag}] ours      ({} tokens): {ours:?}",
        out_ids.len()
    );
    eprintln!(
        "[parity/{tag}] llama.cpp           : {:?}",
        reference.content
    );

    // Exact-match the greedy prefix over the shorter of the two (EOS or the
    // token budget may truncate one side earlier than the other).
    let (a, b) = (ours.trim(), reference.content.trim());
    let n = a.len().min(b.len());
    assert!(n >= 8, "outputs too short to compare: {n} chars");
    assert_eq!(&a[..n], &b[..n], "greedy sequences diverge");
}

#[test]
#[ignore = "needs LFM2.5-1.2B weights (MUMMU_LFM2_DIR), its BF16 GGUF (MUMMU_LFM2_BF16_GGUF) + llama-server (MUMMU_LLAMA_SERVER)"]
fn lfm2_first_forward_top_k_matches_llama_cpp_reference() {
    first_forward_leg(&LFM2_1_2B);
}

#[test]
#[ignore = "needs LFM2.5-1.2B weights (MUMMU_LFM2_DIR), its BF16 GGUF (MUMMU_LFM2_BF16_GGUF) + llama-server (MUMMU_LLAMA_SERVER)"]
fn lfm2_greedy_sequence_matches_llama_cpp_reference() {
    greedy_leg(&LFM2_1_2B);
}

#[test]
#[ignore = "needs LFM2.5-230M weights (MUMMU_LFM2_230M_DIR), its BF16 GGUF (MUMMU_LFM2_230M_BF16_GGUF) + llama-server (MUMMU_LLAMA_SERVER)"]
fn lfm2_230m_first_forward_top_k_matches_llama_cpp_reference() {
    first_forward_leg(&LFM2_230M);
}

#[test]
#[ignore = "needs LFM2.5-230M weights (MUMMU_LFM2_230M_DIR), its BF16 GGUF (MUMMU_LFM2_230M_BF16_GGUF) + llama-server (MUMMU_LLAMA_SERVER)"]
fn lfm2_230m_greedy_sequence_matches_llama_cpp_reference() {
    greedy_leg(&LFM2_230M);
}
