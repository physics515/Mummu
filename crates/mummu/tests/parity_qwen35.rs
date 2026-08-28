//! qwen35 (Qwen3.5/3.8 hybrid Gated-DeltaNet) parity gate vs a
//! **same-weights llama.cpp reference** — the only implementation with local
//! parity available for this family (no Candle port; Ollama runs the same
//! llama.cpp underneath).
//!
//! One BF16 GGUF is both sides' source: mummu dequantizes it to f32 and runs
//! on the default GPU; llama-server serves the identical file on CPU.
//! Prompts travel as token-id arrays from the GGUF's own tokenizer, never
//! through llama.cpp's template stack. Ignored by default; run with
//!
//! ```text
//! MUMMU_QWEN35_GGUF=path/to/Qwen3.5-2B-BF16.gguf \
//! MUMMU_LLAMA_SERVER=path/to/llama-server.exe \
//!   cargo test -p mummu --release --test parity_qwen35 -- --ignored --nocapture
//! ```

mod llama_ref;

use std::path::PathBuf;

use llama_ref::{LlamaServer, logprobs_at};
use mummu::models::CausalLm;
use mummu::models::qwen35::{self, LoadedQwen35};
use tokenizers::Tokenizer;

/// One model for both legs (the real_inference pattern): a 2B at f32 is
/// ~7.5 GB and CubeCL retains freed pool memory per device, so two
/// independent loads in one sequential run exceed the 16 GB reference card.
static QWEN35_SLOT: mummu::cache::ModelSlot<LoadedQwen35> = mummu::cache::ModelSlot::new();

/// Run `f` with the shared GPU model (loading it on first use).
/// The shared GPU model as a guard the caller awaits through — `ModelSlot::with`
/// takes a sync closure, and generation is a future under burn 0.22.
async fn gpu_model(gguf: &std::path::Path) -> mummu::cache::SlotGuard<'static, LoadedQwen35> {
    let device = mummu::backend::gpu_device();
    QWEN35_SLOT
        .acquire(gguf, |p| qwen35::load_from_gguf(p, &device))
        .await
        .expect("weights load checked")
}

const PROMPT: &str = "List the first five prime numbers.";
const MAX_TOKENS: usize = 24;
const TOP_K: usize = 5;

/// Same rationale as the LFM2 gate: the reference's ggml BF16 kernels round
/// activations per dot product; our path upcasts the weights once and stays
/// f32. The strict-order id match is the primary assert. Measured on the dev
/// GPU (4070 Ti SUPER, Vulkan/SPIR-V) 2026-08-21: max |Δlogprob| 5.02e-2
/// with top-5 ids AND a 24-token greedy sequence exactly identical — a hair
/// past LFM2's 5e-2, consistent with this family's wider heads (256) and
/// 24-layer recurrence accumulating more of the reference's per-dot bf16
/// rounding. 7.5e-2 is ~1.5x headroom on the measurement.
const LOGPROB_ABS_TOLERANCE: f64 = 7.5e-2;

fn reference_gguf() -> PathBuf {
    let p = std::env::var_os("MUMMU_QWEN35_GGUF")
        .map(PathBuf::from)
        .expect("set MUMMU_QWEN35_GGUF to a qwen35 BF16 GGUF");
    assert!(p.is_file(), "MUMMU_QWEN35_GGUF is not a file: {p:?}");
    p
}

fn server(gguf: &std::path::Path) -> LlamaServer {
    let exe =
        llama_ref::server_exe().expect("set MUMMU_LLAMA_SERVER to a llama.cpp llama-server binary");
    static NEXT_PORT: std::sync::atomic::AtomicU16 = std::sync::atomic::AtomicU16::new(18491);
    let port = NEXT_PORT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    LlamaServer::start(&exe, gguf, port).expect("llama-server starts")
}

/// ChatML with Qwen3's think conventions — what this family's imported
/// template renders for a plain text turn.
fn prompt_ids(gguf: &std::path::Path) -> (Tokenizer, Vec<u32>) {
    let f = mummu::gguf::GgufFile::open(gguf).expect("gguf opens");
    let tok = mummu::tokenizer::tokenizer_from_gguf(&f).expect("tokenizer from gguf");
    let rendered = mummu::chat::ChatMl::qwen3().render(&[mummu::chat::Turn::user(PROMPT)]);
    let ids = tok
        .encode(rendered.as_str(), false)
        .expect("encodes")
        .get_ids()
        .to_vec();
    assert!(ids.len() >= 8, "rendered prompt suspiciously short");
    (tok, ids)
}

/// Leg 1 — the first forward's top-k must match the reference in order.
#[tokio::test]
#[ignore = "needs a qwen35 BF16 GGUF (MUMMU_QWEN35_GGUF) + llama-server (MUMMU_LLAMA_SERVER)"]
async fn qwen35_first_forward_topk_matches_llama_cpp() {
    let gguf = reference_gguf();
    let (_tok, ids) = prompt_ids(&gguf);

    let server = server(&gguf);
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

    let device = mummu::backend::gpu_device();
    let logits = {
        let loaded = gpu_model(&gguf).await;
        let mut cache = loaded.new_cache();
        loaded
            .forward(&ids, 0, &mut cache, &device)
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .expect("logits readback")
    };

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

    eprintln!("[parity/qwen35] top-{TOP_K} ids ours: {our_ids:?}");
    eprintln!("[parity/qwen35] top-{TOP_K} ids ref : {ref_ids:?}");
    eprintln!("[parity/qwen35] max |Δlogprob| vs llama.cpp: {max_abs_diff:e}");

    assert_eq!(
        our_ids, ref_ids,
        "top-{TOP_K} ids diverge from the llama.cpp reference"
    );
    assert!(
        max_abs_diff <= LOGPROB_ABS_TOLERANCE,
        "logprobs diverge: max |Δ| = {max_abs_diff} > {LOGPROB_ABS_TOLERANCE}"
    );
}

/// Leg 2 — a 24-token greedy sequence must match the reference token by token.
#[tokio::test]
#[ignore = "needs a qwen35 BF16 GGUF (MUMMU_QWEN35_GGUF) + llama-server (MUMMU_LLAMA_SERVER)"]
async fn qwen35_greedy_sequence_matches_llama_cpp() {
    let gguf = reference_gguf();
    let (tok, ids) = prompt_ids(&gguf);

    let server = server(&gguf);
    let reference = server
        .greedy_completion(&ids, MAX_TOKENS, 1)
        .expect("reference completion");
    let ref_ids: Vec<u32> = reference
        .steps
        .iter()
        .map(|step| step.first().expect("every step has a winner").0)
        .collect();

    let device = mummu::backend::gpu_device();
    let ours = gpu_model(&gguf)
        .await
        .greedy_generate(&ids, MAX_TOKENS, &device)
        .await
        .expect("greedy decode");

    eprintln!(
        "[parity/qwen35] ours: {:?}",
        tok.decode(&ours, false).unwrap_or_default()
    );
    eprintln!(
        "[parity/qwen35] ref : {:?}",
        tok.decode(&ref_ids, false).unwrap_or_default()
    );

    // The reference may stop at EOS before MAX_TOKENS; compare the overlap
    // and require it to be substantial.
    let n = ours.len().min(ref_ids.len());
    assert!(n >= 8, "too few overlapping tokens to compare ({n})");
    assert_eq!(
        &ours[..n],
        &ref_ids[..n],
        "greedy sequences diverge within the first {n} tokens"
    );
}
