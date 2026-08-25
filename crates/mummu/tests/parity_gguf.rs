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

mod gguf_compare;
mod llama_ref;

use std::path::PathBuf;

use gguf_compare::{compare_against_llama_cpp, next_port};

use mummu::gguf::GgufFile;
use mummu::models::{lfm2, olmoe, qwen2, qwen3};

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

/// Port range for this binary's servers; distinct from `parity_lfm2`'s and
/// `parity_f16`'s so concurrent legs never collide.
const PORT_BASE: u16 = 18481;

fn env_path(var: &str, what: &str) -> PathBuf {
    let p = std::env::var_os(var)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {var} to {what}"));
    assert!(p.is_file(), "{var} is not a file: {p:?}");
    p
}

#[tokio::test]
#[ignore = "needs the Qwen2.5 Q4_K_M GGUF (MUMMU_GGUF_PATH) + llama-server (MUMMU_LLAMA_SERVER)"]
async fn qwen2_q4_gguf_matches_llama_cpp_on_the_same_file() {
    let gguf = env_path("MUMMU_GGUF_PATH", "the qwen2.5-1.5b-instruct q4_k_m gguf");
    compare_against_llama_cpp(
        "qwen2",
        &gguf,
        next_port(PORT_BASE),
        LOGPROB_ABS_TOLERANCE,
        &mummu::backend::gpu_device(),
        |p, d| qwen2::load_from_gguf(p, d).expect("gguf load checked"),
        |user| {
            mummu::chat::ChatMl::qwen2().render(&[
                mummu::chat::Turn::system("You are a helpful assistant."),
                mummu::chat::Turn::user(user),
            ])
        },
    )
    .await;
}

#[tokio::test]
#[ignore = "needs a Qwen3 Q4_K_M GGUF (MUMMU_QWEN3_GGUF_PATH) + llama-server (MUMMU_LLAMA_SERVER)"]
async fn qwen3_q4_gguf_matches_llama_cpp_on_the_same_file() {
    // Qwen3 is a thinking model (emits <think>…) — the greedy leg still holds
    // byte-for-byte because both sides receive the IDENTICAL prompt-id array
    // and greedy-decode, so the reasoning tokens match too; no chat/template
    // stack runs on either side (the LFM2 raw-completion caveat applies).
    let gguf = env_path("MUMMU_QWEN3_GGUF_PATH", "a qwen3 q4_k_m gguf");
    compare_against_llama_cpp(
        "qwen3",
        &gguf,
        next_port(PORT_BASE),
        LOGPROB_ABS_TOLERANCE,
        &mummu::backend::gpu_device(),
        |p, d| qwen3::load_from_gguf(p, d).expect("gguf load checked"),
        |user| {
            mummu::chat::ChatMl::qwen2().render(&[
                mummu::chat::Turn::system("You are a helpful assistant."),
                mummu::chat::Turn::user(user),
            ])
        },
    )
    .await;
}

#[tokio::test]
#[ignore = "needs the LFM2.5 Q4_K_M GGUF (MUMMU_LFM2_GGUF_PATH) + llama-server (MUMMU_LLAMA_SERVER)"]
async fn lfm2_q4_gguf_matches_llama_cpp_on_the_same_file() {
    let gguf = env_path(
        "MUMMU_LFM2_GGUF_PATH",
        "the lfm2.5-1.2b-instruct q4_k_m gguf",
    );
    compare_against_llama_cpp(
        "lfm2",
        &gguf,
        next_port(PORT_BASE),
        LOGPROB_ABS_TOLERANCE,
        &mummu::backend::gpu_device(),
        |p, d| lfm2::load_from_gguf(p, d).expect("gguf load checked"),
        |user| mummu::chat::ChatMl::lfm2().render(&[mummu::chat::Turn::user(user)]),
    )
    .await;
}

#[tokio::test]
#[ignore = "needs the OLMoE Q4_K_M GGUF (MUMMU_OLMOE_GGUF_PATH), llama-server \
            (MUMMU_LLAMA_SERVER), ~30 GB free COMMIT and ~28 GB of scratch disk \
            beside the gguf (28 GB f32 build, CPU backend)"]
async fn olmoe_q4_gguf_matches_llama_cpp_on_the_same_file() {
    // The zoo's first MoE leg. Our side runs on the CPU backend — the ~28 GB
    // f32 resident-everything build does not fit a 16 GB card; parity is
    // backend-independent by construction (same weights, same math).
    let gguf = env_path(
        "MUMMU_OLMOE_GGUF_PATH",
        "the OLMoE-1B-7B-0125-Instruct q4_k_m gguf",
    );
    // OLMoE has no hardcoded ChatMl renderer yet: render its zephyr-style
    // template (from the GGUF's own chat_template metadata) by hand, BOS
    // first — both sides get the identical id array, so no template stack is
    // in the loop on either side.
    let f = GgufFile::open(&gguf).expect("gguf header parses");
    let bos = f
        .get("tokenizer.ggml.bos_token_id")
        .and_then(mummu::gguf::GgufValue::as_u64)
        .and_then(|v| {
            f.get("tokenizer.ggml.tokens")
                .and_then(mummu::gguf::GgufValue::as_array)
                .and_then(|t| t.get(usize::try_from(v).ok()?))
                .and_then(mummu::gguf::GgufValue::as_str)
                .map(String::from)
        })
        .unwrap_or_default();
    drop(f);
    compare_against_llama_cpp(
        "olmoe",
        &gguf,
        next_port(PORT_BASE),
        LOGPROB_ABS_TOLERANCE,
        // The host, not the card: the ~28 GB f32 build does not fit 16 GiB.
        &mummu::backend::cpu_device(),
        |p, d| olmoe::load_from_gguf(p, d).expect("gguf load checked"),
        move |user| format!("{bos}<|user|>\n{user}\n<|assistant|>\n"),
    )
    .await;
}
