//! **f16 parity leg** — the trust gate the f16 path never had.
//!
//! Every strict gate in the repo runs f32 (or GGUF-dequantized-to-f32): the
//! Candle logits fixture, the llama.cpp GGUF legs, the Ollama greedy leg. The
//! f16 path was validated only for *liveness* (`real_f16.rs`: no NaN, right
//! VRAM, coherent text) and for agreeing with f32 on ONE token in one process
//! (`real_mixed_dtype.rs`). That is not parity, and it is why the 2026-08-06
//! flash-attention evaluation could not adopt its one winning quadrant (f16
//! prefill) — an f16-only numeric fork would have shipped unverified.
//!
//! This leg closes that: llama.cpp runs the SAME Q4_K_M file, our side loads
//! it onto **`GpuF16`** (dequantize once, cast to f16 on load, f32 attention
//! -score island), and the two are compared by the same top-k + byte-identical
//! greedy asserts as every other port. Own test binary because `GpuF16` locks
//! Burn's per-device dtype policy — one alias per process (see
//! `mummu-bench/tests/budget_f16.rs`).
//!
//! ```text
//! MUMMU_GGUF_PATH=path/to/qwen2.5-1.5b-instruct-q4_k_m.gguf \
//! MUMMU_QWEN3_GGUF_PATH=path/to/qwen3-0.6b-q4_k_m.gguf \
//! MUMMU_LLAMA_SERVER=path/to/llama-server.exe \
//!   cargo test -p mummu --release --test parity_f16 -- --ignored --nocapture
//! ```

mod gguf_compare;
mod llama_ref;

use std::path::PathBuf;

use gguf_compare::{compare_against_llama_cpp, next_port};
use mummu::backend::{GpuF16, inventory};
use mummu::models::{qwen2, qwen3};

/// Max |Δlogprob| for the f16 legs. The f32 legs run at 7.5e-1 against
/// measured 2.66e-1 (Qwen2) / 4.02e-1 (Qwen3); f16 adds its own rounding on
/// top of the reference's Q8_K activation quantization, so this sits one step
/// looser. The primary assert is unchanged and unrelaxed: the top-3 ids in
/// order and the 24-token greedy sequence byte-identical.
const LOGPROB_ABS_TOLERANCE: f64 = 1.5e0;

/// Port range for this binary; distinct from `parity_gguf`'s and
/// `parity_lfm2`'s so a concurrent run never collides.
const PORT_BASE: u16 = 18501;

fn env_path(var: &str, what: &str) -> PathBuf {
    let p = std::env::var_os(var)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {var} to {what}"));
    assert!(p.is_file(), "{var} is not a file: {p:?}");
    p
}

/// Skip (rather than fail) on a machine whose GPU cannot do f16 at all.
fn shader_f16_or_skip(tag: &str) -> bool {
    if inventory().any_shader_f16() {
        return true;
    }
    eprintln!("[parity/gguf/{tag}] no SHADER_F16 adapter — skipping");
    false
}

#[test]
#[ignore = "needs the Qwen2.5 Q4_K_M GGUF (MUMMU_GGUF_PATH), llama-server \
            (MUMMU_LLAMA_SERVER), and a SHADER_F16 GPU"]
fn qwen2_q4_gguf_matches_llama_cpp_in_f16() {
    if !shader_f16_or_skip("qwen2-f16") {
        return;
    }
    let gguf = env_path("MUMMU_GGUF_PATH", "the qwen2.5-1.5b-instruct q4_k_m gguf");
    compare_against_llama_cpp(
        "qwen2-f16",
        &gguf,
        next_port(PORT_BASE),
        LOGPROB_ABS_TOLERANCE,
        |p, d| qwen2::load_from_gguf::<GpuF16>(p, d).expect("gguf load checked"),
        |user| {
            mummu::chat::ChatMl::qwen2().render(&[
                mummu::chat::Turn::system("You are a helpful assistant."),
                mummu::chat::Turn::user(user),
            ])
        },
    );
}

#[test]
#[ignore = "needs a Qwen3 Q4_K_M GGUF (MUMMU_QWEN3_GGUF_PATH), llama-server \
            (MUMMU_LLAMA_SERVER), and a SHADER_F16 GPU"]
fn qwen3_q4_gguf_matches_llama_cpp_in_f16() {
    if !shader_f16_or_skip("qwen3-f16") {
        return;
    }
    let gguf = env_path("MUMMU_QWEN3_GGUF_PATH", "a qwen3 q4_k_m gguf");
    compare_against_llama_cpp(
        "qwen3-f16",
        &gguf,
        next_port(PORT_BASE),
        LOGPROB_ABS_TOLERANCE,
        |p, d| qwen3::load_from_gguf::<GpuF16>(p, d).expect("gguf load checked"),
        |user| {
            mummu::chat::ChatMl::qwen2().render(&[
                mummu::chat::Turn::system("You are a helpful assistant."),
                mummu::chat::Turn::user(user),
            ])
        },
    );
}
