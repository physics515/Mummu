//! How many GPU dispatches does ONE decode token cost?
//!
//! `graph-capture-probe` measured what a captured replay saves per dispatch
//! (~8-17 us of CPU-side launch work). Converting that into ms/token needs the
//! other factor: the dispatch count of a real decode step. This measures it.
//!
//! Method — a DIFFERENTIAL, because a generation is prefill plus N decode
//! steps and only the decode part scales. Run the same prompt twice at two
//! token counts and subtract: the prefill, the model load and the warm-up are
//! identical in both, so
//!
//!     dispatches_per_token = (count_hi - count_lo) / (tokens_hi - tokens_lo)
//!
//! and every fixed cost cancels instead of having to be modelled. The count
//! itself comes from `CubeCL`'s own profiling logger at `minimal`, which logs
//! exactly the kernels that run and no timing — so this is `CubeCL`'s count, not
//! an inference of ours.
//!
//! Run (one process per token count, so each gets a clean log):
//! ```text
//! MUMMU_QWEN2_DIR=~/.cache/mummu-models/qwen2.5-1.5b-instruct \
//! CUBECL_DEBUG_LOG=/tmp/cc-8.log \
//!   cargo run --release -p mummu --example decode-dispatch-count -- 8
//! ```
//! then again with a different count, and subtract.

#![warn(clippy::pedantic, clippy::nursery, clippy::all)]

use std::path::PathBuf;

use mummu::models::CausalLm;
use mummu::models::qwen2;

/// Refuse an unbounded generation: this probe is a counter, not a benchmark,
/// and the log it drives grows with every dispatch.
const MAX_TOKENS: usize = 64;

fn main() {
    let Some(dir) = std::env::var_os("MUMMU_QWEN2_DIR").map(PathBuf::from) else {
        eprintln!("set MUMMU_QWEN2_DIR to a qwen2.5 checkpoint directory");
        return;
    };
    if !dir.is_dir() {
        eprintln!("MUMMU_QWEN2_DIR is not a directory: {}", dir.display());
        return;
    }

    let tokens: usize = match std::env::args().nth(1).map(|a| a.parse::<usize>()) {
        Some(Ok(n)) if (1..=MAX_TOKENS).contains(&n) => n,
        _ => {
            eprintln!("usage: decode-dispatch-count <tokens 1..={MAX_TOKENS}>");
            return;
        }
    };
    assert!(tokens >= 1, "token count must be positive");
    assert!(
        tokens <= MAX_TOKENS,
        "token count exceeds the {MAX_TOKENS} bound"
    );

    let device = mummu::backend::gpu_device();
    let loaded = match qwen2::load_from_dir(&dir, &device) {
        Ok(model) => model,
        Err(err) => {
            eprintln!("weights failed to load: {err}");
            return;
        }
    };

    // A fixed prompt, so the prefill is byte-identical between the two runs
    // whose counts get subtracted.
    let prompt: Vec<u32> = vec![151_644, 872, 198, 9707, 151_645, 198, 151_644, 77091, 198];

    let out = pollster::block_on(loaded.greedy_generate(&prompt, tokens, &device));
    match out {
        Ok(ids) => println!("tokens_requested={tokens} tokens_returned={}", ids.len()),
        Err(err) => eprintln!("decode failed: {err}"),
    }
}
