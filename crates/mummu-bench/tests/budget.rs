//! The perf-budget gate from `bench/BASELINE.md`: fails when TTFT or decode
//! throughput for Qwen2.5-1.5B regresses past its budget on the reference
//! GPU. Ignored by default (multi-GB weights + a real GPU); run with
//!
//! ```text
//! MUMMU_QWEN2_DIR=path/to/qwen2.5-1.5b cargo test -p mummu-bench --release -- --ignored --nocapture
//! ```
//!
//! Budgets are deliberately looser than the recorded numbers (see
//! BASELINE.md) so machine jitter doesn't flake the gate while a real
//! regression (a kernel falling off a fast path, an accidental sync per
//! layer) still trips it.

use std::path::PathBuf;
use std::time::Instant;

use mummu::backend::Gpu;
use mummu::decode::argmax_id;
use mummu::models::CausalLm;
use mummu::models::qwen2;
use tokenizers::Tokenizer;

/// From bench/BASELINE.md (recorded 100.5 ms / 13.3 tok/s on 2026-07-10).
const TTFT_BUDGET_MS: f64 = 150.0;
const DECODE_BUDGET_TOKENS_PER_S: f64 = 10.0;
const DECODE_STEPS: usize = 32;

/// Long-context prefill: the row where the attention formulation shows up at
/// all (explicit attention materializes a `[1, heads, t, t]` scores tensor —
/// 201 MiB of f32 here, against 62 KiB at the ~36-token prompt). Budget is
/// ~1.5x the recorded number, matching the looseness of the rows above.
const LONG_PREFILL_TOKENS: usize = 2048;
const LONG_PREFILL_BUDGET_MS: f64 = 900.0;

fn qwen2_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("MUMMU_QWEN2_DIR")?);
    dir.is_dir().then_some(dir)
}

#[test]
#[ignore = "needs multi-GB local weights (MUMMU_QWEN2_DIR) + the reference GPU"]
fn qwen2_stays_inside_its_perf_budgets() {
    let Some(dir) = qwen2_dir() else {
        panic!("set MUMMU_QWEN2_DIR to a dir with config.json/tokenizer.json/model.safetensors");
    };
    let text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nExplain, in three sentences, why the sky is blue.<|im_end|>\n<|im_start|>assistant\n";
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json loads");
    let ids = tok.encode(text, false).expect("encodes").get_ids().to_vec();
    assert!(ids.len() >= 16, "budget prompt suspiciously short");

    let device = burn::tensor::Device::<Gpu>::default();
    let loaded = qwen2::load_from_dir::<Gpu>(&dir, &device).expect("weights load checked");

    // Warm-up: first-run autotune + pipeline compilation must not count.
    let mut cache = loaded.new_cache();
    let logits = loaded.forward(&ids, 0, &mut cache, &device);
    let _ = argmax_id(logits).expect("warm-up argmax");

    // TTFT: fresh cache, full prefill, first token (argmax readback = sync).
    let start = Instant::now();
    let mut cache = loaded.new_cache();
    let logits = loaded.forward(&ids, 0, &mut cache, &device);
    let mut next = argmax_id(logits).expect("argmax");
    let ttft_ms = start.elapsed().as_secs_f64() * 1e3;

    // Decode throughput over a warm cache.
    let start = Instant::now();
    for past in (ids.len()..).take(DECODE_STEPS) {
        let logits = loaded.forward(&[next], past, &mut cache, &device);
        next = argmax_id(logits).expect("argmax");
    }
    let tok_per_s = DECODE_STEPS as f64 / start.elapsed().as_secs_f64();

    // Long-context prefill over the same weights, prompt tiled to length.
    let long_ids: Vec<u32> = ids
        .iter()
        .cycle()
        .take(LONG_PREFILL_TOKENS)
        .copied()
        .collect();
    assert_eq!(long_ids.len(), LONG_PREFILL_TOKENS);
    let mut long_cache = loaded.new_cache();
    let _ = argmax_id(loaded.forward(&long_ids, 0, &mut long_cache, &device)).expect("warm-up");
    let start = Instant::now();
    let mut long_cache = loaded.new_cache();
    let logits = loaded.forward(&long_ids, 0, &mut long_cache, &device);
    let _ = argmax_id(logits).expect("argmax");
    let long_prefill_ms = start.elapsed().as_secs_f64() * 1e3;

    eprintln!(
        "[budget] TTFT {ttft_ms:.1} ms (budget {TTFT_BUDGET_MS} ms), \
         decode {tok_per_s:.1} tok/s (budget {DECODE_BUDGET_TOKENS_PER_S} tok/s), \
         prefill@{LONG_PREFILL_TOKENS} {long_prefill_ms:.0} ms \
         (budget {LONG_PREFILL_BUDGET_MS} ms)"
    );
    assert!(
        ttft_ms <= TTFT_BUDGET_MS,
        "TTFT regression: {ttft_ms:.1} ms > {TTFT_BUDGET_MS} ms budget"
    );
    assert!(
        tok_per_s >= DECODE_BUDGET_TOKENS_PER_S,
        "decode regression: {tok_per_s:.1} tok/s < {DECODE_BUDGET_TOKENS_PER_S} tok/s budget"
    );
    assert!(
        long_prefill_ms <= LONG_PREFILL_BUDGET_MS,
        "long-prefill regression: {long_prefill_ms:.0} ms > {LONG_PREFILL_BUDGET_MS} ms budget"
    );
}
