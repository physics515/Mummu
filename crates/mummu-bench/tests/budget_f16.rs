//! The **f16** perf-budget gate from `bench/BASELINE.md`: Qwen2.5-1.5B TTFT
//! and greedy decode on `GpuF16`. Ignored by default (multi-GB weights + a
//! SHADER_F16 GPU); run with
//!
//! ```text
//! MUMMU_QWEN2_DIR=path/to/qwen2.5-1.5b cargo test -p mummu-bench --release --test budget_f16 -- --ignored --nocapture
//! ```
//!
//! This lives in its OWN test binary on purpose, and that is the whole
//! reason it exists. Burn resolves an unspecified tensor dtype against a
//! per-DEVICE policy that the first alias to touch the device locks; a
//! process that builds `Gpu` before `GpuF16` therefore ran "f16" work in f32
//! for weeks, and the criterion bench — which benches both aliases in one
//! process — recorded f16 rows that matched f32 to a tenth of a millisecond
//! and were believed (see the 2026-08-06 bisect in `bench/BASELINE.md`). One
//! alias per process makes that impossible, and the dtype assert below makes
//! it loud if it ever becomes possible again: a gate that cannot tell which
//! precision it measured is not a gate.

use std::path::PathBuf;
use std::time::Instant;

use burn::tensor::DType;
use mummu::backend::{GpuF16, inventory};
use mummu::decode::argmax_id;
use mummu::models::CausalLm;
use mummu::models::qwen2;
use tokenizers::Tokenizer;

/// Budgets are set against what THIS harness measures (25 ms / 16.4 tok/s on
/// 2026-08-06), not against criterion's steady-state f16 row (20.4 ms /
/// 48.8 tok/s) — the two differ by ~3x on decode because a single 32-step
/// burst never leaves autotune warm-up, and the f16 path has far more
/// variants to tune than f32. Both numbers are honest about different
/// things: criterion's is steady-state throughput, this one is roughly what
/// the first 32 tokens of a cold session cost. Budgets sit ~0.75x / 2.4x off
/// the measurement, matching the f32 gate's looseness.
const TTFT_BUDGET_MS: f64 = 60.0;
const DECODE_BUDGET_TOKENS_PER_S: f64 = 12.0;
const DECODE_STEPS: usize = 32;

fn qwen2_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("MUMMU_QWEN2_DIR")?);
    dir.is_dir().then_some(dir)
}

#[test]
#[ignore = "needs multi-GB local weights (MUMMU_QWEN2_DIR) + a SHADER_F16 GPU"]
fn qwen2_f16_stays_inside_its_perf_budgets() {
    let Some(dir) = qwen2_dir() else {
        panic!("set MUMMU_QWEN2_DIR to a dir with config.json/tokenizer.json/model.safetensors");
    };
    if !inventory().any_shader_f16() {
        eprintln!("[budget/f16] no SHADER_F16 adapter — skipping");
        return;
    }
    let text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nExplain, in three sentences, why the sky is blue.<|im_end|>\n<|im_start|>assistant\n";
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json loads");
    let ids = tok.encode(text, false).expect("encodes").get_ids().to_vec();
    assert!(ids.len() >= 16, "budget prompt suspiciously short");

    let device = burn::tensor::Device::<GpuF16>::default();
    let loaded = qwen2::load_from_dir::<GpuF16>(&dir, &device).expect("weights load checked");

    // Warm-up: first-run autotune + pipeline compilation must not count.
    // TWO passes, not one — a single f16 prefill leaves the prefill kernels
    // still re-tuning and the TTFT reading swings 24 -> 110 ms between runs.
    // The first pass also produces the tensor whose dtype proves this really
    // is the f16 path — the assert this whole binary exists for.
    for pass in 0..2 {
        let mut cache = loaded.new_cache();
        let logits = loaded.forward(&ids, 0, &mut cache, &device);
        if pass == 0 {
            assert_eq!(
                logits.dtype(),
                DType::F16,
                "this gate must measure f16: a device policy locked by another alias would \
                 silently make these f32 numbers"
            );
        }
        let _ = argmax_id(logits).expect("warm-up argmax");
    }

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

    eprintln!(
        "[budget/f16] TTFT {ttft_ms:.1} ms (budget {TTFT_BUDGET_MS} ms), \
         decode {tok_per_s:.1} tok/s (budget {DECODE_BUDGET_TOKENS_PER_S} tok/s)"
    );
    assert!(
        ttft_ms <= TTFT_BUDGET_MS,
        "f16 TTFT regression: {ttft_ms:.1} ms > {TTFT_BUDGET_MS} ms budget"
    );
    assert!(
        tok_per_s >= DECODE_BUDGET_TOKENS_PER_S,
        "f16 decode regression: {tok_per_s:.1} tok/s < {DECODE_BUDGET_TOKENS_PER_S} tok/s budget"
    );
}
