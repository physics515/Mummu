//! The CPU-tier perf-budget gate from `bench/BASELINE.md`: Qwen2.5-0.5B
//! greedy decode on the `Cpu` (burn-flex) backend. Ignored by default
//! (weights on disk); run with
//!
//! ```text
//! MUMMU_QWEN2_05B_DIR=path/to/qwen2.5-0.5b cargo test -p mummu-bench --release --test budget_cpu -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use mummu::backend::Cpu;
use mummu::decode::argmax_id;
use mummu::models::CausalLm;
use mummu::models::qwen2;
use tokenizers::Tokenizer;

/// From bench/BASELINE.md (recorded 11.7 tok/s on 2026-07-10; ~2x headroom).
const DECODE_BUDGET_TOKENS_PER_S: f64 = 6.0;
const DECODE_STEPS: usize = 8;

fn qwen2_05b_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("MUMMU_QWEN2_05B_DIR")?);
    dir.is_dir().then_some(dir)
}

#[test]
#[ignore = "needs local Qwen2.5-0.5B weights (MUMMU_QWEN2_05B_DIR)"]
fn qwen2_05b_cpu_decode_stays_inside_its_budget() {
    let Some(dir) = qwen2_05b_dir() else {
        panic!(
            "set MUMMU_QWEN2_05B_DIR to a dir with config.json/tokenizer.json/model.safetensors"
        );
    };
    let raw = mummu::chat::ChatMl::qwen2().render(&[
        mummu::chat::Turn::system("You are a concise assistant."),
        mummu::chat::Turn::user("What is 2+2? Answer in one short sentence."),
    ]);
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json loads");
    let ids = tok
        .encode(raw.as_str(), false)
        .expect("encodes")
        .get_ids()
        .to_vec();

    let device = burn::tensor::Device::<Cpu>::default();
    let loaded = qwen2::load_from_dir::<Cpu>(&dir, &device).expect("weights load checked");

    // Prefill (uncounted warm-up: first-run dispatch paths), then time decode.
    let mut cache = loaded.new_cache();
    let logits = loaded.forward(&ids, 0, &mut cache, &device);
    let mut next = argmax_id(logits).expect("argmax");

    let start = Instant::now();
    let mut out = Vec::with_capacity(DECODE_STEPS);
    for past in (ids.len()..).take(DECODE_STEPS) {
        let logits = loaded.forward(&[next], past, &mut cache, &device);
        next = argmax_id(logits).expect("argmax");
        out.push(next);
    }
    let tok_per_s = DECODE_STEPS as f64 / start.elapsed().as_secs_f64();
    let text = tok.decode(&out, true).expect("decode");

    eprintln!(
        "[budget/cpu] 0.5B decode {tok_per_s:.2} tok/s on flex (budget {DECODE_BUDGET_TOKENS_PER_S}); text: {text:?}"
    );
    assert!(
        text.contains('4'),
        "CPU decode must stay coherent, got: {text:?}"
    );
    assert!(
        tok_per_s >= DECODE_BUDGET_TOKENS_PER_S,
        "CPU decode regression: {tok_per_s:.2} tok/s < {DECODE_BUDGET_TOKENS_PER_S} tok/s budget"
    );
}
