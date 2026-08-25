//! The MoE-tier perf-budget gate from `bench/BASELINE.md`: OLMoE-1B-7B
//! greedy decode on the `Cpu` (burn-flex) backend, straight from its Q4_K_M
//! GGUF. This tier is bounded by **latency per token**, not tok/s — the
//! dense-mask expert forward touches all 7B params per token (see the
//! routed-compute item in ROADMAP P2), so a budget in seconds/token is the
//! honest unit and the number to beat when that lands.
//!
//! Ignored by default (the file is ~4.2 GB and the f32 build needs ~28 GB of
//! RAM); run with
//!
//! ```text
//! MUMMU_OLMOE_GGUF_PATH=path/to/OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf \
//!   cargo test -p mummu-bench --release --test budget_moe -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use mummu::decode::argmax_id;
use mummu::gguf::{GgufFile, GgufValue};
use mummu::models::CausalLm;
use mummu::models::olmoe;

/// From bench/BASELINE.md (recorded 0.76 s/token warm on 2026-08-03). The
/// ceiling carries ~2.6x headroom so ordinary host-load noise is not a false
/// alarm — this backend's decode tracks CPU availability closely.
const DECODE_BUDGET_SECS_PER_TOKEN: f64 = 2.0;
const DECODE_STEPS: usize = 4;

/// Loading dequantizes ~7B params to f32; on the reference machine that is
/// ~82 s. Well clear of any plausible import regression, and it catches a
/// pathological one.
const LOAD_BUDGET_SECS: f64 = 300.0;

#[tokio::test]
#[ignore = "needs the OLMoE Q4_K_M GGUF (MUMMU_OLMOE_GGUF_PATH), ~30 GB free COMMIT \
            and ~28 GB of scratch disk beside the gguf"]
async fn olmoe_moe_cpu_decode_stays_inside_its_budget() {
    let Some(path) = std::env::var_os("MUMMU_OLMOE_GGUF_PATH").map(PathBuf::from) else {
        panic!("set MUMMU_OLMOE_GGUF_PATH to the OLMoE-1B-7B q4_k_m gguf");
    };
    assert!(
        path.is_file(),
        "MUMMU_OLMOE_GGUF_PATH is not a file: {path:?}"
    );

    let f = GgufFile::open(&path).expect("gguf header parses");
    let tok = mummu::tokenizer::tokenizer_from_gguf(&f).expect("tokenizer from metadata");
    let bos = f
        .get("tokenizer.ggml.bos_token_id")
        .and_then(GgufValue::as_u64)
        .and_then(|v| {
            f.get("tokenizer.ggml.tokens")
                .and_then(GgufValue::as_array)
                .and_then(|t| t.get(usize::try_from(v).ok()?))
                .and_then(GgufValue::as_str)
                .map(String::from)
        })
        .unwrap_or_default();
    drop(f);
    let raw =
        format!("{bos}<|user|>\nWhat is 2 + 2? Answer in one short sentence.\n<|assistant|>\n");
    let ids = tok
        .encode(raw.as_str(), false)
        .expect("encodes")
        .get_ids()
        .to_vec();

    let device = mummu::backend::cpu_device();
    let start = Instant::now();
    let loaded = olmoe::load_from_gguf(&path, &device).expect("gguf load checked");
    let load_secs = start.elapsed().as_secs_f64();

    // Prefill (uncounted warm-up), then time the decode steps.
    let mut cache = loaded.new_cache();
    let logits = loaded.forward(&ids, 0, &mut cache, &device);
    let mut next = argmax_id(logits).await.expect("argmax");

    let start = Instant::now();
    let mut out = Vec::with_capacity(DECODE_STEPS);
    for past in (ids.len()..).take(DECODE_STEPS) {
        let logits = loaded.forward(&[next], past, &mut cache, &device);
        next = argmax_id(logits).await.expect("argmax");
        out.push(next);
    }
    let secs_per_token = start.elapsed().as_secs_f64() / DECODE_STEPS as f64;
    let text = tok.decode(&out, true).expect("decode");

    eprintln!(
        "[budget/moe] OLMoE-1B-7B load {load_secs:.1}s (budget {LOAD_BUDGET_SECS}), \
         decode {secs_per_token:.2} s/token (budget {DECODE_BUDGET_SECS_PER_TOKEN}); text: {text:?}"
    );
    assert!(
        text.contains('4') || text.to_lowercase().contains("four"),
        "MoE decode must stay coherent, got: {text:?}"
    );
    assert!(
        load_secs <= LOAD_BUDGET_SECS,
        "MoE load regression: {load_secs:.1}s > {LOAD_BUDGET_SECS}s budget"
    );
    assert!(
        secs_per_token <= DECODE_BUDGET_SECS_PER_TOKEN,
        "MoE decode regression: {secs_per_token:.2} s/token > {DECODE_BUDGET_SECS_PER_TOKEN} budget"
    );
}
