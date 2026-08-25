//! The **warm-up curve** the f16 budget gate only sees one point of: how many
//! decoded tokens a *cold process* needs before f16 reaches its steady-state
//! rate, and how much of that cost the persistent autotune cache already
//! carries across processes.
//!
//! `bench/BASELINE.md` records two honest f16 decode numbers that differ ~3x —
//! 16.9 tok/s from `budget_f16.rs` (one 32-step burst) and 48.8 tok/s from
//! criterion (many 32-step samples). This harness measures the whole curve in
//! ONE process instead of one point in each, so the gap stops being an
//! inference from two harnesses and becomes a measurement.
//!
//! Own test binary, like every `GpuF16` leg: one dtype alias per process (see
//! `budget_f16.rs`).
//!
//! ```text
//! MUMMU_QWEN2_DIR=path/to/qwen2.5-1.5b cargo test -p mummu-bench --release --test warmup_f16 -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use burn::tensor::DType;
use mummu::backend::inventory;
use mummu::decode::argmax_id;
use mummu::models::CausalLm;
use mummu::models::qwen2;
use tokenizers::Tokenizer;

/// Tokens per burst — the same 32 the two existing f16 rows use, so a burst
/// here is directly comparable to `budget_f16.rs`'s single measurement.
const BURST_TOKENS: usize = 32;
/// Bounded: 8 bursts = 256 decoded tokens, ~5-15 s of GPU time depending on
/// where in the curve the run sits. Long enough to reach the criterion
/// steady state, short enough that the gate stays a gate.
const BURSTS: usize = 8;
/// The steady-state floor. Measured 37.6 / 40.3 tok/s on an idle-ish machine
/// — but this row is unusually sensitive to *host* CPU contention (f16 does
/// ~3x less GPU work per dispatch than f32, so host-side sequencing is a
/// larger share of its per-token time; see `bench/BASELINE.md`), and a run
/// sharing the box with a workspace build measured 25.8. The floor is
/// therefore set just above the **f32** decode rate (~16 tok/s): what it
/// defends is the claim that steady-state f16 is meaningfully the faster path,
/// which no amount of contention should erase, rather than a specific number
/// the machine cannot promise.
const STEADY_BUDGET_TOKENS_PER_S: f64 = 20.0;

fn qwen2_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("MUMMU_QWEN2_DIR")?);
    dir.is_dir().then_some(dir)
}

#[tokio::test]
#[ignore = "needs multi-GB local weights (MUMMU_QWEN2_DIR) + a SHADER_F16 GPU"]
async fn f16_decode_warms_up_within_a_bounded_number_of_tokens() {
    let Some(dir) = qwen2_dir() else {
        panic!("set MUMMU_QWEN2_DIR to a dir with config.json/tokenizer.json/model.safetensors");
    };
    if !inventory().any_shader_f16() {
        eprintln!("[warmup/f16] no SHADER_F16 adapter — skipping");
        return;
    }
    let text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nExplain, in three sentences, why the sky is blue.<|im_end|>\n<|im_start|>assistant\n";
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json loads");
    let ids = tok.encode(text, false).expect("encodes").get_ids().to_vec();
    assert!(ids.len() >= 16, "warm-up prompt suspiciously short");

    let device =
        mummu::backend::gpu_device_f16().expect("f16 device settings lock once per process");
    let load_start = Instant::now();
    let loaded = qwen2::load_from_dir(&dir, &device).expect("weights load checked");
    let load_s = load_start.elapsed().as_secs_f64();

    // NO warm-up pass: the cold cost is the measurement. The first prefill is
    // timed like any other, and its logits carry the dtype proof this binary
    // exists for.
    let start = Instant::now();
    let mut cache = loaded.new_cache();
    let logits = loaded.forward(&ids, 0, &mut cache, &device);
    assert_eq!(
        logits.dtype(),
        DType::F16,
        "this harness must measure f16: a device policy locked by another alias would \
         silently make these f32 numbers"
    );
    let cold_top_id = argmax_id(logits).await.expect("argmax");
    let cold_ttft_ms = start.elapsed().as_secs_f64() * 1e3;

    // Each burst reproduces criterion's `decode_32_tokens` sample exactly:
    // fresh cache, untimed prefill, then 32 timed steps. Decoding 256
    // CONSECUTIVE tokens instead would grow the KV cache 8x across the run and
    // confound warm-up with attention length — measured, and it does: the
    // consecutive form's last burst reads ~20% below its own plateau.
    let mut rates = Vec::with_capacity(BURSTS);
    for _ in 0..BURSTS {
        let mut cache = loaded.new_cache();
        let logits = loaded.forward(&ids, 0, &mut cache, &device);
        let mut next = argmax_id(logits).await.expect("argmax");
        assert_eq!(
            next, cold_top_id,
            "every burst prefills the same prompt, so its first token must not change"
        );
        let start = Instant::now();
        let mut past = ids.len();
        for _ in 0..BURST_TOKENS {
            let logits = loaded.forward(&[next], past, &mut cache, &device);
            next = argmax_id(logits).await.expect("argmax");
            past += 1;
        }
        let elapsed = start.elapsed().as_secs_f64();
        assert!(elapsed > 0.0, "a burst cannot take zero time");
        assert_eq!(
            past,
            ids.len() + BURST_TOKENS,
            "cache position must advance one step per decoded token"
        );
        rates.push(BURST_TOKENS as f64 / elapsed);
    }
    assert_eq!(rates.len(), BURSTS, "every burst must record a rate");

    let first = rates[0];
    let steady = rates[BURSTS - 1];
    let curve: Vec<String> = rates
        .iter()
        .enumerate()
        .map(|(i, r)| format!("{}:{r:.1}", (i + 1) * BURST_TOKENS))
        .collect();
    eprintln!(
        "[warmup/f16] load {load_s:.1} s, cold TTFT {cold_ttft_ms:.1} ms, \
         burst tok/s by cumulative token — {}",
        curve.join(" ")
    );
    eprintln!(
        "[warmup/f16] first burst {first:.1} tok/s, steady {steady:.1} tok/s, \
         warm-up cost {:.2}x (budget: steady >= {STEADY_BUDGET_TOKENS_PER_S} tok/s)",
        steady / first
    );

    assert!(
        steady >= STEADY_BUDGET_TOKENS_PER_S,
        "f16 steady-state regression: {steady:.1} tok/s < {STEADY_BUDGET_TOKENS_PER_S} tok/s budget"
    );
    assert!(
        steady >= first,
        "warm-up must not run backwards: steady {steady:.1} tok/s < first burst {first:.1} tok/s"
    );
}
