//! Proof that `CausalLm::warm_up` actually pays the cold-start tax
//! `warmup_f16.rs` measures: after one warm-up call, a cold process's FIRST
//! 32-token burst already runs at steady-state speed instead of ~1/3 of it.
//!
//! Own test binary for two reasons: one dtype alias per process (see
//! `budget_f16.rs`), and — the load-bearing one — warm-up is a
//! once-per-process effect, so a second test in the same binary would find
//! the GPU already warm and prove nothing.
//!
//! ```text
//! MUMMU_QWEN2_DIR=path/to/qwen2.5-1.5b cargo test -p mummu-bench --release --test warmup_api_f16 -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use burn::tensor::DType;
use mummu::backend::inventory;
use mummu::decode::argmax_id;
use mummu::models::CausalLm;
use mummu::models::qwen2;
use tokenizers::Tokenizer;

/// Same burst size as `warmup_f16.rs` and `budget_f16.rs`, so the numbers
/// here sit on the same axis as the recorded rows.
const BURST_TOKENS: usize = 32;
/// Warm-up depth: the measured curve flattens after one 32-token burst, so
/// this is exactly the depth the harness says is needed — not a margin.
const WARM_UP_STEPS: usize = 32;
/// How close to the second burst the FIRST burst must land for the warm-up to
/// count as having worked. Un-warmed, the ratio is ~0.33 (12.5 vs 37.6 tok/s);
/// this floor sits well above that and below run-to-run jitter.
const WARMED_RATIO_FLOOR: f64 = 0.80;

fn qwen2_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("MUMMU_QWEN2_DIR")?);
    dir.is_dir().then_some(dir)
}

#[tokio::test]
#[ignore = "needs multi-GB local weights (MUMMU_QWEN2_DIR) + a SHADER_F16 GPU"]
async fn warm_up_puts_the_first_burst_at_steady_state() {
    let Some(dir) = qwen2_dir() else {
        panic!("set MUMMU_QWEN2_DIR to a dir with config.json/tokenizer.json/model.safetensors");
    };
    if !inventory().any_shader_f16() {
        eprintln!("[warmup-api/f16] no SHADER_F16 adapter — skipping");
        return;
    }
    let text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nExplain, in three sentences, why the sky is blue.<|im_end|>\n<|im_start|>assistant\n";
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json loads");
    let ids = tok.encode(text, false).expect("encodes").get_ids().to_vec();
    assert!(ids.len() >= 16, "warm-up prompt suspiciously short");

    let device =
        mummu::backend::gpu_device_f16().expect("f16 device settings lock once per process");
    let loaded = qwen2::load_from_dir(&dir, &device).expect("weights load checked");

    // The API under test — the whole cold cost is meant to land here.
    let warm_start = Instant::now();
    let forwards = loaded
        .warm_up(&ids, WARM_UP_STEPS, &device)
        .await
        .expect("warm-up runs");
    let warm_s = warm_start.elapsed().as_secs_f64();
    assert_eq!(
        forwards,
        WARM_UP_STEPS + 1,
        "warm_up must report one prefill plus its decode steps"
    );

    let mut rates = [0.0f64; 2];
    for rate in &mut rates {
        let mut cache = loaded.new_cache();
        let logits = loaded.forward(&ids, 0, &mut cache, &device);
        assert_eq!(
            logits.dtype(),
            DType::F16,
            "this harness must measure f16: a device policy locked by another alias would \
             silently make these f32 numbers"
        );
        let mut next = argmax_id(logits).await.expect("argmax");
        let start = Instant::now();
        for past in (ids.len()..).take(BURST_TOKENS) {
            let logits = loaded.forward(&[next], past, &mut cache, &device);
            next = argmax_id(logits).await.expect("argmax");
        }
        let elapsed = start.elapsed().as_secs_f64();
        assert!(elapsed > 0.0, "a burst cannot take zero time");
        *rate = BURST_TOKENS as f64 / elapsed;
    }

    let [first, second] = rates;
    let ratio = first / second;
    eprintln!(
        "[warmup-api/f16] warm_up {warm_s:.2} s ({forwards} forwards), \
         then burst 1 {first:.1} tok/s vs burst 2 {second:.1} tok/s = {ratio:.2}x \
         (floor {WARMED_RATIO_FLOOR}x; un-warmed this ratio is ~0.33x)"
    );
    assert!(
        ratio >= WARMED_RATIO_FLOOR,
        "warm_up did not warm the decode path: first burst {first:.1} tok/s is only \
         {ratio:.2}x the second burst's {second:.1} tok/s"
    );
}
