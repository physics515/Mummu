//! qwen4exp (Qwen3.8-Flash-Next) parity gate: our CPU (flex) forward against
//! the RECORDED llama.cpp reference in
//! `tests/fixtures/qwen4exp_ud_q4kxl_parity.json`.
//!
//! The reference cannot run beside our load on this box (see
//! `tests/qwen4exp_fixture`), so it was recorded once by
//! `parity_qwen4exp_record.rs` and is replayed here through the SAME verdict
//! every live GGUF leg uses (`gguf_compare::assert_matches_reference`, via
//! `qwen4exp_fixture::compare_leg`): top-3 strict order, top-5 overlap >= 4,
//! rank-aligned max |Δlogprob| <= 0.75, and the 24-token greedy text
//! byte-equal. Both legs run on one load; a failing leg does not hide the
//! other's numbers.
//!
//! ```text
//! MUMMU_QWEN4EXP_DIR=/home/physics515/.cache/mummu-models/qwen3.8-flash-next \
//!   cargo test -p mummu --release --test parity_qwen4exp -- --ignored --nocapture
//! ```
//!
//! Point the directory at an NVMe copy: the routed experts and PLE rows are
//! read on demand from the shards, and random reads from spinning disks are
//! ~1000x slower. The f32 trunk is ~20 GB of RAM. A missing directory SKIPS
//! with a message. `MUMMU_QWEN4EXP_TRACE=1` with the `trace_*` test prints
//! the per-layer intermediates to walk a failure to its first divergent op.

mod gguf_compare;
mod llama_ref;
mod qwen4exp_fixture;

use std::path::PathBuf;
use std::time::Instant;

use llama_ref::logprobs_at;
use mummu::gguf::GgufFile;
use mummu::models::CausalLm;
use mummu::models::qwen4exp::{self, LoadedQwen4exp};
use qwen4exp_fixture::{FIRST_SHARD, Fixture, compare_leg};

fn first_shard() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("MUMMU_QWEN4EXP_DIR")?);
    let p = dir.join(FIRST_SHARD);
    p.is_file().then_some(p)
}

/// `(VmRSS, VmHWM)` of this process in MiB, from `/proc/self/status`.
fn rss_mib() -> (u64, u64) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let field = |key: &str| {
        status
            .lines()
            .find_map(|l| l.strip_prefix(key))
            .and_then(|v| v.split_whitespace().next())
            .and_then(|kb| kb.parse::<u64>().ok())
            .map_or(0, |kb| kb / 1024)
    };
    (field("VmRSS:"), field("VmHWM:"))
}

fn load(first: &std::path::Path) -> (LoadedQwen4exp, tokenizers::Tokenizer) {
    let f = GgufFile::open_sharded(first).expect("split set opens");
    let tok = mummu::tokenizer::tokenizer_from_gguf(&f).expect("tokenizer from gguf");
    drop(f);
    let device = mummu::backend::cpu_device();
    let t0 = Instant::now();
    let model = qwen4exp::load_from_gguf(first, &device).expect("qwen4exp loads");
    let (rss, hwm) = rss_mib();
    eprintln!(
        "[parity/qwen4exp] load {:.1} s, RSS {rss} MiB, peak RSS {hwm} MiB",
        t0.elapsed().as_secs_f64()
    );
    (model, tok)
}

fn readback(t: burn::tensor::Tensor<2>) -> Vec<f32> {
    t.into_data()
        .convert::<f32>()
        .try_into_vec::<f32>()
        .expect("logits readback")
}

/// Our top-`k` ids with their log-probabilities, best first.
fn top(logits: &[f32], k: usize) -> Vec<(u32, f64)> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let ids: Vec<u32> = idx[..k].iter().map(|&i| i as u32).collect();
    ids.iter().copied().zip(logprobs_at(logits, &ids)).collect()
}

#[test]
#[ignore = "needs the Flash-Next split set on NVMe (MUMMU_QWEN4EXP_DIR) and ~25 GB RAM"]
fn qwen4exp_cpu_matches_the_recorded_llama_cpp_reference() {
    let Some(first) = first_shard() else {
        eprintln!("skipped: set MUMMU_QWEN4EXP_DIR to the shard directory");
        return;
    };
    let fx = Fixture::load();
    let (model, tok) = load(&first);
    let device = mummu::backend::cpu_device();

    let mut failures = Vec::new();
    for leg in &fx.legs {
        let t0 = Instant::now();
        let mut cache = model.new_cache();
        let logits = readback(model.forward(&leg.prompt_ids, 0, &mut cache, &device));
        let prefill = t0.elapsed().as_secs_f64();
        drop(cache);

        let t1 = Instant::now();
        let greedy = pollster::block_on(model.greedy_generate(
            &leg.prompt_ids,
            gguf_compare::MAX_TOKENS,
            &device,
        ))
        .expect("greedy decode");
        let generate = t1.elapsed().as_secs_f64();
        // generate = one more prefill + one forward per further token.
        let steps = greedy.len().saturating_sub(1).max(1);
        let (rss, hwm) = rss_mib();
        eprintln!(
            "[parity/qwen4exp/{}] prompt {} tokens: prefill {prefill:.2} s ({:.3} s/token); \
             greedy {} tokens in {generate:.1} s (~{:.2} s/decode token after its prefill); \
             RSS {rss} MiB, peak {hwm} MiB",
            leg.name,
            leg.prompt_ids.len(),
            prefill / leg.prompt_ids.len() as f64,
            greedy.len(),
            (generate - prefill).max(0.0) / steps as f64,
        );
        eprintln!(
            "[parity/qwen4exp/{}] ours top-5 {:?}",
            leg.name,
            top(&logits, 5)
        );
        eprintln!(
            "[parity/qwen4exp/{}] ref  top-5 {:?}",
            leg.name,
            leg.first_forward_top()
        );
        let first_diff = greedy.iter().zip(&leg.greedy_ids).position(|(a, b)| a != b);
        eprintln!(
            "[parity/qwen4exp/{}] greedy ids ours {greedy:?}\n[parity/qwen4exp/{}] greedy ids ref  {:?}\n\
             [parity/qwen4exp/{}] first differing position: {first_diff:?}",
            leg.name, leg.name, leg.greedy_ids, leg.name
        );

        let verdict = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            compare_leg(leg, &logits, &greedy, &tok);
        }));
        match verdict {
            Ok(()) => eprintln!("[parity/qwen4exp/{}] PASS", leg.name),
            Err(e) => {
                let msg = e
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string()))
                    .unwrap_or_else(|| "non-string panic".into());
                eprintln!("[parity/qwen4exp/{}] FAIL: {msg}", leg.name);
                failures.push(format!("{}: {msg}", leg.name));
            }
        }
    }
    assert!(failures.is_empty(), "qwen4exp parity failed: {failures:#?}");
}

/// The primes prompt's prefill only, for `MUMMU_QWEN4EXP_TRACE=1`: prints
/// the same named intermediates as the reference's
/// `ref-intermediates-summary.txt` (per-layer sums and last-token values).
#[test]
#[ignore = "debugging aid: needs MUMMU_QWEN4EXP_DIR; set MUMMU_QWEN4EXP_TRACE=1"]
fn trace_the_primes_prefill() {
    let Some(first) = first_shard() else {
        eprintln!("skipped: set MUMMU_QWEN4EXP_DIR to the shard directory");
        return;
    };
    let fx = Fixture::load();
    let leg = fx.leg("primes");
    let (model, _tok) = load(&first);
    let device = mummu::backend::cpu_device();
    let mut cache = model.new_cache();
    let t0 = Instant::now();
    let logits = readback(model.forward(&leg.prompt_ids, 0, &mut cache, &device));
    eprintln!(
        "[trace/qwen4exp] prefill {:.2} s; ours top-5 {:?}; ref top-5 {:?}",
        t0.elapsed().as_secs_f64(),
        top(&logits, 5),
        leg.first_forward_top()
    );
}

/// Is our port inside llama.cpp's noise, or off it? Samples the first
/// forward of both legs under `mummu::nn::refarith` (llama.cpp's activation
/// quantization and f16 flash attention, emulated) with tiny embedding
/// perturbations, each giving an independent noise realization, beside the
/// exact f32 path under the same perturbations as the control (which must not
/// move). Compare the printed spread with
/// `the_gate_is_tighter_than_llama_cpps_own_spread_on_this_model`'s
/// realizations. Measured 2026-09-16 (11 emulated realizations): primes
/// `<|im_end|>` mean -14.99, sd 0.49 (llama.cpp's 4 distinct: mean -15.06,
/// sd 0.33); moon mean -12.33, sd 0.24 (llama.cpp: -12.22, sd 0.30); the
/// unchanged verdict passed both legs in 5 of 11. `NOISE_SEEDS` (default 8)
/// and `NOISE_EPS` (default 1e-5) tune the sampling.
#[test]
#[ignore = "diagnostic: needs MUMMU_QWEN4EXP_DIR and ~25 GB RAM"]
fn noise_realizations_of_the_first_forward() {
    let Some(first) = first_shard() else {
        eprintln!("skipped: set MUMMU_QWEN4EXP_DIR to the shard directory");
        return;
    };
    let fx = Fixture::load();
    let (model, tok) = load(&first);
    let device = mummu::backend::cpu_device();
    let seeds: usize = std::env::var("NOISE_SEEDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let eps: f64 = std::env::var("NOISE_EPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1e-5);
    for (mode, on, n) in [("exact", false, 3usize), ("emulated", true, seeds)] {
        mummu::nn::refarith::set_enabled(on);
        for s in 0..=n {
            let seed = (s > 0).then_some(s as u64);
            mummu::nn::refarith::set_perturbation(seed, eps);
            for leg in &fx.legs {
                let t0 = Instant::now();
                let mut cache = model.new_cache();
                let logits = readback(model.forward(&leg.prompt_ids, 0, &mut cache, &device));
                let ref_top = leg.first_forward_top();
                let ref_ids: Vec<u32> = leg.steps[0].top.iter().map(|e| e.id).collect();
                let ours_at: Vec<String> = logprobs_at(&logits, &ref_ids)
                    .iter()
                    .map(|v| format!("{v:.3}"))
                    .collect();
                let verdict = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    compare_leg(leg, &logits, &leg.greedy_ids, &tok);
                }));
                eprintln!(
                    "[noise] mode={mode} seed={seed:?} leg={} secs={:.1} verdict={} ours_top5={:?} ours_at_ref_top10=[{}] ref_top5={:?}",
                    leg.name,
                    t0.elapsed().as_secs_f64(),
                    if verdict.is_ok() { "PASS" } else { "FAIL" },
                    top(&logits, 5)
                        .iter()
                        .map(|(i, l)| (*i, (l * 1000.0).round() / 1000.0))
                        .collect::<Vec<_>>(),
                    ours_at.join(", "),
                    ref_top
                        .iter()
                        .map(|(i, l)| (*i, (l * 1000.0).round() / 1000.0))
                        .collect::<Vec<_>>(),
                );
            }
        }
    }
}

/// llama.cpp's own first forward under mathematically equivalent settings,
/// recorded by `tools/qwen4exp_reference_variants.py`.
const VARIANTS_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/qwen4exp_reference_variants.json"
);

#[derive(serde::Deserialize)]
struct Variants {
    format: u32,
    variants: Vec<Variant>,
}

#[derive(serde::Deserialize)]
struct Variant {
    tag: String,
    extra_args: Vec<String>,
    /// leg name -> best-first top-10 `(id, logprob)` of the first forward.
    first_forward_top: std::collections::BTreeMap<String, Vec<VariantTop>>,
}

#[derive(serde::Deserialize)]
struct VariantTop {
    id: u32,
    logprob: f64,
}

/// The measurement the qwen4exp tolerance has to be judged against: replay
/// each recorded llama.cpp realization, AS THE CANDIDATE, through the
/// unchanged first-forward verdict against the fixture. Settings that only
/// change kernels whose math is identical (`--no-repack` changes the Q4_K
/// expert dot's float summation order; `-fa off` the attention kernel) fail
/// the gate against llama.cpp's own recording, because activation
/// quantization makes the 48-layer forward chaotic in float rounding. The
/// settings that do not touch the arithmetic reproduce the fixture bit for
/// bit, which is what makes the replay trustworthy.
///
/// The text half of the verdict is fed the fixture's own greedy text (only
/// the first forward was recorded), so only the logprob half is exercised.
/// Candidate logits are the top-10 logprobs at their ids over a -1e4 floor;
/// log-softmax then returns them to ~1e-6.
#[test]
fn the_gate_is_tighter_than_llama_cpps_own_spread_on_this_model() {
    let fx = Fixture::load();
    let text = std::fs::read_to_string(VARIANTS_PATH).expect("variants fixture reads");
    let recorded: Variants = serde_json::from_str(&text).expect("variants fixture parses");
    assert_eq!(recorded.format, 1, "stale variants fixture");

    let vocab = 248_320;
    let mut verdicts = Vec::new();
    for v in &recorded.variants {
        for leg in &fx.legs {
            let top = &v.first_forward_top[&leg.name];
            let mut logits = vec![-1.0e4_f32; vocab];
            for e in top {
                logits[e.id as usize] = e.logprob as f32;
            }
            let bit_identical = top
                .iter()
                .zip(&leg.steps[0].top)
                .all(|(a, b)| a.id == b.id && a.logprob == b.logprob);
            let pass = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                gguf_compare::assert_matches_reference(
                    &format!("variant/{}/{}", v.tag, leg.name),
                    &logits,
                    leg.greedy_ids.len(),
                    &leg.content,
                    &leg.first_forward_top(),
                    &leg.content,
                    qwen4exp_fixture::LOGPROB_ABS_TOLERANCE,
                );
            }))
            .is_ok();
            eprintln!(
                "[variants] {:<20} {:?} {:<6} identical-to-fixture={bit_identical} verdict={}",
                v.tag,
                v.extra_args,
                leg.name,
                if pass { "PASS" } else { "FAIL" }
            );
            verdicts.push((v.tag.as_str(), leg.name.as_str(), bit_identical, pass));
        }
    }
    let expected = [
        ("fixture-args", "primes", true, true),
        ("fixture-args", "moon", true, true),
        ("no-ctx-checkpoints", "primes", true, true),
        ("no-ctx-checkpoints", "moon", true, true),
        ("no-repack", "primes", false, true),
        ("no-repack", "moon", false, false),
        ("fa-off", "primes", false, false),
        ("fa-off", "moon", false, false),
        ("no-repack-fa-off", "primes", false, true),
        ("no-repack-fa-off", "moon", false, false),
        ("kv-f32", "primes", true, true),
        ("kv-f32", "moon", true, true),
        ("threads-8", "primes", true, true),
        ("threads-8", "moon", true, true),
    ];
    assert_eq!(
        verdicts, expected,
        "the recorded variants no longer replay as measured"
    );
}
