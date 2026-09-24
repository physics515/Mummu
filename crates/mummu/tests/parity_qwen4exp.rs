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
//!   cargo test -p mummu --release --test parity_qwen4exp -- --ignored --nocapture \
//!   --exact qwen4exp_cpu_matches_the_recorded_llama_cpp_reference
//!
//! Name the gate test: this binary also holds diagnostics (the noise sweep
//! flips process-global `nn::refarith` state), and a gate verdict must never
//! share a process with them. The gate itself also asserts that state is
//! clean before it measures anything.
//! ```
//!
//! Point the directory at an `NVMe` copy: the routed experts and PLE rows are
//! read on demand from the shards, and random reads from spinning disks are
//! ~1000x slower. The f32 trunk is ~20 GB of RAM. A missing directory SKIPS
//! with a message. `MUMMU_QWEN4EXP_TRACE=1` with the `trace_*` test prints
//! the per-layer intermediates to walk a failure to its first divergent op.

#![warn(clippy::pedantic, clippy::nursery, clippy::all)]

use mummu_testkit::gguf_compare;
use mummu_testkit::llama_ref;
use mummu_testkit::qwen4exp_fixture;

use std::path::PathBuf;
use std::time::Instant;

use llama_ref::logprobs_at;
use mummu::gguf::GgufFile;
use mummu::models::CausalLm;
use mummu::models::qwen4exp::{self, LoadedQwen4exp};
use mummu_num::{f64_from_usize, narrow};
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
    let ids: Vec<u32> = idx[..k]
        .iter()
        .map(|&i| u32::try_from(i).expect("vocab index fits u32"))
        .collect();
    ids.iter().copied().zip(logprobs_at(logits, &ids)).collect()
}

#[test]
#[ignore = "needs the Flash-Next split set on NVMe (MUMMU_QWEN4EXP_DIR) and ~25 GB RAM"]
fn qwen4exp_cpu_matches_the_recorded_llama_cpp_reference() {
    let Some(first) = first_shard() else {
        eprintln!("skipped: set MUMMU_QWEN4EXP_DIR to the shard directory");
        return;
    };
    // The verdict is only meaningful on the exact path: a diagnostic that
    // ran earlier in this process (the noise sweep) may have left the
    // reference-arithmetic emulation or an embedding perturbation switched on.
    assert!(
        !mummu::nn::refarith::enabled() && !mummu::nn::refarith::perturbation_active(),
        "the parity gate must run on the exact path: nn::refarith emulation or \
         perturbation is active in this process (unset MUMMU_REF_ARITH, run the gate by name)"
    );
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
            prefill / f64_from_usize(leg.prompt_ids.len()),
            greedy.len(),
            (generate - prefill).max(0.0) / f64_from_usize(steps),
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

/// Teacher-forced per-op check against a FULL-precision llama.cpp dump of
/// one leg's prefill (`tools/qwen4exp_dump_tensors.cpp`): every named op
/// gets llama.cpp's exact input, so each printed `rel` is that op's own
/// error. `MUMMU_QWEN4EXP_TEACHER=<dump>/<leg>` selects the leg by the
/// directory name; add `MUMMU_REF_ARITH=1` to use llama.cpp's activation
/// grids (then a structurally identical op agrees to ~1e-4 or better), and
/// `MUMMU_QWEN4EXP_TEACHER_FORCE=0` to compare without forcing.
///
/// Measured 2026-09-16 on the primes leg, emulated and forced (median /
/// max relative error over layers): HC mix 1.1e-7 / 1.7e-4, router 1.8e-7,
/// shared expert 3.4e-7, routed experts 3.5e-5 / 2.4e-4, `DeltaNet`
/// 3.9e-5 / 2.9e-4, attention 1.1e-3 / 4.4e-3 (f16 flash attention, not
/// exactly emulated), head 2.3e-7. This check found the `DeltaNet` L2 form
/// (1.5e-2 on layer 28 before `GdnL2::AddEps`) and the missing `Q8_1` `s`
/// rounding (5e-4 on Q5_1-down experts). On the exact path the same ops
/// sit at 0.7-2.2e-2, which is llama.cpp's activation-quantization noise.
/// Free-running with the emulation on, `l_last` still drifts from 2.8e-3
/// at layer 0 to 4e-2 at layer 44, because float ops that are not
/// bit-identical flip rounding decisions: no emulation short of bit
/// exactness reproduces the reference's realization.
#[test]
#[ignore = "diagnostic: needs MUMMU_QWEN4EXP_DIR, MUMMU_QWEN4EXP_TEACHER and ~25 GB RAM"]
fn teacher_forced_ops_against_a_llama_cpp_dump() {
    let Some(first) = first_shard() else {
        eprintln!("skipped: set MUMMU_QWEN4EXP_DIR to the shard directory");
        return;
    };
    let Some(dump) = std::env::var_os("MUMMU_QWEN4EXP_TEACHER").map(PathBuf::from) else {
        eprintln!("skipped: set MUMMU_QWEN4EXP_TEACHER to a dumped leg directory");
        return;
    };
    let name = dump
        .file_name()
        .and_then(|n| n.to_str())
        .expect("the dump directory is named after its leg")
        .to_string();
    let fx = Fixture::load();
    let leg = fx.leg(&name);
    let (model, _tok) = load(&first);
    let device = mummu::backend::cpu_device();
    let mut cache = model.new_cache();
    let t0 = Instant::now();
    let logits = readback(model.forward(&leg.prompt_ids, 0, &mut cache, &device));
    eprintln!(
        "[teacher/qwen4exp/{name}] prefill {:.2} s; ours top-5 {:?}; ref top-5 {:?}",
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
/// realizations. Measured 2026-09-16 with 11 emulated realizations per run
/// (seeds None and 1..=10), llama.cpp's 4 distinct realizations for
/// comparison:
/// - primes `<|im_end|>`: -14.91 (sd 0.53) after the `DeltaNet` L2 fix, then
///   -15.52 (sd 0.56) once the `Q8_1` `s` rounding was emulated. llama.cpp:
///   -15.06 (sd 0.33).
/// - moon: -12.35 (sd 0.28), then -12.32 (sd 0.20). llama.cpp: -12.22
///   (sd 0.30).
/// - The unchanged first-forward verdict passed both legs in 2 of 11, then
///   in 3 of 11.
///
/// Earlier runs on older code gave different realizations again: 2 of 11
/// before the L2 fix, and 5 of 11 before the chunked-GDN solve changed, a
/// 6e-6 move on the exact path. The emulated forward is as chaotic as the
/// reference. `NOISE_SEEDS` (default 8) and `NOISE_EPS` (default 1e-5) tune
/// the sampling.
/// Restores the process-global emulation state however the noise sweep
/// exits, so nothing that runs after it in the same process measures a
/// perturbed or emulated forward by accident.
struct Restore(bool);

impl Drop for Restore {
    fn drop(&mut self) {
        mummu::nn::refarith::set_enabled(self.0);
        mummu::nn::refarith::set_perturbation(None, 0.0);
    }
}

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
    let _restore = Restore(mummu::nn::refarith::enabled());
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

/// The port's own consistency on the REAL weights, which no recorded
/// reference covers: a ~300-token prompt fed one-shot must give the same
/// last-token logits as the same prompt fed in uneven prefill chunks (the
/// chunked GDN prefill crosses its 64-token chunk boundary, attention reads
/// cached KV, the PLE hash and conv carry across calls) finished by a
/// single-token cached decode step. Exact f32 arithmetic on both sides, so
/// only float rounding may differ.
#[test]
#[ignore = "needs MUMMU_QWEN4EXP_DIR and ~25 GB RAM"]
fn chunked_prefill_and_cached_decode_match_one_shot_on_the_real_model() {
    let Some(first) = first_shard() else {
        eprintln!("skipped: set MUMMU_QWEN4EXP_DIR to the shard directory");
        return;
    };
    let (model, tok) = load(&first);
    let device = mummu::backend::cpu_device();
    let paragraph = "The lighthouse keeper counted the ships that passed each night, \
        writing their names in a ledger that had outlived three keepers before him. \
        Some nights the fog hid everything but the horn, and he wrote only the time. ";
    let text: String = std::iter::repeat_n(paragraph, 6).collect();
    let (_, ids) = qwen4exp_fixture::render_prompt_ids(&tok, &text);
    let n = ids.len();
    assert!((250..2000).contains(&n), "prompt is {n} tokens");

    let t0 = Instant::now();
    let mut cache = model.new_cache();
    let one_shot = readback(model.forward(&ids, 0, &mut cache, &device));
    let one_shot_s = t0.elapsed().as_secs_f64();
    drop(cache);

    // Uneven chunks: 1, 70 (crosses 64), 129, the rest but one, then decode.
    let t1 = Instant::now();
    let mut cache = model.new_cache();
    let cuts = [0, 1, 71, 200, n - 1];
    for w in cuts.windows(2) {
        model.forward_advance(&ids[w[0]..w[1]], w[0], &mut cache, &device);
    }
    let stepped = readback(model.forward(&ids[n - 1..], n - 1, &mut cache, &device));
    let stepped_s = t1.elapsed().as_secs_f64();

    let max_logit = one_shot
        .iter()
        .zip(&stepped)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let top_one = top(&one_shot, 10);
    let top_step = top(&stepped, 10);
    let max_lp = top_one
        .iter()
        .zip(logprobs_at(
            &stepped,
            &top_one.iter().map(|e| e.0).collect::<Vec<_>>(),
        ))
        .map(|(a, b)| (a.1 - b).abs())
        .fold(0f64, f64::max);
    eprintln!(
        "[consistency/qwen4exp] {n} tokens: one-shot {one_shot_s:.1} s, chunked+decode {stepped_s:.1} s; \
         max |dlogit| {max_logit:e}, max |dlogprob| over one-shot top-10 {max_lp:e}\n  one-shot top-5 {:?}\n  stepped  top-5 {:?}",
        &top_one[..5],
        &top_step[..5]
    );
    assert_eq!(
        top_one.iter().map(|e| e.0).collect::<Vec<_>>(),
        top_step.iter().map(|e| e.0).collect::<Vec<_>>(),
        "top-10 ids differ between one-shot and chunked+decode"
    );
    assert!(
        max_lp < 1e-2,
        "chunked+decode logprobs drift {max_lp} from one-shot"
    );
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
/// change kernels whose math is identical (`--no-repack` changes the `Q4_K`
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
                logits[e.id as usize] = narrow(e.logprob);
            }
            // "Identical" means the recorded double is the same value bit
            // for bit, not merely within some tolerance.
            let bit_identical = top
                .iter()
                .zip(&leg.steps[0].top)
                .all(|(a, b)| a.id == b.id && a.logprob.to_bits() == b.logprob.to_bits());
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

/// Long-context structure against llama.cpp: the recorded ~560-token leg
/// (`qwen4exp_fixture::LONG_FIXTURE_PATH`) crosses nine GDN chunks and
/// attends over hundreds of cached positions, which the two short legs never
/// reach. The verdict is the part of the reference that is NOT arithmetic
/// noise: the first forward's top-1 and all [`qwen4exp_fixture::LONG_MAX_TOKENS`]
/// greedy ids, id for id. The tail logprobs are printed, not bounded (at
/// -20 nats they sit inside llama.cpp's own spread; see
/// `the_gate_is_tighter_than_llama_cpps_own_spread_on_this_model`).
#[test]
#[ignore = "needs MUMMU_QWEN4EXP_DIR and ~25 GB RAM"]
fn long_prompt_greedy_matches_the_recorded_llama_cpp_reference() {
    let Some(first) = first_shard() else {
        eprintln!("skipped: set MUMMU_QWEN4EXP_DIR to the shard directory");
        return;
    };
    let fx = Fixture::load_from(qwen4exp_fixture::LONG_FIXTURE_PATH);
    let leg = fx.leg("long");
    let (model, tok) = load(&first);
    let (rendered, ids) = qwen4exp_fixture::render_prompt_ids(&tok, &leg.prompt);
    assert_eq!(
        rendered, leg.rendered,
        "ChatMl::qwen3() renders differently"
    );
    assert_eq!(
        ids, leg.prompt_ids,
        "our tokenizer no longer produces the recorded ids"
    );
    let device = mummu::backend::cpu_device();

    let t0 = Instant::now();
    let mut cache = model.new_cache();
    let logits = readback(model.forward(&leg.prompt_ids, 0, &mut cache, &device));
    let prefill = t0.elapsed().as_secs_f64();
    drop(cache);
    let t1 = Instant::now();
    let greedy = pollster::block_on(model.greedy_generate(
        &leg.prompt_ids,
        qwen4exp_fixture::LONG_MAX_TOKENS,
        &device,
    ))
    .expect("greedy decode");
    let generate = t1.elapsed().as_secs_f64();
    let ours_top = top(&logits, 5);
    let ref_top = leg.first_forward_top();
    let (rss, hwm) = rss_mib();
    eprintln!(
        "[long/qwen4exp] {} prompt tokens: prefill {prefill:.1} s ({:.3} s/token); greedy {} tokens in \
         {generate:.1} s (~{:.2} s/decode token after its prefill); RSS {rss} MiB, peak {hwm} MiB\n\
         [long/qwen4exp] ours top-5 {ours_top:?}\n[long/qwen4exp] ref  top-5 {ref_top:?}\n\
         [long/qwen4exp] ours greedy {greedy:?} {:?}\n[long/qwen4exp] ref  greedy {:?} {:?}",
        leg.prompt_ids.len(),
        prefill / f64_from_usize(leg.prompt_ids.len()),
        greedy.len(),
        (generate - prefill).max(0.0) / f64_from_usize(greedy.len().saturating_sub(1).max(1)),
        tok.decode(&greedy, true).unwrap_or_default(),
        leg.greedy_ids,
        leg.content,
    );
    assert_eq!(
        ours_top[0].0, ref_top[0].0,
        "first-forward top-1 differs from llama.cpp on the long prompt"
    );
    assert_eq!(
        greedy, leg.greedy_ids,
        "greedy ids differ from llama.cpp on the long prompt"
    );
}
