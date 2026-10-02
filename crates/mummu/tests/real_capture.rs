//! Captured decode on real weights: the same tokens, faster, a busier GPU.
//!
//! Qwen3-0.6B on the accelerator, greedy, three ways — the ordinary dynamic
//! step, the static step executed op by op, and the static step captured
//! once per bucket and replayed (`mummu::capture`). The gate is exact:
//! every mode must decode the same tokens (a captured graph that read a
//! stale input would repeat a token, one that skipped a write would drift).
//! The numbers it prints are the point of the change: ms per token and how
//! busy the card was while decoding.
//!
//! ```text
//! MUMMU_QWEN3_DIR=~/.cache/mummu-models/qwen3-0.6b \
//! MUMMU_QWEN35_GGUF=~/.cache/mummu-models/qwen3.5-2b/Qwen3.5-2B-BF16.gguf \
//!   cargo test --release -p mummu --test real_capture -- --ignored --nocapture --test-threads 1
//! ```
//!
//! `MUMMU_CAPTURE_F16=1` runs the card in f16, where decode is launch-bound
//! and capture moves the most. Run it from a directory whose `cubecl.toml`
//! sets `check_mode = "auto"` for numbers without the repo's kernel
//! validation (`burn.toml`).

#![warn(clippy::pedantic, clippy::nursery, clippy::all)]

use std::ops::ControlFlow;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use mummu::capture::{DecodeRequest, StaticDecode, StepMode, generate, generate_batch_greedy};
use mummu::decode::SamplerOptions;
use mummu::models::{qwen3, qwen35};
use mummu_num::f64_from_usize;

const TOKENS: usize = 160;

fn dir() -> PathBuf {
    let d = std::env::var_os("MUMMU_QWEN3_DIR")
        .map(PathBuf::from)
        .expect("set MUMMU_QWEN3_DIR to a Qwen3 safetensors dir");
    assert!(d.is_dir(), "{} is not a directory", d.display());
    d
}

/// The card, in f16 when `MUMMU_CAPTURE_F16` is set — the precision whose
/// decode is launch-bound, so the one capture should move most. A device's
/// precision locks on first use, so it is one or the other per process.
fn gpu() -> burn::tensor::Device {
    if half() {
        mummu::backend::gpu_device_f16().expect("an f16-capable card")
    } else {
        mummu::backend::gpu_device()
    }
}

/// Whether the card computes in f16 (see [`gpu`]).
fn half() -> bool {
    std::env::var_os("MUMMU_CAPTURE_F16").is_some()
}

/// Tokens two decodes must share before they may differ. In f32 the static
/// step is the dynamic step's arithmetic in another layout and must decode
/// exactly the same tokens. In f16 a different reduction order (a batch of
/// rows, a grouped score matmul) moves logits by a few ulps, which flips a
/// near-tie eventually — measured on the 2B at token 74, a ship's name. A
/// broken step diverges within the first few tokens, so a long shared
/// prefix still catches it.
fn must_share(len: usize) -> usize {
    if half() { len.min(48) } else { len }
}

fn shared_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Sample the card's effective busy share every 100 ms until stopped.
fn sample_gpu(stop: Arc<AtomicBool>) -> std::thread::JoinHandle<Vec<f64>> {
    std::thread::spawn(move || {
        let mut v = Vec::new();
        while !stop.load(Ordering::SeqCst) {
            if let Some(u) = mummu::vram::utilization() {
                v.push(u.effective());
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        v
    })
}

/// What the card and this process's allocator hold, for the log.
fn memory(device: &burn::tensor::Device, at: &str) {
    let card = mummu::vram::memory().map_or(0, |m| m.used >> 20);
    let pools = device.memory_pool_usage();
    eprintln!(
        "[real_capture] memory {at}: card {card} MiB used; allocator {} MiB reserved, {} MiB live",
        pools.as_ref().map_or(0, |u| u.bytes_reserved >> 20),
        pools.as_ref().map_or(0, |u| u.bytes_in_use >> 20)
    );
    for r in device.memory_pool_report().unwrap_or_default() {
        if r.pages > 0 {
            eprintln!(
                "[real_capture]   pool {} KiB x {} pages (peak {}, largest {} KiB)",
                r.page_size >> 10,
                r.pages,
                r.pages_peak,
                r.largest_alloc >> 10
            );
        }
    }
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / f64_from_usize(v.len().max(1))
}

struct Run {
    tokens: Vec<u32>,
    ms_per_token: f64,
    busy: f64,
}

fn run<M: StaticDecode + Sync>(model: &M, prompt: &[u32], mode: StepMode) -> Run {
    let device = gpu();
    let greedy = SamplerOptions::greedy();
    // Warm: kernels compiled, autotune settled, graphs not reused (each run
    // captures its own) — the timed run measures the steady state.
    let req = |max_tokens| DecodeRequest {
        prompt_ids: prompt,
        max_tokens,
        opts: &greedy,
        device: &device,
        mode,
    };
    let _ = pollster::block_on(generate(
        model,
        req(24),
        |_| ControlFlow::Continue(()),
        None,
    ));
    let stop = Arc::new(AtomicBool::new(false));
    let watcher = sample_gpu(Arc::clone(&stop));
    let mut stamps = Vec::with_capacity(TOKENS);
    let tokens = pollster::block_on(generate(
        model,
        req(TOKENS),
        |_| {
            stamps.push(Instant::now());
            ControlFlow::Continue(())
        },
        None,
    ))
    .expect("decodes");
    stop.store(true, Ordering::SeqCst);
    let busy = mean(&watcher.join().expect("GPU sampler"));
    // Per-token time from the second token on: the first carries the prefill.
    let span = stamps
        .last()
        .unwrap()
        .duration_since(stamps[1])
        .as_secs_f64();
    let ms_per_token = span * 1e3 / f64_from_usize(stamps.len() - 2);
    Run {
        tokens,
        ms_per_token,
        busy,
    }
}

/// Dynamic, static and captured decodes of `prompt`: the same tokens, and
/// how long each took per token.
fn compare_modes<M: StaticDecode + Sync>(
    model: &M,
    prompt: &[u32],
    decode: impl Fn(&[u32]) -> String,
) {
    let mut results = Vec::new();
    for mode in [StepMode::Dynamic, StepMode::Static, StepMode::Captured] {
        let r = run(model, prompt, mode);
        eprintln!(
            "[real_capture] {mode:?}: {} tokens, {:.2} ms/token ({:.1} tok/s), GPU {:.0}% effective",
            r.tokens.len(),
            r.ms_per_token,
            1e3 / r.ms_per_token,
            r.busy * 100.0
        );
        results.push((mode, r));
    }
    let reference = &results[0].1.tokens;
    assert!(reference.len() > 32, "the reference decode stopped early");
    for (mode, r) in &results[1..] {
        if let Some(at) = r.tokens.iter().zip(reference).position(|(a, b)| a != b) {
            eprintln!(
                "[real_capture] {mode:?} diverges at token {at}: {:?} | dynamic {:?}",
                decode(&r.tokens[at.saturating_sub(8)..(at + 8).min(r.tokens.len())]),
                decode(&reference[at.saturating_sub(8)..(at + 8).min(reference.len())])
            );
        }
        let shared = shared_prefix(&r.tokens, reference);
        assert!(
            shared >= must_share(reference.len()) && (half() || r.tokens.len() == reference.len()),
            "{mode:?} decoded different tokens from token {shared} on"
        );
    }
    eprintln!(
        "[real_capture] {:?}",
        decode(reference).chars().take(160).collect::<String>()
    );
}

/// Batching: several prompts decode together, one dispatch per step for all
/// of them. Each slot must decode exactly what it decodes alone; the
/// aggregate tokens/s is what the card gives when it is fed more than one
/// row per weight read.
fn batch_scaling<M: StaticDecode + Sync>(model: &M, encode: impl Fn(&str) -> Vec<u32>) {
    let device = gpu();
    let topics = [
        "a lighthouse keeper",
        "the history of bread",
        "how volcanoes form",
        "a cat who learns chess",
        "the moons of Jupiter",
        "a city under the sea",
        "the first computer",
        "a garden in winter",
    ];
    let prompts: Vec<Vec<u32>> = (0..16)
        .map(|i| encode(&format!("Tell me about {}.", topics[i % topics.len()])))
        .collect();
    let steps = 96;
    let solo: Vec<Vec<u32>> = prompts[..2]
        .iter()
        .map(|p| {
            generate_batch_greedy(model, &[p.as_slice()], steps, &device, StepMode::Captured)
                .expect("solo")[0]
                .clone()
        })
        .collect();
    memory(&device, "after the solo runs");
    for slots in [1usize, 2, 4, 8, 16] {
        let batch: Vec<&[u32]> = prompts[..slots].iter().map(Vec::as_slice).collect();
        // Warm this batch shape, then time it.
        let _ = generate_batch_greedy(model, &batch, 8, &device, StepMode::Captured);
        let stop = Arc::new(AtomicBool::new(false));
        let watcher = sample_gpu(Arc::clone(&stop));
        let t0 = Instant::now();
        let out = generate_batch_greedy(model, &batch, steps, &device, StepMode::Captured)
            .expect("batch");
        let secs = t0.elapsed().as_secs_f64();
        stop.store(true, Ordering::SeqCst);
        let busy = mean(&watcher.join().expect("GPU sampler"));
        memory(&device, &format!("{slots} slots"));
        let generated: usize = out.iter().map(Vec::len).sum();
        eprintln!(
            "[real_capture] {slots:>2} slots: {generated} tokens in {secs:.2}s = {:.1} tok/s aggregate, {:.1} ms/step, GPU {:.0}% effective",
            f64_from_usize(generated) / secs,
            secs * 1e3 / f64_from_usize(steps),
            busy * 100.0
        );
        for (s, want) in solo.iter().enumerate().take(slots.min(2)) {
            let shared = shared_prefix(&out[s], want);
            assert!(
                shared >= must_share(want.len()) && (half() || out[s].len() == want.len()),
                "{slots} slots: slot {s} decoded differently from alone from token {shared} on"
            );
        }
    }
}

fn qwen3() -> (qwen3::LoadedQwen3, tokenizers::Tokenizer) {
    let dir = dir();
    let model = qwen3::load_from_dir(&dir, &gpu()).expect("loads");
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    (model, tok)
}

fn qwen35() -> (qwen35::LoadedQwen35, tokenizers::Tokenizer) {
    let path = std::env::var_os("MUMMU_QWEN35_GGUF")
        .map(PathBuf::from)
        .expect("set MUMMU_QWEN35_GGUF to a qwen35 GGUF");
    let f = mummu::gguf::GgufFile::open(&path).expect("gguf opens");
    let tok = mummu::tokenizer::tokenizer_from_gguf(&f).expect("tokenizer from gguf");
    drop(f);
    (qwen35::load_from_gguf(&path, &gpu()).expect("loads"), tok)
}

fn chat(tok: &tokenizers::Tokenizer, user: &str) -> Vec<u32> {
    let rendered = mummu::chat::ChatMl::qwen3().render(&[mummu::chat::Turn::user(user)]);
    tok.encode(rendered.as_str(), false)
        .expect("encodes")
        .get_ids()
        .to_vec()
}

const STORY: &str = "Write a long story about a lighthouse keeper.";

#[test]
#[ignore = "needs the local Qwen3 safetensors dir (MUMMU_QWEN3_DIR) + GPU"]
fn captured_decode_matches_and_runs_faster() {
    let (model, tok) = qwen3();
    compare_modes(&model, &chat(&tok, STORY), |ids| {
        tok.decode(ids, true).unwrap_or_default()
    });
}

#[test]
#[ignore = "needs the local Qwen3 safetensors dir (MUMMU_QWEN3_DIR) + GPU"]
fn batched_decode_matches_each_slot_and_scales() {
    let (model, tok) = qwen3();
    batch_scaling(&model, |user| chat(&tok, user));
}

/// The production family: `DeltaNet` layers whose recurrent state the step
/// advances in place, gated attention over a partial rotation.
#[test]
#[ignore = "needs a local qwen35 GGUF (MUMMU_QWEN35_GGUF) + GPU"]
fn qwen35_captured_decode_matches_and_runs_faster() {
    let (model, tok) = qwen35();
    compare_modes(&model, &chat(&tok, STORY), |ids| {
        tok.decode(ids, true).unwrap_or_default()
    });
}

#[test]
#[ignore = "needs a local qwen35 GGUF (MUMMU_QWEN35_GGUF) + GPU"]
fn qwen35_batched_decode_matches_each_slot_and_scales() {
    let (model, tok) = qwen35();
    batch_scaling(&model, |user| chat(&tok, user));
}
