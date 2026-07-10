//! Real-model benchmarks: TTFT and decode tok/s for Qwen2.5-1.5B on the
//! machine's default GPU. Budgets and the last recorded numbers live in
//! `bench/BASELINE.md`. Run with
//!
//! ```text
//! MUMMU_QWEN2_DIR=path/to/qwen2.5-1.5b cargo bench -p mummu-bench
//! ```
//!
//! Without `MUMMU_QWEN2_DIR` (or weights on disk) only the harness smoke runs,
//! so `cargo bench` stays green on machines without the multi-GB checkpoint.

use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use criterion::{Criterion, criterion_group, criterion_main};
use mummu::backend::Gpu;
use mummu::decode::argmax_id;
use mummu::models::qwen2::{self, LoadedQwen2};
use tokenizers::Tokenizer;

/// Decode steps timed per criterion sample: long enough to amortize per-step
/// jitter, short enough that the KV cache stays near its steady-state length.
const DECODE_STEPS_PER_SAMPLE: usize = 32;

fn qwen2_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("MUMMU_QWEN2_DIR")?);
    dir.is_dir().then_some(dir)
}

/// The benchmark prompt — fixed so numbers are comparable across runs.
fn prompt_ids(dir: &std::path::Path) -> Vec<u32> {
    let text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nExplain, in three sentences, why the sky is blue.<|im_end|>\n<|im_start|>assistant\n";
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json loads");
    let ids = tok.encode(text, false).expect("encodes").get_ids().to_vec();
    assert!(ids.len() >= 16, "benchmark prompt suspiciously short");
    ids
}

fn bench_harness_smoke(c: &mut Criterion) {
    c.bench_function("harness_smoke", |b| {
        b.iter(|| black_box(1u64).wrapping_add(black_box(1u64)));
    });
}

fn bench_qwen2_real(c: &mut Criterion) {
    let Some(dir) = qwen2_dir() else {
        eprintln!("[mummu-bench] MUMMU_QWEN2_DIR not set — skipping real-model benches");
        return;
    };
    let device = burn::tensor::Device::<Gpu>::default();
    let loaded: LoadedQwen2<Gpu> =
        qwen2::load_from_dir(&dir, &device).expect("weights load checked");
    let ids = prompt_ids(&dir);

    // TTFT: fresh cache, full prefill, first token argmax (the id readback is
    // the GPU sync point, so the measured span covers real work end-to-end).
    let mut group = c.benchmark_group("qwen2.5-1.5b/gpu");
    group.sample_size(10);
    group.bench_function("ttft_prefill_first_token", |b| {
        b.iter(|| {
            let mut cache = loaded.new_cache();
            let logits = loaded.forward(&ids, 0, &mut cache, &device);
            black_box(argmax_id(logits).expect("argmax"))
        });
    });

    // Decode: per sample, prefill once (untimed), then time N greedy decode
    // steps through the warm KV cache. Per-token latency = measured / N.
    group.bench_function("decode_32_tokens", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut cache = loaded.new_cache();
                let logits = loaded.forward(&ids, 0, &mut cache, &device);
                let mut next = argmax_id(logits).expect("argmax");
                let start = Instant::now();
                for (step, past) in (ids.len()..).take(DECODE_STEPS_PER_SAMPLE).enumerate() {
                    let logits = loaded.forward(&[next], past, &mut cache, &device);
                    next = argmax_id(logits).expect("argmax");
                    black_box((step, next));
                }
                total += start.elapsed();
            }
            total
        });
    });
    group.finish();
}

criterion_group!(benches, bench_harness_smoke, bench_qwen2_real);
criterion_main!(benches);
