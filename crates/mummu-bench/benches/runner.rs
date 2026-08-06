//! Real-model benchmarks: TTFT and decode tok/s for Qwen2.5-1.5B on the
//! machine's default GPU, in f32 and (where `SHADER_F16` exists) f16.
//! Budgets and the last recorded numbers live in `bench/BASELINE.md`. Run with
//!
//! ```text
//! MUMMU_QWEN2_DIR=path/to/qwen2.5-1.5b cargo bench -p mummu-bench
//! ```
//!
//! Without `MUMMU_QWEN2_DIR` (or weights on disk) only the harness smoke runs,
//! so `cargo bench` stays green on machines without the multi-GB checkpoint.

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use burn::tensor::backend::Backend;
use criterion::{Criterion, criterion_group, criterion_main};
use mummu::backend::{Gpu, GpuF16, inventory};
use mummu::decode::argmax_id;
use mummu::models::CausalLm;
use mummu::models::qwen2::{self, LoadedQwen2};
use tokenizers::Tokenizer;

/// Decode steps timed per criterion sample: long enough to amortize per-step
/// jitter, short enough that the KV cache stays near its steady-state length.
const DECODE_STEPS_PER_SAMPLE: usize = 32;

/// Prefill length for the long-context TTFT row. Chosen so the explicit
/// attention path's `[1, heads, t, t]` scores tensor is hundreds of MiB —
/// large enough that avoiding it is a measurable effect, small enough that
/// the 16 GB reference card still holds the f32 model beside it.
const LONG_PREFILL_TOKENS: usize = 2048;

fn qwen2_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("MUMMU_QWEN2_DIR")?);
    dir.is_dir().then_some(dir)
}

/// The benchmark prompt — fixed so numbers are comparable across runs.
fn prompt_ids(dir: &Path) -> Vec<u32> {
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

/// TTFT + decode for one backend; the model drops (and its VRAM frees) when
/// this returns, so backends bench sequentially without stacking checkpoints.
fn bench_qwen2_on<B: Backend>(c: &mut Criterion, dir: &Path, group_name: &str)
where
    B::Device: Default,
{
    let device = B::Device::default();
    let loaded: LoadedQwen2<B> = qwen2::load_from_dir(dir, &device).expect("weights load checked");
    let ids = prompt_ids(dir);

    // TTFT: fresh cache, full prefill, first token argmax (the id readback is
    // the GPU sync point, so the measured span covers real work end-to-end).
    let mut group = c.benchmark_group(group_name);
    group.sample_size(10);
    group.bench_function("ttft_prefill_first_token", |b| {
        b.iter(|| {
            let mut cache = loaded.new_cache();
            let logits = loaded.forward(&ids, 0, &mut cache, &device);
            black_box(argmax_id(logits).expect("argmax"))
        });
    });

    // Long-context prefill: the same work, but at a sequence length where the
    // attention formulation actually matters. Explicit attention materializes
    // a `[1, heads, t, t]` scores tensor — 201 MiB of f32 at t = 2048 for this
    // model's 12 heads, versus 62 KiB at the 36-token prompt above — so this
    // is the row that moves when the attention step changes shape.
    let long_ids: Vec<u32> = ids
        .iter()
        .cycle()
        .take(LONG_PREFILL_TOKENS)
        .copied()
        .collect();
    assert_eq!(long_ids.len(), LONG_PREFILL_TOKENS);
    group.bench_function("ttft_prefill_2048", |b| {
        b.iter(|| {
            let mut cache = loaded.new_cache();
            let logits = loaded.forward(&long_ids, 0, &mut cache, &device);
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

fn bench_qwen2_real(c: &mut Criterion) {
    let Some(dir) = qwen2_dir() else {
        eprintln!("[mummu-bench] MUMMU_QWEN2_DIR not set — skipping real-model benches");
        return;
    };
    bench_qwen2_on::<Gpu>(c, &dir, "qwen2.5-1.5b/gpu");
    if inventory().any_shader_f16() {
        bench_qwen2_on::<GpuF16>(c, &dir, "qwen2.5-1.5b/gpu-f16");
    } else {
        eprintln!("[mummu-bench] no SHADER_F16 adapter — skipping the f16 benches");
    }
}

criterion_group!(benches, bench_harness_smoke, bench_qwen2_real);
criterion_main!(benches);
