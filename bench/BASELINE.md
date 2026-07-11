# Benchmark baseline & budgets

Reference machine: Ryzen 9 7950X3D · 128 GB · **RTX 4070 Ti SUPER 16 GB** (wgpu/Vulkan, `Fusion<Wgpu>`).
Bench: `MUMMU_QWEN2_DIR=<qwen2.5-1.5b> cargo bench -p mummu-bench` (criterion, `benches/runner.rs`;
fixed ~36-token ChatML prompt). A change that pushes a budget over its ceiling does not ship; update the
recorded numbers (and this file's date) only on a legitimate improvement.

## Qwen2.5-1.5B-Instruct · single GPU · f32

| Metric | Recorded (2026-07-11) | Budget |
| --- | --- | --- |
| TTFT (fresh cache: full prefill + first token) | 88.4 ms | ≤ 150 ms |
| Decode latency (32 greedy tokens, warm KV cache) | 2.263 s → **70.7 ms/token ≈ 14.1 tok/s** | ≥ 10 tok/s |
| Peak GPU memory during decode (whole card, ~4.0 GiB desktop ambient → ~7.9 GiB runner) | 11.9 GiB | ≤ 13 GiB whole-card |

## Qwen2.5-1.5B-Instruct · single GPU · **f16** (weights + KV; f32 attention-score island)

| Metric | Recorded (2026-07-11) | Budget |
| --- | --- | --- |
| TTFT (fresh cache: full prefill + first token) | 88.0 ms | ≤ 150 ms |
| Decode latency (32 greedy tokens, warm KV cache) | 2.270 s → **70.9 ms/token ≈ 14.1 tok/s** | ≥ 10 tok/s |
| Peak GPU memory during decode (whole card, 3.1 GiB ambient → **~3.6 GiB runner**) | 6.75 GiB | ≤ 8 GiB whole-card |

f16 decode speed matches f32 — the WGSL decode path is dispatch-bound, not bandwidth-bound (see Notes),
so halved weight traffic buys nothing yet; the win is VRAM (**~7.9 → ~3.6 GiB runner**, room for larger
models/contexts). The SPIR-V compiler feature (ROADMAP P6) is the identified speed lever for both dtypes.

## Qwen2.5-0.5B-Instruct · CPU (burn-flex) · f32

| Metric | Recorded (2026-07-10) | Budget |
| --- | --- | --- |
| Decode (8 greedy tokens, warm KV cache; `mummu-bench/tests/budget_cpu.rs`) | **11.7 tok/s** | ≥ 6 tok/s |

Notes
- 2026-07-11: the f32 attention-score island (NaN fix for f16) coincided with an f32 *improvement*
  (TTFT 100.5 → 88.4 ms, decode 13.3 → 14.1 tok/s) — softmax now always runs in f32 with fusion
  re-tuning around it; both budget gates re-passed (`budget.rs` 96.8 ms / 10.2 tok/s, `budget_cpu.rs`
  8.66 tok/s measured concurrently with the GPU bench).
- Effective weight-streaming bandwidth at ~71 ms/token over ~6.2 GB of f32 weights is ~88 GB/s vs the
  card's ~672 GB/s — the decode path is kernel/dispatch-bound, not bandwidth-bound. The CubeCL SPIR-V
  compiler feature (ROADMAP P6) is the identified lever.
- `harness_smoke` (sub-ns) exists only to keep `cargo bench` green without the multi-GB weights.
