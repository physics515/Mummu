# Benchmark baseline & budgets

Reference machine: Ryzen 9 7950X3D · 128 GB · **RTX 4070 Ti SUPER 16 GB** (wgpu/Vulkan, `Fusion<Wgpu>`,
f32). Bench: `MUMMU_QWEN2_DIR=<qwen2.5-1.5b> cargo bench -p mummu-bench` (criterion, `benches/runner.rs`;
fixed ~36-token ChatML prompt). A change that pushes a budget over its ceiling does not ship; update the
recorded numbers (and this file's date) only on a legitimate improvement.

## Qwen2.5-1.5B-Instruct · single GPU · f32

| Metric | Recorded (2026-07-10) | Budget |
| --- | --- | --- |
| TTFT (fresh cache: full prefill + first token) | 100.5 ms | ≤ 150 ms |
| Decode latency (32 greedy tokens, warm KV cache) | 2.414 s → **75.4 ms/token ≈ 13.3 tok/s** | ≥ 10 tok/s |
| Peak GPU memory during decode (whole card, ~4.0 GiB desktop ambient → ~7.9 GiB runner) | 11.9 GiB | ≤ 13 GiB whole-card |

Datapoint (not yet a budget): the same model on `GpuF16` peaks at **8.7 GiB whole-card (~4.7 GiB
runner)** — VRAM roughly halves as expected — but decodes NaN today (see the ROADMAP P6
mixed-precision-islands item), so no f16 perf row exists yet.

Notes
- Effective weight-streaming bandwidth at 75 ms/token over ~6.2 GB of f32 weights is ~83 GB/s vs the
  card's ~672 GB/s — the decode path is kernel/dispatch-bound, not bandwidth-bound. The CubeCL SPIR-V
  compiler feature and the f16 path (ROADMAP P6) are the identified levers.
- `harness_smoke` (sub-ns) exists only to keep `cargo bench` green without the multi-GB weights.
