# Benchmark baseline & budgets

Reference machine: Ryzen 9 7950X3D · 128 GB · **RTX 4070 Ti SUPER 16 GB** (wgpu/Vulkan,
`Fusion<Wgpu>`, **SPIR-V compiler** — `fusion<cubecl<wgpu<spirv>>>` since 2026-07-12).
Bench: `MUMMU_QWEN2_DIR=<qwen2.5-1.5b> cargo bench -p mummu-bench` (criterion, `benches/runner.rs`;
fixed ~36-token ChatML prompt). A change that pushes a budget over its ceiling does not ship; update the
recorded numbers (and this file's date) only on a legitimate improvement.

## Qwen2.5-1.5B-Instruct · single GPU · f32

| Metric | Recorded (2026-07-12, SPIR-V) | Budget |
| --- | --- | --- |
| TTFT (fresh cache: full prefill + first token) | 96.7 ms | ≤ 150 ms |
| Decode latency (32 greedy tokens, warm KV cache) | 1.737 s → **54.3 ms/token ≈ 18.4 tok/s** | ≥ 10 tok/s |
| Peak GPU memory during the real-inference suite (whole card, ~3.5 GiB desktop ambient → ~8.0 GiB runner) | 11.5 GiB | ≤ 13 GiB whole-card |

## Qwen2.5-1.5B-Instruct · single GPU · **f16** (weights + KV; f32 attention-score island)

| Metric | Recorded (2026-07-12, SPIR-V) | Budget |
| --- | --- | --- |
| TTFT (fresh cache: full prefill + first token) | 97.2 ms | ≤ 150 ms |
| Decode latency (32 greedy tokens, warm KV cache) | 1.744 s → **54.5 ms/token ≈ 18.4 tok/s** | ≥ 10 tok/s |
| Peak GPU memory during decode (whole card, 3.1 GiB ambient → **~3.6 GiB runner**, 2026-07-11 measure) | 6.75 GiB | ≤ 8 GiB whole-card |

The SPIR-V compiler (burn `vulkan` feature) cut decode latency **23%** on both dtypes
(70.7 → 54.3 ms/token) at the cost of ~9 ms TTFT (88.4 → 96.7 ms, still ⅔ under its ceiling); parity
held byte-identically (max |Δlogit| 2.670e-5, Ollama greedy leg exact). Decode remains
dispatch-bound — f16 still buys VRAM (~7.9 → ~3.6 GiB runner), not speed; ~54 ms/token streams f32
weights at only ~114 GB/s vs the card's ~672 GB/s, so kernel/dispatch overhead is still the ceiling.

## Qwen2.5-0.5B-Instruct · CPU (burn-flex) · f32

| Metric | Recorded (2026-07-10) | Budget |
| --- | --- | --- |
| Decode (8 greedy tokens, warm KV cache; `mummu-bench/tests/budget_cpu.rs`) | **11.7 tok/s** | ≥ 6 tok/s |

## OLMoE-1B-7B-0125-Instruct · CPU (burn-flex) · f32 from Q4_K_M GGUF

The MoE tier (64 experts, top-8 routing; 1B active / 7B total). Budgeted in **seconds per token**, not
tok/s: the dense-mask expert forward computes every expert for every token, so decode touches all 7B
params rather than the 1B the routing implies — this row is the number the routed-compute work in
ROADMAP P2 has to beat. GPU is out of reach until keep-quantized VRAM (P9): ~28 GB resident in f32.

| Metric | Recorded (2026-08-03) | Budget |
| --- | --- | --- |
| Load (dequantize ~7B params to f32; `mummu-bench/tests/budget_moe.rs`) | **81.9 s** | ≤ 300 s |
| Decode (4 greedy tokens, warm KV cache) | **0.76 s/token** | ≤ 2.0 s/token |

End-to-end (`tests/real_olmoe.rs`, prefill included and amortized over a 6-token answer) reads
1.15 s/token — the same path, measured the way a caller experiences it rather than warm-cache steady
state. Use the 0.76 s/token row for regression comparisons; both are recorded so the gap is not
mistaken for drift.

Notes
- 2026-07-11: the f32 attention-score island (NaN fix for f16) coincided with an f32 *improvement*
  (TTFT 100.5 → 88.4 ms, decode 13.3 → 14.1 tok/s) — softmax now always runs in f32 with fusion
  re-tuning around it; both budget gates re-passed (`budget.rs` 96.8 ms / 10.2 tok/s, `budget_cpu.rs`
  8.66 tok/s measured concurrently with the GPU bench).
- Effective weight-streaming bandwidth at ~71 ms/token over ~6.2 GB of f32 weights is ~88 GB/s vs the
  card's ~672 GB/s — the decode path is kernel/dispatch-bound, not bandwidth-bound. The CubeCL SPIR-V
  compiler feature (ROADMAP P6) is the identified lever.
- 2026-08-03: a routed-expert **gather** for the single-token decode step (device-side `select` of the
  8 routed slices — numerically the dense path minus its exactly-zero terms) measured **1.58 s/token vs
  the dense path's 1.15** on the end-to-end harness, so it was rejected under this file's rule and
  reverted: the ~200 MB per-layer gather copy costs more than the dense matmul it removes. Routes left
  open are in the ROADMAP P2 item.
- `harness_smoke` (sub-ns) exists only to keep `cargo bench` green without the multi-GB weights.
