# Benchmark baseline & budgets

Reference machine: Ryzen 9 7950X3D · 128 GB · **RTX 4070 Ti SUPER 16 GB** (wgpu/Vulkan,
`Fusion<Wgpu>`, **SPIR-V compiler** — `fusion<cubecl<wgpu<spirv>>>` since 2026-07-12).
Bench: `MUMMU_QWEN2_DIR=<qwen2.5-1.5b> cargo bench -p mummu-bench` (criterion, `benches/runner.rs`;
fixed ~36-token ChatML prompt). A change that pushes a budget over its ceiling does not ship; update the
recorded numbers (and this file's date) only on a legitimate improvement.

## Qwen2.5-1.5B-Instruct · single GPU · f32

| Metric | Recorded (2026-08-06) | Budget |
| --- | --- | --- |
| TTFT (fresh cache: full prefill + first token) | 90.9 ms | ≤ 150 ms |
| Prefill @ 2048 tokens (`ttft_prefill_2048`) | **593 ms** | ≤ 900 ms |
| Decode latency (32 greedy tokens, warm KV cache) | 1.921 s → **60.0 ms/token ≈ 16.7 tok/s** | ≥ 10 tok/s |
| Peak GPU memory during the real-inference suite (whole card, ~3.5 GiB desktop ambient → ~8.0 GiB runner) | 11.5 GiB | ≤ 13 GiB whole-card |

## Qwen2.5-1.5B-Instruct · single GPU · **f16** (weights + KV; f32 attention-score island)

| Metric | Recorded (2026-08-06) | Budget |
| --- | --- | --- |
| TTFT (fresh cache: full prefill + first token) | 20.4 ms | ≤ 150 ms |
| Prefill @ 2048 tokens (`ttft_prefill_2048`) | **210 ms** | ≤ 900 ms |
| Decode latency (32 greedy tokens, warm KV cache) | 0.656 s → **20.5 ms/token ≈ 48.8 tok/s** | ≥ 10 tok/s |
| Peak GPU memory during decode (whole card, 3.1 GiB ambient → **~3.6 GiB runner**, 2026-07-11 measure) | 6.75 GiB | ≤ 8 GiB whole-card |

**2026-08-06 re-measure — the f16 path is 2.7× faster than 2026-07-12 recorded, f32 10% slower.**
Both tables above were re-run on an idle card (criterion, three runs each, unchanged shipping code)
because the flash-attention evaluation below needed an honest control. f16 decode moved
54.5 → 20.5 ms/token and f16 TTFT 97.2 → 20.4 ms — an improvement nothing in this repo claimed, so
it came from underneath (driver, or a dependency in the CubeCL/wgpu stack, or the 2026-07-30 dtype
pinning removing a per-call policy lookup). f32 decode moved the other way, 54.3 → 60.0 ms/token;
the gate never tripped because its ceiling is 10 tok/s, four ms/token of slack away. Both drifts are
pre-existing at HEAD, not caused by anything this run shipped — bisecting the f32 one is a ROADMAP
item.

The SPIR-V compiler (burn `vulkan` feature) cut decode latency **23%** on both dtypes
(70.7 → 54.3 ms/token) at the cost of ~9 ms TTFT (88.4 → 96.7 ms, still ⅔ under its ceiling); parity
held byte-identically (max |Δlogit| 2.670e-5, Ollama greedy leg exact). **The "f16 buys VRAM, not
speed" reading from that run no longer holds**: as of the 2026-08-06 re-measure f16 decodes at
20.5 ms/token against f32's 60.0 — 2.9× — so f16 is now the fast path as well as the small one, and
the f32 path is the one that looks dispatch-bound (60 ms/token streams ~6.2 GB of f32 weights at
~103 GB/s against the card's ~672 GB/s).

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
- 2026-08-06: **burn 0.21's fused `attention` op (CubeCL flash attention) was measured and rejected.**
  Swapping the explicit q·kᵀ → scale → mask → f32-softmax → ·v chain for one
  `tensor::module::attention(…, is_causal: true)` dispatch is a drop-in — same default `1/sqrt(hd)`
  scale, same bottom-right causal alignment, and the kernel keeps the f32 island itself
  (`AccumulatorPrecision::Strict(F32)`) — and it passed a new equivalence unit test against the
  explicit formulation. It still does not ship, because the A/B (criterion, same session, idle card,
  two runs per arm) says it costs more than it buys on this hardware:

  | metric | explicit (shipping) | fused SDPA | delta |
  | --- | --- | --- | --- |
  | f32 TTFT (36 tok) | 90.9 ms | 91.9 ms | +1.1 % |
  | f32 prefill @ 2048 | 593 ms | 629 ms | **+6.0 %** |
  | f32 decode | 60.0 ms/token | 61.1 ms/token | +1.8 % |
  | f16 TTFT (36 tok) | 20.4 ms | 16.7 ms | **−18 %** |
  | f16 prefill @ 2048 | 210 ms | 164 ms | **−22 %** |
  | f16 decode | 20.5 ms/token | 22.9 ms/token | **+11 %** |

  The one real win is f16 prefill, where the accelerated flash kernel's plane matmuls have tiles to
  fill. Decode is `seq_q = 1` — a matvec with no tile reuse, where flash's machinery is pure overhead
  and (the leading hypothesis) an opaque op the Fusion backend cannot absorb into the surrounding
  stream the way it absorbs the explicit chain's scale/mask/softmax/cast. f32 loses everywhere for
  want of an accelerated path to exploit. Adopting only the winning quadrant would mean a
  dtype-conditional fork in the hottest leaf function, on a path no strict parity gate covers (the
  gates run f32 and GGUF-dequant-to-f32), so it waits for a deliberate decision with f16 parity
  coverage behind it. Re-measure after the burn 0.22 / wgpu 30 bump: wgpu 30 lifts `SHADER_F16` to
  WGSL, which changes which kernels are even candidates here.
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
