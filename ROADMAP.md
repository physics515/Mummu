# Mummu — Roadmap

> The single source of truth for Mummu — a from-scratch **Burn** model runner shared by
> [laurelane](https://github.com/physics515/laurelane) and [Nanna](https://github.com/physics515/Nanna).
> The only docs are this file + [README.md](README.md): shipped capability becomes a FEATURE in the
> README (perf claims link a benchmark artifact); everything not-done / discovered / next is a `[ ]`
> here; git history + PRs are the record. Edit surgically; never rewrite wholesale.

**Stack:** Rust 2024 · **Burn 0.21** (`wgpu` 29 + `burn-flex` CPU, `fusion` + `autotune`, multi-device) ·
`burn-store` · HF `tokenizers` · runs on **any hardware** — CPU, one GPU, or several (multi-GPU + CPU
offload). Reference dev machine: Ryzen 9 7950X3D · 128 GB · RTX 4070 Ti SUPER 16 GB.
*(2026-08-21) Pin watch: **burn 0.21.0 is still the latest STABLE release** (crates.io, checked
2026-08-21) — the workspace is already on the newest burn there is; a `cargo update` pass brought
transitive deps current (7 patch/minor bumps, 243 lib tests green). 0.22 exists only as pre-releases
(`0.22.0-pre.2`, 2026-08-10). The wgpu-30 unblock below therefore waits on **stable 0.22** — the
parity-pinned stack does not ride pre-releases; when 0.22.0 lands, budget a full parity re-run (rope,
quantization API churn, wgpu 30's f16-in-WGSL win noted below).*
*(2026-07-16) Pin watch: wgpu 30 and tokenizers 0.23.1 are out; both held (burn 0.21 resolves wgpu 29,
and the tokenizers 0.23 change relevant to us — `add_tokens` normalizing content at insertion — touches
exactly the added-token path `tokenizer_from_gguf` rides, so the bump waits for a parity re-run) —
https://github.com/huggingface/tokenizers/releases.* *(2026-07-16, later run) **tokenizers 0.23.1 is
now IN** — the parity re-run that pin was waiting on happened and it is clean. Migration was two API
shifts: `add_tokens`/`add_special_tokens` take `impl IntoIterator<Item = AddedToken>` (not a slice),
and they plus `with_normalizer` now return `Result` — all three are handled loudly rather than
`let _ =`'d, so a rejected added token is an error instead of a silently-missing id. The feared
normalization-at-insertion change is a non-event for us because `tokenizer_from_gguf` already
verifies every added token's id post-build. Evidence: both GGUF tokenizer byte-identity batteries
still pass, and every parity number is **bit-identical to 0.22** — LFM2.5-1.2B 1.4879674843625068e-2,
230M 3.244355439983515e-2, GGUF-vs-llama.cpp 2.6617605580289094e-1 (qwen2) / 2.5971455430462065e-1
(lfm2), Qwen2-vs-Candle 2.29e-5 with the Ollama greedy leg exact; both tool-call emissions
unchanged; budgets 105.5 ms / 12.2 tok/s GPU, 11.74 tok/s CPU. **wgpu 30 stays held** — it is not
ours to pick, burn 0.21 resolves wgpu 29 transitively, so it unblocks with a burn bump, not a
`cargo upgrade`.* *(2026-07-22) Pin watch re-checked: burn is still 0.21 (no 0.22), tokenizers still
0.23.1 — the pinned combo is current. Worth knowing for the eventual burn bump: wgpu 30's changelog
lifts `SHADER_F16` to **all** shader kinds (WGSL included — previously SPIR-V passthrough only) and
adds the Vulkan f16 IO polyfill (PR #7884, already confirmed live here) — so when burn moves to
wgpu 30, f16 stops being SPIR-V-only and the WGSL fallback path (non-Vulkan APIs) can run f16 too —
https://github.com/gfx-rs/wgpu/blob/trunk/CHANGELOG.md.* *(2026-07-30) Pin watch: **burn
v0.22.0-pre.1 tagged 2026-07-29** — 0.21 is still the latest stable and stays pinned, but the pre-release
telegraphs a MAJOR migration (see the new P0 item below): the `Tensor` backend generic is removed in
favor of a high-level `Device` struct, and backends **lose their associated element types** in favor of
device defaults — the exact `B::FloatElem` seam Mummu's dtype pinning and `GpuF16` alias ride. Also
upstream: a `FloatCastAdapter` in burn-store (our `CastFloatAdapter`'s role), burnpack split into
`burn-pack`, BitNet `Calibration::AbsMean` ternary quant, quant fallbacks for slice/gather/select/expand,
and a remote multi-device backend (iroh). tokenizers 0.23.1 remains current —
https://github.com/Tracel-AI/burn/releases* *(2026-08-03) Pin watch: burn 0.22 is **still pre-release**
(0.22.0-pre.1 remains the newest tag; 0.21.0 the latest stable) — the P0 migration item stays gated.
CubeCL tagged **0.11.0-pre.1** the same day (2026-07-29): a frontend mega-refactor (references), a
**Metal backend**, a new CPU runtime, tiled layouts, and CUDA stream priority hints — all of which
arrive with the burn bump, not before — https://github.com/tracel-ai/cubecl/releases*
*(2026-08-09) Pin watch: **burn 0.22 is still pre-release** (0.22.0-pre.1 remains the newest tag,
0.21.0 the latest stable) and cubecl still 0.11.0-pre.1, so the P0 migration stays gated; tokenizers
0.23.1 and every other direct dep are already current, and `cargo upgrade --incompatible` offers only
the standing wgpu 29→30 pin. Two cubecl-0.11 changelog entries matter to this run's autotune work and
are folded into the items below: **#1423 "Disable persistent tune cache option"** (a
`CUBECL_AUTOTUNE_CACHE` env var that bypasses persistent cache read/write and keeps tuning
in-memory-per-process) and **#1422 "Feat/autotune throughput"** (throughput-based autotuning, beside
#1408 "Peak device throughput") — https://github.com/tracel-ai/cubecl/releases ·
https://github.com/tracel-ai/cubecl/pull/1423*

## North Star

One binary that **imports any open model**, **quantizes it to fit**, and runs it — LLM, embeddings, and
(later) vision — **natively in Rust across all the hardware the user has**: CPU, one GPU, or several GPUs
together with CPU offload, used **to the fullest**. From-scratch and parity-verified, fast enough to run a
real agent loop offline. **Burn is the core** — the one inference *and* training engine; every model and
backend is built from scratch on Burn (via CubeCL), never a second runtime. Two apps (laurelane, Nanna)
build on it; Mummu is the runner, they keep the domain glue. Every run moves one phase toward that end
state — depth over breadth.

## Provenance

The core is **proven in laurelane** (from-scratch Qwen2 / LFM2.5 / all-MiniLM in Burn, GPU + CPU, KV
cache, byte-identical parity vs Candle, validated on a 4070 Ti SUPER). Mummu **extracts and generalizes**
that work into a standalone, app-agnostic crate, then advances it (f16, quant, LoRA, training, vision).
Items marked *(ex-laurelane)* have a working reference implementation to port + generalize; the rest is
forward work distilled from Nanna's P12.

## Performance & Benchmarking

Performance is a **gate**, not a phase. Governing metric: **task throughput @ budget** — decode tok/s and
TTFT while fitting the *target device set's* VRAM/RAM (one GPU, several, or CPU): the biggest useful model
that fits, run as fast as the hardware allows, every device busy. A change ships only when parity holds and
a benchmark holds/improves its budget; README perf claims link an artifact.
- [x] `mummu-bench` (criterion) — TTFT, decode tok/s, VRAM per model × device-set (1 GPU / N GPUs / CPU / hybrid).
      *(2026-07-09) Crate + harness wired (smoke bench); real model benches land with P5.* *(2026-07-10)
      Real Qwen2.5-1.5B GPU benches live: TTFT 100.5 ms, decode 13.3 tok/s (f32, criterion). More
      model × device-set combos accrete as those paths land (CPU, f16, multi-GPU).*
- [x] `bench/BASELINE.md` budgets (VRAM ceiling, min decode tok/s, max TTFT) + a `cargo test` gate that fails a regression.
      *(2026-07-10) Baseline recorded (TTFT ≤ 150 ms, ≥ 10 tok/s, ≤ 13 GiB whole-card; measured 100.5 ms /
      13.3 tok/s / 11.9 GiB peak incl. ~4 GiB ambient); gate = `mummu-bench/tests/budget.rs` (ignored,
      weights + GPU) passing at 110.4 ms / 11.8 tok/s.*
- [ ] Decode is dispatch-bound, not bandwidth-bound: 75 ms/token streams ~6.2 GB of f32 weights at only
      ~83 GB/s vs the 4070 Ti SUPER's ~672 GB/s — the SPIR-V compiler feature (P6 item) and f16 are the
      levers to chase; re-baseline after each. *(2026-07-11) Confirmed empirically: f16 (half the weight
      traffic) decodes at exactly f32's speed — 70.9 vs 70.7 ms/token — so bandwidth isn't the limiter;
      SPIR-V (TensorCores at f16) is the remaining lever.* *(2026-07-12) SPIR-V pulled: decode
      70.7 → 54.3 ms/token (+30% tok/s) on BOTH dtypes — and f16 still exactly matches f32, so the path
      stays dispatch-bound (~114 GB/s effective vs ~672). Next levers: whatever closes the remaining
      per-token dispatch gap (fewer kernels per step — deeper fusion of the decode step, or CubeCL
      graph/megakernel work) rather than bandwidth.* *(2026-07-16) Third, accidental corroboration: the
      budget gate run while a cargo compile was still busy on the CPU measured **9.2 tok/s**, and the
      same gate on an idle machine measured **13.0 tok/s** — a ~30% swing from *host* CPU load alone,
      on an otherwise-idle GPU (5% util). Decode throughput tracking CPU availability is what a
      dispatch-bound path looks like. Operationally: **run the budget gates on a quiet machine** or
      they report contention as a regression.*
      *(2026-08-06)* Two updates. **(1) The 2026-07-11 f16 leg of this argument is RETRACTED — it was
      never an f16 measurement.** "f16 (half the weight traffic) decodes at exactly f32's speed" came
      from a bench that builds `Gpu` then `GpuF16` in one process; the f32 leg locked the per-device
      default dtype policy, so the f16 model ran in f32 and of course matched to a tenth of a
      millisecond. Bisected this run (see the closed item above): real f16 decodes at **20.5 ms/token
      against f32's 60.0**. So the dispatch-bound reading holds for the **f32** path — which the
      independent evidence still supports (the ~30 % swing from *host CPU* load on an idle GPU; SPIR-V's
      +30 %) — and the open question becomes whether f32 is the right default at all rather than why f16
      is not faster. **(2) The next lever is named: graph capture.** The industry
      framing matches our numbers exactly — a kernel launch costs ~5–10 µs of CPU time, an LLM forward
      dispatches hundreds of them, and on batch-1 decode that CPU-side sequencing is 20–40 % of total
      inference time; the standard fix is to capture the decode step's whole launch sequence once and
      replay it, which keeps per-token overhead flat instead of paying it per kernel. burn **0.22 ships
      graph capture** for this purpose (P0 item), which is a far better bet than shaving kernels one at a
      time — this run's flash-attention A/B is the evidence that per-kernel substitution does not move
      this number. — https://gigagpu.com/cuda-graph-optimization-inference/
- [x] Evaluate Burn 0.21's `burn.toml` project config — per-subsystem tuning + a CubeCL kernel-validation
      layer without recompiling; useful as a debug switch for kernel-level parity hunts —
      https://burn.dev/blog/release-0.21.0/ *(2026-07-17 research)* Concretely, a `burn.toml` dropped at
      the project root parameterizes every internal subsystem with no code change / no recompile:
      **fusion's beam search**, **autotune aggressiveness**, **compilation-cache + validation modes**,
      **streaming concurrency**, and **memory-pool persistence**. The load-bearing one for us is the new
      **CubeCL kernel-validation layer** — it catches kernels that generate **out-of-bounds memory
      accesses** (the exact failure class behind a silent wrong-logits parity drift or a
      `STATUS_STACK_OVERFLOW`-adjacent GPU crash). Action when picked up: commit a checked-in
      `burn.toml` with validation ON for the parity/real-model test profiles (catch OOB in CI) and OFF
      for the benchmark profile (no validation overhead in the budget numbers), and re-confirm the
      budgets are unmoved by its presence.
      *(2026-07-21 research)* **Caveat found — `burn.toml` has no per-profile mechanism.** The documented
      schema is a single, global project-root file with flat sections (verified against the release notes):
      `[fusion.beam_search] max_blocks`; `[cubecl.autotune] level = "minimal"|"balanced"|"extensive"|"full"`,
      `cache = "local"|"target"|"global"|{file=…}`; `[cubecl.compilation] check_mode =
      "enforce"|"validate"|"auto"`, `cache`; `[cubecl.streaming] max_streams`; `[cubecl.memory]
      persistent_memory = "enabled"|"disabled"|"enforced"`. Validation is the `[cubecl.compilation]
      check_mode` knob (`"validate"` = the OOB-catching layer). So the item's "ON for test / OFF for bench
      **profiles**" premise isn't directly expressible in ONE file — the open question was whether the
      discovery rule allows a per-directory split.
      *(2026-07-24)* **Shipped — the per-directory split works.** The discovery rule read from
      cubecl-runtime 0.10 source answers the 07-21 question: `RuntimeConfig::from_current_dir` walks UP
      from the process CWD, `cubecl.toml` checked before `burn.toml` at each level, first hit wins — and
      cargo runs each crate's tests/benches with CWD = the package dir. So a repo-root **`burn.toml`**
      sets `[cubecl.compilation] check_mode = "validate"` (bounds-check every launch AND validate
      explicitly-unchecked kernels for OOB) arming the `crates/mummu` parity/real-model suites, while
      **`crates/mummu-bench/cubecl.toml`** opts the budget/bench crate back to the `auto` default so
      recorded numbers never carry validation overhead. Consumers run from their own CWD — untouched.
      Verified live: an A/B with a malformed root burn.toml makes a GPU test fail at config load naming
      the bad key (proof of discovery + parse, from BOTH crates' CWDs), the real-model GPU suite passes
      with validation armed (no OOB found — clean bill), and the budget gates hold their numbers from
      the opted-out bench crate (12.4 tok/s with the files ≈ 12.1 without; first run after a config
      change can read ~30% low while autotune re-tunes — re-run before believing a regression).
- [x] Evaluate **CubeCL's now-complete flash-attention kernel** for the decode/prefill attention step —
      the releases page reports a full implementation (causal **masking**, partitions, row-wise
      reductions, multi-plane ops). Mummu currently materializes attention explicitly (q·kᵀ → f32 softmax
      island → ·v); a fused flash-attention kernel collapses those into one dispatch, which is squarely
      the **fewer-kernels-per-step** lever the dispatch-bound decode note above is chasing (and it drops
      the O(t²) scores tensor at prefill). Gate strictly: the f32-softmax island is the whole reason the
      f16 parity holds, so any flash path must re-pass the parity harness (both legs) AND hold/beat
      `bench/BASELINE.md` before adoption. — https://github.com/tracel-ai/cubecl/releases *(2026-07-17 research)*
      *(2026-08-06) **Evaluated end to end, and rejected on measurement — the numbers are in
      `bench/BASELINE.md`.** It reaches Mummu as burn 0.21's `tensor::module::attention`, which the
      wgpu backend routes to `cubek`'s flash kernel through an autotune set (flash-blackbox-accelerated
      variants vs an unfused fallback). It is a genuine drop-in: `scale: None` asks for the op's own
      `1/sqrt(head_dim)` default — the same factor, and passing it explicitly would silently
      *disqualify* the flash kernel (burn-cubecl routes any custom scale to the fallback) — `is_causal:
      true` reproduces our mask exactly (its causal boundary aligns bottom-right, `col > row + (seq_k −
      seq_q)`, which IS the KV cache's rule at every `past`), and **the f32 island survives inside the
      kernel** (`AccumulatorPrecision::Strict(F32)`), so the whole reason f16 attention doesn't NaN is
      preserved rather than discarded. Implemented, proven equivalent by a new unit test against the
      explicit formulation written out longhand, then A/B'd (criterion, idle card, two runs per arm) —
      and reverted: f16 prefill @2048 −22 % and f16 TTFT −18 % (real wins, the accelerated plane
      matmuls have tiles to fill), but f16 decode **+11 %**, f32 decode +1.8 %, f32 prefill @2048
      **+6.0 %**. Decode is `seq_q = 1`, a matvec with no tile reuse where flash is pure overhead and
      (leading hypothesis) an opaque node the Fusion backend can't absorb the way it absorbs the
      explicit chain's scale/mask/softmax/cast. Adopting only the winning quadrant means a
      dtype-conditional fork in the hottest leaf function on a path **no strict parity gate covers**
      (the gates run f32 and GGUF-dequant-to-f32) — see the split item below. Kept from the work: a
      permanent **`ttft_prefill_2048` row** in the criterion bench and the budget gate (593 ms f32 /
      210 ms f16 recorded, ≤ 900 ms budget), the row where the attention formulation is visible at all —
      the ~36-token bench prompt makes a 62 KiB scores tensor, 2048 tokens makes 201 MiB.*
- [x] **Adopt flash attention for f16 prefill only, once f16 has parity coverage** — the winning
      quadrant of the 2026-08-06 evaluation above: `t > 1` (prefill) on an f16 ambient dtype is
      −22 % prefill @2048 and −18 % TTFT, and it drops the O(t²) scores tensor (201 MiB at 2048 × 12
      heads today, and it is the term that ends long-context prefill on a 16 GB card — a P6 fit lever,
      not only a latency one). Two prerequisites, both deliberate: (a) an **f16 parity leg** — every
      strict gate today runs f32 or GGUF-dequant-to-f32, so an f16-only numeric fork would ship
      unverified; the cheap shape is an in-process f16-vs-f32 first-forward agreement assert (the
      dtype-pinning work makes both aliases coexist — `real_mixed_dtype.rs` already does exactly this
      for one token) and the honest shape is llama.cpp at f16 on the `llama_ref` harness; (b) accept a
      dtype- **and** length-conditional branch in `GqaAttention::forward`, or find a formulation that
      isn't conditional. Re-measure first: the numbers are burn-0.21/wgpu-29-specific.
      *(2026-08-09)* **Prerequisite (a) is DONE — the honest shape, and it passed first run on both
      architectures.** New `tests/parity_f16.rs` (own binary: `GpuF16` locks the per-device dtype
      policy) runs the SAME strict comparison every other port passes, with our side loaded onto
      `GpuF16`: **Qwen2.5-1.5B Q4_K_M** — top-5 ids match llama.cpp **exactly in order**
      (785, 32, 16, 1249, 8420), 24-token greedy **byte-identical**, max |Δlogprob|
      **2.5284926197284596e-1**; **Qwen3-0.6B Q4_K_M** — top-5 exact in order
      (151667, 151644, 151645, 99966, 131545), greedy byte-identical incl. the `<think>` tokens, max
      |Δlogprob| **3.938617118639698e-1**. Both f16 numbers are *below* their f32 twins (2.66e-1 /
      4.02e-1), i.e. f16 adds nothing measurable on top of the reference's own Q8_K activation-quant
      noise — the f32-softmax island is doing its job. Enabling this cost only a refactor: the
      comparator moved out of `parity_gguf.rs` into a shared `tests/gguf_compare/` module (beside
      `llama_ref`, which `parity_lfm2.rs` still uses for transport only) and gained explicit `port` +
      `tolerance` parameters; the f32 legs re-passed unchanged (qwen3 bit-identical at
      4.015608155114805e-1).
      *(2026-08-09, same run) **Item CLOSED — rejected on correctness, and the parity leg built two
      hours earlier is what caught it.** With (a) in hand, (b) was implemented exactly as scoped: a
      `use_fused_attention(t, ambient, masked)` gate (`t > 1 && f16 && masked` — the measured
      quadrant, nothing wider) picking between a new `attend_fused` and the existing chain, extracted
      as `attend_explicit`; plus a CPU-backend unit test holding the two formulations to each other at
      four `(past, t)` pairs (they agree to 1e-5, so the bottom-right causal alignment and the implicit
      `1/sqrt(head_dim)` scale are right) and a gate test pinning the quadrant. **The A/B reproduced
      the win** — two runs per arm, same session, idle-ish card, f32 rows as an untouched control:
      f16 TTFT 24.9/24.6 → **21.0/20.8 ms (−15.6 %)**, f16 prefill@2048 240.6/239.5 → **224.2/221.3 ms
      (−7.2 %)**, f16 decode unchanged within noise (830.8 → 806.7 ms, and it cannot take the path by
      construction), f32 TTFT 98.2 → 98.0 ms / prefill 597.3 → 598.5 ms. **Then the f16 parity gate
      failed: Qwen2.5-1.5B Q4_K_M on `GpuF16` returns NON-FINITE logits through the fused kernel**
      (`logprobs_at`'s finiteness assert), while the same weights through the explicit chain are fine
      and llama.cpp-identical. So the 2026-08-06 reading that "the f32 island survives inside the
      kernel (`AccumulatorPrecision::Strict(F32)`)" does **not** hold in practice for the very model
      whose q·kᵀ overflow motivated the island — the fused path reproduces the pre-island 2026-07-11
      NaN. It is also model-dependent: **Qwen3-0.6B passed** through the same fused path (top-3 exact,
      greedy byte-identical, max |Δlogprob| 3.9386004209988457e-1 vs the explicit 3.938617118639698e-1
      — only the 5th tail id reshuffled), which is exactly what makes it unshippable as a rule in a
      shared leaf function: it would be correct for narrow models and silently NaN for wide ones.
      Reverted; the tree keeps the explicit chain. What would reopen this: burn 0.22 / wgpu 30 (the P0
      item — wgpu 30 lifts `SHADER_F16` to WGSL, changing which kernels are candidates at all), or an
      upstream fix that makes the kernel's score accumulation genuinely f32 for f16 inputs. Re-run
      `tests/parity_f16.rs` FIRST next time; the measurement was never the hard part.*
- [x] **Bisect the f32 decode drift: 54.3 → 60.0 ms/token since 2026-07-12** — the 2026-08-06
      re-measure found the f32 decode row 10 % slower than recorded while f16 read **2.7× faster**.
      *(2026-08-06, closed the same run — and it turned up something bigger than the drift.)* The
      2026-07-12 tree (`e44debf`) and the pre-dtype-pinning tree (`c1826e7`) were checked out and
      benched on the same idle machine: f32 reads **60.1 / 60.0 / 60.0 ms/token** across all three
      trees, so **f32 never regressed** — the 54.3 recorded on 2026-07-12 is not reproducible from
      that same code today, i.e. it was machine state (driver/OS/background), never a Mummu change.
      The f16 finding is the real one: `c1826e7` **panics `DTypeMismatch`** on the f16 bench leg, and
      `e44debf` reports f16 at 60.2 ms/token — a tenth of a millisecond from its own f32 row. That is
      the one-alias-per-process hazard (root-caused 2026-07-23, fixed 2026-07-30): this bench builds
      `Gpu` then `GpuF16` in ONE process, the f32 leg locked the per-device default dtype policy, and
      **the "f16" rows recorded on 2026-07-11 and 2026-07-12 were f32 runs wearing an f16 label** —
      which is exactly why they matched f32 so implausibly closely (70.9 vs 70.7, then 54.5 vs 54.3).
      Real f16 decode is **20.5 ms/token, 2.9× faster than f32**, and always was; the harness could
      not see it. Consequences folded above: the dispatch-bound item's f16 leg is retracted, and
      `bench/BASELINE.md` carries the three-tree table. The f16 *VRAM* figures are unaffected — they
      come from `real_f16.rs`, one alias per process, always genuinely f16. Standing lesson for the
      perf suite: **a measurement that agrees with its control to a tenth of a percent is evidence of
      a wiring bug, not of a null result.**
      *(2026-08-06, same run)* Guard added so this class of bug cannot recur silently:
      **`mummu-bench/tests/budget_f16.rs`**, an f16 budget gate in its OWN test binary (one dtype alias
      per process) that asserts `logits.dtype() == F16` before it believes a single number. Recorded
      21 ms TTFT / 16.9 tok/s, budgets 60 ms / 12 tok/s.
- [x] **Close the f16 autotune warm-up gap: 16.9 tok/s cold vs 48.8 steady** — building the f16 gate
      surfaced it. The gate prefills twice, then times 32 decode steps once, and gets 16.9 tok/s
      (~59 ms/token); criterion, which runs the same 32 steps across many samples, gets 48.8 (20.5
      ms/token). So an f16 session's **first ~32 tokens run at roughly f32 speed** and only then does the
      2.9× appear — the f16 path has many more autotune variants to try than f32 (whose same-harness gap
      is only 13.2 vs 16.7 tok/s). That is a real user-facing cost for a runner whose consumers open
      short-lived agent turns, and it is not a perf mystery so much as a caching question: CubeCL's
      autotune cache is configurable (`[cubecl.autotune] cache = "local"|"target"|"global"|{file=…}`,
      already in the repo-root `burn.toml`'s vocabulary) and a persisted, per-machine cache should let a
      cold process start warm. Route: measure whether a `global`/file-backed autotune cache carries the
      tuning across processes, and if so ship it as the default for consumers; if not, consider a
      warm-up prefill at model install. Gate on the budget rows like everything else.
      *(2026-08-09) **Measured, and the caching premise was wrong — shipped the warm-up instead.** The
      route's first branch is a non-question: CubeCL **already** persists autotune across processes by
      default (`[cubecl.autotune] cache` defaults to `target`; this repo's live cache is
      `crates/mummu-bench/target/autotune/`, written and re-loaded on every run), so a `global`/file
      location changes *where* the cache lives, never *whether* tuning carries. What the cold process
      still pays is **kernel compilation + pipeline creation**, which the wgpu runtime caches nowhere —
      cubecl 0.10 wires `CompilationCache` for the CUDA and HIP runtimes only, so the
      `[cubecl.compilation] cache` knob is inert on our path. No configuration can carry it; only
      spending it earlier can. New harness `mummu-bench/tests/warmup_f16.rs` measures the whole curve in
      one process (8 bursts × 32 decode steps, each burst a fresh cache + untimed prefill so KV length
      stays constant): **the first 32 tokens run at 12.5–16.3 tok/s, burst 2 onward is flat at
      37–41**, i.e. the cold tax is 2.5–3.0× and exactly ONE burst deep — nothing beyond it is
      recoverable in-process. So `budget_f16.rs`'s 16.9 tok/s is not a mystery, it IS the first burst.
      Shipped: **`CausalLm::warm_up(probe_ids, steps, device)`** — one prefill plus `steps` greedy decode
      steps on a throwaway cache, every step's argmax read back (an unsynchronized warm-up returns before
      the GPU runs anything), bounded by `MAX_WARM_UP_STEPS = 256`, default-implemented so every zoo model
      inherits it beside `sanity_check`. REAL-GPU proof (`mummu-bench/tests/warmup_api_f16.rs`, own binary
      because warm-up is a once-per-process effect): after a 4.21 s `warm_up(&ids, 32)`, a cold process's
      FIRST 32-token burst runs at **41.9 tok/s** vs the next burst's 41.0 — ratio 1.02× where un-warmed
      it is 0.33×. Warms the decode step (shape-stable at `t == 1`); prefill kernels key on prompt length,
      documented on the fn. The other half of the gap's premise — "48.8 steady" — did not survive:
      criterion measures 36.8 tok/s today and the same numbers come back on the pre-`cargo update`
      lockfile, so it is machine state, not a regression (see `bench/BASELINE.md`, incl. the measured
      f16-vs-f32 host-CPU sensitivity, +9.8 % vs +3.2 % for the same added load).*
      *(2026-08-09 research, to re-check when burn 0.22 lands)* The "no configuration can carry it"
      finding is a **cubecl-0.10-on-wgpu** statement, not a permanent one, and there are two routes to
      removing the residual cost rather than paying it earlier. (1) wgpu itself ships
      `Device::create_pipeline_cache` / `PipelineCacheDescriptor` (gfx-rs/wgpu #5293) — a driver-blob
      pipeline cache explicitly for "reducing program startup time" — so a cubecl-wgpu compilation
      cache is buildable upstream and is the thing to look for on the 0.22 / wgpu-30 bump. (2) CubeCL's
      documentation already describes **shipping a warm compilation + autotune cache with the binary**
      for a known deployment target, which for a consumer like Nanna (fixed Tauri build, known GPU
      classes) converts the cold start into a build-time cost. Verify the wiring before believing
      either: at cubecl 0.10 `CompilationCache` is constructed in the **cuda and hip** runtimes only
      (read from source this run), which is exactly why `[cubecl.compilation] cache` is inert on our
      path. — https://github.com/gfx-rs/wgpu/issues/5293 ·
      https://docs.rs/wgpu/latest/wgpu/struct.PipelineCache.html
- [ ] **A stale autotune cache is permanent, silent, and cost 21–27 % of f16 decode** — found while
      closing the item above, and it is the reason "just persist the autotune cache" is not a free win
      for consumers. This run's FIRST budget run happened on a contended machine (9.1 tok/s f32, failing
      its own gate before recovering to 13.1); autotune tuned under that contention, wrote its picks to
      `crates/mummu-bench/target/autotune/`, and **every later process loaded them and never re-tuned**.
      Deleting the directory and re-running the same code on the same machine: f16
      `decode_32_tokens` 1.0279 s → 0.8109 / 0.8371 s, while f32 was unmoved (1.9646 → 1.9797 s). CubeCL
      offers no invalidation, no re-tune trigger, and no confidence signal — the cache is keyed by
      (device, kernel, checksum), and a pick made under load is indistinguishable from a good one. That
      is a real hazard for a runner whose consumers ship a persisted cache to end users' machines, where
      the tune may happen during install (busy) and be believed forever. Routes: (a) re-tune on demand —
      an API to clear the cache root (Mummu knows where it is; `CacheConfig::root()` is public) plus a
      documented "re-tune" action in the consumer's settings; (b) pin the cache to a Mummu-owned location
      via `RuntimeConfig::set` (cubecl exposes a one-shot programmatic setter that must run before the
      first `get()`) so the runner controls invalidation rather than inheriting the CWD walk-up; (c)
      validate on load — time one warm-up burst against a recorded expectation and clear the cache when
      it reads far low, which the `warm_up` API already has the shape for. Gate any of them on
      `bench/BASELINE.md` like everything else. *(2026-08-09, measured this run.)*
      *(2026-08-09, same run) **Route (a) shipped: `mummu::tune`.*** `autotune_cache_dir()` reports
      where CubeCL will persist picks — read out of the very config CubeCL discovers
      (`CubeClRuntimeConfig::get().autotune.cache.root()` joined with the `autotune` segment
      `CacheOption::name` adds), so the path is right by construction rather than by a hardcoded copy
      of the discovery rule; `autotune_cache_report()` measures it (files + bytes, absent = empty, not
      an error); `clear_autotune_cache()` removes it and returns what it removed, which is the
      "re-tune GPU kernels" action a consumer's settings UI needs. Bounded and fail-loud throughout —
      a tree deeper than 8 levels or wider than 65 536 files is `TuneError::Implausible` rather than a
      long walk or a wide delete, and every path this module touches ends in the `autotune` segment by
      construction (asserted), so a misconfigured root cannot widen the delete. Documented honestly:
      it takes effect on the **next** process (a running one has already loaded the cache into memory).
      One new direct dep, `cubecl-runtime 0.10`, the same version burn 0.21 already resolves through
      burn-cubecl — feature-unified, zero compile-time cost, the `burn-store`/`wgpu` precedent.
      5 unit tests + a REAL-GPU proof (`tests/real_autotune_cache.rs`): it clears the cache, runs four
      512² matmuls with readbacks, and finds **3 files / 8 303 bytes** written to exactly the reported
      directory, then clears them again and confirms empty. That test deliberately touches
      `crates/mummu/target/autotune` and never the bench crate's — CubeCL's root is the walk-up from
      the process CWD, so the recorded benchmark numbers keep their own tuning. Still open here:
      (b) pinning the cache to a Mummu-owned location via `RuntimeConfig::set`, and (c) detecting a
      bad tune automatically rather than exposing the repair.
      *(2026-08-09 research)* **Upstream is moving on both halves, and it arrives with the burn 0.22
      bump — re-scope (b)/(c) then, don't hand-roll them now.** cubecl 0.11 adds
      **`CUBECL_AUTOTUNE_CACHE`** (PR #1423, "Disable persistent tune cache option"): an env var that
      bypasses persistent read *and* write and keeps tuning `in_memory_cache` per process. That is a
      cleaner (c) than anything we would build — a consumer that suspects a bad tune can run once with
      persistence off and compare, and a *test* harness can opt out entirely so a benchmark never
      inherits another run's picks (worth adopting for `mummu-bench` the moment it lands: this run's
      21–27 % f16 swing came from exactly that inheritance). Also in 0.11: **#1422 "Feat/autotune
      throughput"** + **#1408 "Peak device throughput"**, i.e. autotune scoring moves from latency to
      throughput against a measured device peak — plausibly *more* noise-robust, so re-measure the
      hazard's magnitude after the bump rather than assuming it persists. And CubeCL's own
      documentation frames the persisted caches as a **shipping** artifact — "ship a warm cache with
      your binary when you know the deployment target, so the cold-start cost is paid once at build
      time" — which is the mirror image of this item and worth evaluating for consumers with a fixed
      target (Nanna's Tauri build): a *good* tune shipped deliberately, rather than whatever the user's
      first busy minute produced. — https://github.com/tracel-ai/cubecl/pull/1423 ·
      https://github.com/tracel-ai/cubecl

- [ ] **The 27B decode gap is still unexplained — but it is NOT a broken quantized matmul.**
      *(2026-08-23)* Measured against native Windows ollama on the same box and checkpoint: ollama
      **16.0 tok/s**, mummu **0.21 tok/s** (~76x). Five hypotheses tried, all **falsified**: host round
      trips (fixed, no effect), trunk-on-GPU (24.7 s/tok, worse), layer-granular `n_gpu_layers`-style
      placement (40.7 s/tok, worse), autotune-per-shape (`CUBECL_AUTOTUNE_LEVEL=minimal`, 31.8 s/tok),
      and a suspected upstream defect in wgpu quantized matmul (**retracted, see below**).
      **Retraction.** An earlier version of this entry claimed burn 0.22's wgpu quantized matmul was
      nondeterministically wrong at k=n=8192. That was an artifact of the probe, not the runtime: the
      probe **quantized GB-scale synthetic tensors on the device**, which production never does, and
      ran the card out of memory — and OOM panics were read as kernel unreliability. Re-probed along
      the real path (`examples/pack-precision-probe.rs`: packed bytes from `q4.bin`/`q8.bin` -> device
      -> matmul) at this model's true dimensions (hidden 5120, FFN 17408), **both Q4 and Q8 are correct
      and bit-identical across repeated rounds, native and dequantized alike, at 2.5-3 ms warm**:
      Q4 relative error 0.062, Q8 0.0057 against the pack's own f32 bytes. The lesson is the same one
      that cost this run four wrong turns — *a probe only proves what it measures*, and a synthetic
      probe that does not reproduce the production path proves nothing about it.
      What IS real from the server logs: under `fusion`, a fused autotune key
      (`FusedMatmulAutotuneKey { m: 16, n: 32768, k: 8192, elem_rhs: U32, num_ops: 16 }`) has **all 17
      candidate kernels fail to launch** — Cmma/Mma variants with "No tile size is available for the
      problem" (the packed weight is `U32`, so no tensor-core tile applies), unit fallbacks with
      "Shared memory budget exceeded: kernel requests 294912 bytes, hardware allows 49152". cubecl then
      cannot pick a winner and panics. Those n=32768/k=8192 shapes are **mummu's own fused expert
      slabs**, not the model's tensors. Related upstream: tracel-ai/burn#4949 (same "No tile size"
      family, unfixed). Avoided today by building without `fusion` (below).
      **Consequence:** the planner's `if policy == Q4 && backend == Wgpu { continue }` rule is a burn
      0.21-era exclusion with no current evidence behind it, and it is what forces this model to Q8
      (~13 GiB, which does not fit a 16 GiB card with activation headroom) when Q4 would be ~7.5 GiB
      and fit whole. Lift it; then place precisions per tensor rather than per model (next item).
- [ ] **`fusion` is a net negative on this path.** *(2026-08-23)* Built `mummu-serve` with
      `--no-default-features --features vulkan-spirv`: **4.32 s/tok vs 4.80** with fusion, and fusion
      additionally *killed whole requests* with `burn-fusion .../stream/execution/ordering.rs: Ordering
      is bigger than operations: ordering len 10, operations len 0`. The host install is now the
      no-fusion build. Fusion remains a default cargo feature for the library; revisit after the
      quantized-matmul defect above is resolved, since fusion is what turns 16 separate ops into the
      unlaunchable fused key.
      *(2026-08-25) Sharper, and it is now a **correctness** statement, not a perf one: with `fusion`
      on — mummu's DEFAULT feature set — the Qwen2.5-1.5B parity gate cannot run at all. The first
      forward dies on the same defect, `burn-fusion 0.22.0-pre.2
      stream/execution/ordering.rs:49 — Ordering is bigger than operations: ordering len 3,
      operations len 0, num_executed 0, optimization len 3`, raised on the `DSU-4-0` device runner and
      surfacing on the caller as `client.rs:200` unwrapping a `CallError`. The SAME gate on the SAME
      weights with `--no-default-features --features vulkan-spirv` passes: top-5 ids exact in order
      (785, 16, 32, 1249, 8420) against the committed Candle fixture, max |Δlogit| **1.0681152e-4**
      (bound 1e-3). So `fusion` in the default set is currently shipping a configuration whose parity
      is **unverifiable**, and every recorded parity number in this file was measured on a stack that
      no longer runs it. Two follow-ups, both `[ ]` below: decide whether `fusion` stays a default
      before 0.22 stabilizes, and run the gates on BOTH feature sets so this cannot hide again.
- [x] **The persistent autotune cache never worked on the host — a TOML escaping bug.** *(2026-08-23)*
      `cubecl.toml` wrote the cache root as a TOML **basic** string containing a Windows path
      (`"D:\Docker Containers\..."`), where backslashes are escape sequences. cubecl's response to a
      parse error is to *ignore the file silently*: `Ignoring "...\CubeCL.toml", which doesn't have the
      right format => TOML parse error at line 10, column 26`. The cache directory had **zero** entries.
      Fixed by using a TOML *literal* string (single quotes, no escapes). Worth generalizing: any
      Windows path in a config file needs `'...'`, and a config loader that ignores malformed input
      without failing loudly will hide this class of bug indefinitely.
- [x] **Mixed-precision residency: per-tensor Q8/Q4/Q2 on one device, sized to live VRAM.**
      *(2026-08-23)* The fit planner used to pick ONE precision for a whole model, which is the coarsest
      possible answer: this 27B needs ~13 GiB at Q8 (does not fit a 16 GiB card with headroom) and ~7 GiB
      at Q4 (leaves ~6 GiB of budget unspent). New [`mummu::mix`] assigns a precision **per tensor**,
      greedily demoting whichever tensor has the lowest `sensitivity x error-increase` per byte freed, so
      attention projections, the LM head and the edge layers hold Q8 while the bulk FFN slabs sit at Q4.
      The error model is measured, not assumed (`examples/ladder-probe.rs`, on a real 89 M-param pack
      tensor against the pack's own f32 bytes): **Q8 0.0058, Q4 0.0997, Q2 0.6226** relative. That
      steepness is why the planner spends budget keeping tensors off Q2 rather than spreading pain
      evenly. Block scales are counted — at Q2 an f32 scale per 32 values is **37%** of the tensor, so
      ignoring them overstates headroom by a third. 7 unit tests, including monotonicity (tightening the
      budget must never *raise* a tensor's precision, or the rebalancer oscillates).
      **`QuantPolicy` gained `Q2`** plus `demote`/`promote`/`LADDER`. Q2 is the bottom rung burn can hold
      packed in VRAM (cubecl has `Q2S`; there is no 1-bit type, so **Q1 would need a custom CubeCL
      kernel** and is not on this stack). No pack stores Q2 either — it is reachable only by requantizing
      a tensor that is already resident, which is exactly the pressure response.
      **Budget now tracks live VRAM.** New [`mummu::vram`] loads NVML at runtime (a missing DLL degrades
      to `None`, never a failed process start) and reports the card's *global* used/free — verified
      against nvidia-smi (16376/859/15516 vs 16376/541/15521, the delta being the probe's own
      allocation). `MUMMU_GPU_BUDGET_GB` became a **ceiling**, capped by what is actually free. This
      matters on a real desktop: Firefox, Discord, Steam, Docker and two driver overlays held enough of
      this card that a 9.8 GiB placement well inside its configured budget died with `out of device
      memory` mid-generation. Also added `backend::video_memory()` (DXGI `QueryVideoMemoryInfo`) —
      useful as a per-process ceiling but **not** as a free-memory reading: Windows permits
      oversubscription and reported a ~15 GiB budget while another process held 9 GiB of the card.
- [ ] **Blocked: the layered GPU path OOMs well inside its budget, and it is the allocator, not the
      weights.** *(2026-08-23)* With 10.03 GiB of weights planned onto a card with 15.2 GiB free and a
      3 GiB activation reserve, generation dies with `failed to reserve 19660800 bytes ... out of device
      memory allocating 263143424 bytes`. Tracing nvidia-smi through the load shows VRAM going
      **451 -> 15958 MiB within 3 seconds** — before the 52 s load finishes — so that is cubecl
      reserving its pool up front, not weights accumulating, and the failure is *inside* the pool.
      Ruled out: dequantize-per-matmul (the native quantized path is active; the "no usable native
      quantized matmul" line never appears). Next step is cubecl's memory-pool configuration
      (chunk sizing / fragmentation), not another placement or precision change.
- [ ] **Wire the mix to the running controller.** The pieces exist and are tested separately —
      `mix::plan` (what precision each tensor should be), `vram::memory` (what changed), and
      `adapt::Controller` (when to react, with its dwell and alloc-failure handling). Joining them is a
      rebalance step that requantizes resident tensors down a rung under pressure and re-reads them from
      the pack on the way back up. Note from the ladder probe: **promotion should re-read the pack, not
      invert a demotion** — the pack's Q4 measured 0.0997 against 0.1090 for Q8-then-requantize, because
      the pack quantized from the original f32 while a demotion compounds two roundings.
- [x] **Use every device that helps: the integrated GPU joins the placement, and the CPU stops being
      given quantized weights.** *(2026-08-23)* Measured first, on the decode shape with a real pack
      weight (`examples/device-throughput.rs`, warm, 89 M params):

      | device | ms | GB/s of weights |
      |---|---|---|
      | dGPU (RTX 4070 Ti SUPER, wgpu) | 1.59 | 35.1 |
      | iGPU (Radeon, wgpu) | 14.15 | 3.9 |
      | CPU (flex, f32) | 13.82 | 4.0 |
      | CPU (flex, Q8 native) | 244 | 0.2 |
      | CPU (flex, Q4 native) | 268 | 0.2 |
      | CPU (flex, F16 from pack) | 13.26 | 4.2 |

      Three findings, each of which changed the code. **(1) On burn-flex a quantized matmul is 18x
      slower than a float one.** Quantization exists to fit a device short of memory; the host has
      96 GiB and is short of speed. The CPU tier's ladder started at `Q8`, so a host given a large share
      of the clusters was pushed into its slowest mode exactly when it had the most work — now
      `[F32, Q8, Q4]`. Host-resident weights on the layered path load at **F16 straight from the pack**
      (the pack already stores `f16.bin`), which is the same speed as f32 at half the bytes.
      **(2) The iGPU is level with the CPU, not above it** — its value is that it is an *additional*
      worker. Run concurrently it held 14.3 ms while the CPU ran flat out, so there is no measurable
      contention for the memory controller. It now appears as `BackendChoice::IntegratedGpu` with a
      bounded slice of system RAM **subtracted from the CPU tier's budget**, since it allocates from the
      same pool. Placement on the 27B went from `996 dGPU + 1052 CPU` to **`996 dGPU + 885 iGPU +
      167 CPU`** — 885 clusters moved off the slow path.
      **(3) cubecl has no peer-to-peer transfer for wgpu**: `comm_init` and `send` on its server trait
      are `unimplemented!()`, so a direct dGPU -> iGPU move panics with a bare "not implemented". Added
      `backend::move_to`, which stages through host memory when both ends are accelerators and is a
      no-op otherwise. Without it a two-GPU placement cannot run at all.
- [x] **When the FFN is tiered, the trunk belongs on the host.** *(2026-08-23)* Lifting the stale Q4
      exclusion let the fit planner put the trunk on the card, which *starved the clusters of the only
      fast device*: a trunk on the GPU took 7.8 of its 9 GiB and left room for 302 of 2048 clusters,
      against 996 when the trunk stayed on the host. Measured 6.2 s/tok against 4.32. This is the same
      result as the earlier trunk-on-GPU experiment (24.7 against 4.8), now with a mechanism rather than
      a mystery: every cluster runs on every token, so the card holding more of them beats the card
      holding the trunk. The candidate order in the fit planner now tries the host first **when and only
      when the FFN will be tiered** — without tiering the whole model goes to one device and the fastest
      one that fits is simply right.
- [x] **Chat over a WebSocket, so a cold load survives Cloudflare.** *(2026-08-23)* Behind Cloudflare an
      HTTP response that produces no bytes for 100 seconds is cut with a 524 — and a cold `/api/chat`
      produces none for **minutes** while a 27B is read from disk and placed across devices. SSE does not
      help: the clock runs from the request and there is nothing to stream yet. Cloudflare does not apply
      that timeout to WebSockets, so `GET /api/chat/ws` carries exactly the frames `POST /api/chat` sends
      (`start_chat` is shared, so the two cannot drift) and the UI prefers it, falling back to POST+SSE
      only if the socket never opens. Verified end to end: `101 Switching Protocols` with
      `Server: cloudflare` and a `CF-RAY` header, deltas and a `done` frame.
      **The upgrade alone was not the fix.** An idle WebSocket is reaped too, so the server pings every
      15 s through the load — and the first implementation of that *did not work*: the heartbeat task was
      not polled **once in 557 seconds**, then fired 38 missed pings in a burst when the load finished, by
      which point the connection would already be gone. The load is minutes of CPU-bound work with no
      await in it, and as a plain async task it starved the runtime. Fixed narrowly, in `mummu::cache`,
      by wrapping **just the load** in `block_in_place` (guarded on a multi-thread runtime, since it
      panics on the current-thread flavor the tests use) — not by pushing the whole generation onto the
      blocking pool, which would have turned the async decode loop back into blocking work. Pings then
      arrive at 16/31/46 s and hold a 544-second connection. `MissedTickBehavior::Delay` on the interval,
      because a burst of late pings proves nothing to a proxy that already timed out.
- [ ] **The gap to a fused runtime is a KERNEL problem, not a placement one — measured, and the reason
      scheduler tuning stops here.** *(2026-08-24)* Five measured iterations took the 27B from 4.88 to
      **3.91 s/token** and moved the discrete GPU from 996 to **1784 of 2048 clusters**. Then the
      measurements said to stop. `examples/cluster-overhead.rs`, one full-width FFN matmul on the 4070
      Ti SUPER, warm:

      | rung | ms | vs f32 |
      |---|---|---|
      | f32 | 1.68 | 1.00 |
      | Q8 | 1.60 | 0.95 |
      | Q4 | 1.64 | 0.98 |
      | **F16** | **1.09** | **0.65** |

      **Quantization buys capacity and no speed.** Decode is bandwidth-bound, so a 4-bit weight moving an
      eighth of the bytes should be ~8x faster; it is not, because burn's quantized matmul materializes
      f32 regardless. Effective bandwidth makes it plain: **221 GB/s** counting f32 bytes against
      **34.6 GB/s** counting the Q4 bytes actually stored, on a 672 GB/s card.
      That reproduces both sides of the comparison exactly. mummu moves 108 GB/token of f32-equivalent
      traffic (~489 ms at 221 GB/s); ollama moves 16.9 GB/token of 4-bit traffic (~62 ms at 272 GB/s),
      and 62 ms is what ollama measures. **So the entire model on the fast card with zero transfers is
      still 7.7x short of ollama before placement enters the picture.** Scheduling was optimizing a term
      that is not the bottleneck.
      **F16 is faster but not exploitable while capacity binds**: 1.55x per matmul against 3.2x the
      bytes, so all-Q4 (2048 clusters resident, 264 ms) beats all-F16 (1811 resident, 237 spilled,
      272 ms). The scheduler already lands on the right side of that, and `mix`'s phase 2 promotes into
      F16 automatically once a model fits — no change needed.
      **What is left, ranked by measured size.** (1) A CubeCL matmul that reads packed weights directly,
      the way llama.cpp's Q4_K kernels do — worth the 6.4x traffic ratio, and the only route to parity.
      (2) Wider FFN clusters: 2 x 8704 costs 1.33x fused against 32 x 544's 2.36x, so ~1.8x, but cluster
      width is a pack-time choice and needs a repack. (3) Everything else measures under 1.2x.
- [x] **The ~28 ms/layer merge wait was the GPU's real compute time, hidden behind wgpu's
      deferred-mapped readbacks — found after seven falsified theories, and it re-ranks nothing:
      the packed kernel was already the answer.** *(2026-08-25)* wgpu's `into_data` returns in 1-4 ms
      while the FIRST CPU TOUCH of the returned bytes blocks on the GPU fence
      (`examples/mapped-wait-probe.rs`: first touch 27.0-27.5 ms behind a queued GPU chain, three
      rounds). So every drain design paid the remote FFN's device time at whatever op first read the
      bytes — which is why the cost was thread-agnostic, migrated between profiler scopes as code
      moved, and ignored VRAM budgets (13 vs 15 GiB both churn and both stall), timer resolution,
      thread priorities, and every queue theory. Falsified along the way, each by measurement or
      source reading: GPU-wait-at-join (worker clocks), lazy cross-backend moves (`read_sync` in
      dispatch), flex laziness (type-level), stderr/backtrace locks (quiet run), pool-panic churn
      (zero-panic run stalls identically), Windows timer quanta (`timeBeginPeriod(1)` changed
      nothing), cubecl's bounded 32-slot device queue (burn-flex has no cubecl channel — its ops are
      inline on the calling thread), and worker core starvation (an above-normal-priority worker
      submits at +0.5 ms and the join still waits the full fence).
      The layer timeline (`MUMMU_TRACE_LAYER`) closed it: on remote-heavy layers the planner keeps
      ~1 local cluster, the caller reaches the join at **+1.2 ms**, and the GPU needs **~26.5 ms**
      for its 28-29 clusters — nothing exists to overlap against. Per-cluster at m=1: **dGPU
      0.91 ms vs host 0.40 ms** — the discrete GPU is 2.3x SLOWER per cluster than the CPU it
      relieves, the latency-side face of the 7.7x bandwidth measurement above. Scheduling is DONE:
      enqueue-first ordering, deferred join, fence absorbed on the worker, priority boost — all in
      place, all correct, all irrelevant next to the kernel. What ships from the hunt: the
      third-design drain in `nn/moe.rs` (per-executor readback INSIDE the kernel-gap catch — the
      panic surfaces at the read, so the old catch around the compute could never see it; plain-f32
      accumulate on the worker; bytes built on the caller's captured device with the readback's
      dtype), `mapped-wait-probe` + the timeline trace as permanent instruments, and a corrected
      account in the drain's design doc. The radial-lookahead alternative (start layer L+1 on the
      known prefix during L's drain; exact for the parallel component under RMSNorm's radial split)
      was built flag-gated in verify mode and retired on its own kill criterion: acceptance 0% at
      1e-2, worst rel err 0.71, ||b_perp||/||a|| 0.17-0.75 against a ~0.05 viability bar. Interim
      lever worth a planner experiment: clusters on the HOST are 2.3x cheaper than on the dGPU at
      m=1, so shifting remote clusters host-ward wins latency wherever host memory allows (Q4-on-host
      would clear the F16 budget wall); the real fix stays (1) in the ranked list above.
- [x] **The packed m=1 GEMV shipped — on every backend — and it un-hid two more starved service
      threads. Warm decode 4.03 -> 3.19 s/token; the pool churn is dead.** *(2026-08-25)*
      `nn/packed_gemv.rs`: a `#[backend_extension]` op reading the stored quantization directly —
      a `#[cube]` wgpu kernel (one unit per packed u32 word: one word + one shared scale + one x[k]
      broadcast -> eight FMAs, no cross-unit reduction; parity 4e-7 against dequantize-then-matmul
      at production shapes), a rayon i8 GEMV over flex's unpacked storage, and a Fusion CustomOpIr
      wrapper. Wired into the cluster executor and `qlinear` under the existing downgrade contract
      (`MUMMU_PACKED_GEMV=0/off/false` reverts). Prefill runs row-by-row through the same exact op,
      so the dequantize-first fallback's ~260 MB transients never exist — VRAM pool panics went
      505-710 per run -> **0**.
      With real GPU work sub-millisecond, the residual ~26 ms fence decomposed into thread-priority
      bugs, not device time: cubecl's `DSD-*` device server AND cubecl-wgpu's unnamed poll thread
      (the one that signals readback-map completions) both starved at normal priority behind the
      trunk's spinning gemm herd. `backend::boost_device_server_threads()` raises the named servers
      after load; mummu-serve starts the global rayon pool with every gemm worker at BELOW_NORMAL,
      so microsecond-scale service threads always preempt spinners. Fence 26.5 -> 8.7 ms/group.
      Then the host slab flipped to Q4 (i8-resident, 3.6x less RAM than the F16-widened slab;
      `MUMMU_HOST_SLAB=f16` reverts) with the planner pricing flex residency at its true 9/5 of
      packed bytes. Measured sum on the 27B, same session, quiet box: warm **3.19-3.25 s/token**
      from a 4.03 baseline, cold request **139 s** wall (from ~450), host RAM **-14 GiB**, plan 552
      host + 1496 wgpu clusters.
      **Read that pair as a within-session RATIO, not an absolute.** Re-measuring the identical
      commit hours later on an idle box gave **3.46-3.57**, and merged main gave **3.65-3.77** —
      the same binaries, drifting ~10% between sessions. Worse, the first run after a 20 GB cold
      load measured **4.70-7.66** while the machine was still settling, a 2.4x spread that would
      read as a catastrophic regression to anyone comparing it against a number from another day.
      This box's decode gate is only meaningful as (quiet, warm, >=3 runs, same session), and a
      cross-session absolute is worth about +/-10% at best. A bisect against the pre-merge commit
      cleared the 2026-08-25 nightly of the apparent 13% loss: the residual ~5% does not separate
      from noise at this sample size. Standing figure for main: **~3.5-3.8 s/token**, ~59x off
      ollama's measured 16.0 tok/s on this box and checkpoint.
      Next, ranked: (1) the residual ~8.5 ms/group fence — suspected polling handoff, worth a
      timeline pass now that priorities are clean; (2) SIMD the flex i8 inner loop (mlp.* is
      ~3.3 ms/call, still ~4x off the traffic bound); (3) a real m<=64 packed GEMM for prefill;
      (4) the CPU trunk (delta.proj 13.8 ms/call is now the top scope) — relocation or Q8+GEMV;
      (5) MTP/NextN. The 62 ms/token ollama wall still requires the trunk off the host.
- [ ] **Do NOT make the clustered path universal until a fits-on-one-device fast path exists.**
      *(2026-08-24)* Partitioning earns its keep only when a model must span devices. Measured, the
      split costs **2.4x** in dispatch (32 cluster matmuls 4.13 ms against 1.68 ms fused), so imposing it
      on a model that fits one card would make every small model 2.4x slower to buy placement
      flexibility it never uses. The generalization is right; it needs the planner to skip partitioning
      when scheduler A puts everything on one device.
- [x] **The placement-dependent decode panic, root-caused: burn's wgpu q_matmul cannot launch some
      m=1 x Q8 shapes — so wgpu now always dequantizes before the multiply.** *(2026-08-24)* The
      symptom seen twice (iteration 4, and again after the drill-down deploy): prefill works, the first
      single-token decode step kills the generation, and whether it happens depends on the tier plan.
      Reproduced with a captured backtrace: `cubecl-std quant/view.rs:223 — quantized view float vector
      size 1 must be a positive multiple of num_quants 4`, on the enqueue thread. `num_quants 4` is
      Q8S; the kernel chosen for an m=1 matmul vectorizes the float side at 1 for some weight widths,
      and the cluster grouping decides the widths — which is why iteration 5's placement "fixed" it and
      a later plan brought it back. NOT the retracted probe-artifact claim: this one has a backtrace
      from production and a deterministic reproduction.
      Fix in `nn::compute_weight`: on wgpu (discrete and integrated), a quantized weight is always
      dequantized before `matmul`. Measured cost: none — 1.61 ms native vs 1.61 ms dequantized for a
      full-width FFN matmul, bit-identical results. Storage stays packed, so capacity is untouched;
      CUDA keeps its probed native path. Known remaining exposure, deliberately not patched yet:
      `qwen35::qlinear` multiplies `Param` weights raw, so a future whole-model-on-wgpu load at Q8
      (the layered path) would hit the same kernel gap — route it through the same guard when that
      path comes back into use.
- [x] **Surviving memory pressure: bounded dequantize transients, an unquantized iGPU, and partials
      that fail loudly.** *(2026-08-24)* The wgpu dequantize guard (previous entry) traded a panic for
      an OOM: per-call dequantize of a ~19-cluster group is a ~210 MB f32 transient, cubecl's pool
      allocates ~1 GiB chunks to hold those, and with other apps holding 5 GiB of the card the discrete
      GPU died around the 8th token. Three changes, each earning its place the hard way:
      **(1) Accelerator executor groups cap at 8 clusters** (`WGPU_GROUP_MAX_CLUSTERS`): transients drop
      to ~90 MB, and uniform widths let the pool and autotune converge on a handful of shapes instead of
      one per layer. Dispatch cost is noise — the full per-cluster split measured 2.4x, so ~3 groups
      instead of 1 is a few percent. The host keeps whole groups; flex has no pool chunking.
      **(2) The integrated GPU never touches a quantized weight** — its ladder is F32-only. Both packed
      options broke on it in production (the m=1 kernel panic; the dequantize churn), its memory is
      system RAM where f32 residency is the cheap resource, and f32 is what its 14.15 ms/cluster rating
      was measured on. Two follow-on bugs from this change, both caught before merge: the
      source-precision cap emptied an [F32] ladder and the empty-ladder fallback pushed a hardcoded Q4 —
      silently re-arming the exact kernels the ladder was narrowed to avoid (fallback now keeps the
      device's own coarsest rung); and with F32 the expensive slot, phase-1 placement's "strictly
      faster only" gate made the iGPU **unreachable** — admission parks everything on the cheap host
      slot, and 71 &lt; 72 meant zero clusters ever moved to it (found by adversarial review, verified
      line-by-line against `plan_tiers`). Phase 1 now trusts scheduler A's quota as the arbiter: an
      equal-speed idle device relieves a holder that is past its balanced share, which the
      trunk-preloaded host always is. Pinned by `an_equal_speed_idle_device_relieves_an_overloaded_one`.
      **(3) A panicking FFN worker now fails the generation instead of the answer.** `run_dense`'s
      parallel path collected partials with `h.join().ok()`, so a worker that died from the iGPU OOM
      had its device's entire cluster contribution silently dropped from the FFN sum — producing a
      fluent, wrong reply ("Blue, red, and green." with 562 clusters missing from every layer). Worker
      panics now `resume_unwind` on the caller. A wrong answer that parses is strictly worse than an
      error.
      **(4) The dequantize guard narrowed to the family that actually breaks.** Every `quant/view`
      panic in every log says `num_quants 4` — four values per u32, the 8-BIT schemes. Q4S packs eight
      and never produced the panic; blanket-dequantizing it was pure OOM fuel (44 device-memory panics
      in one three-request run, zero of them quantized-view). Now on wgpu only Q8S/Q8F/E4M3/E5M2
      dequantize — small groups, tiny transients — while Q4S runs native with zero transients, at
      measured-identical speed either way. The lesson generalizes: when a workaround's blast radius is
      bigger than the bug's, narrow the workaround until the logs say the bug's family and nothing else.
      **(5) ...and the transient cap turned out to be its own width bug.** With Q4 native again, three
      fresh panics appeared reading `num_quants 8` — Q4S's variant of the same vector-size assert. The
      8-cluster cap (added to bound dequantize transients) had forced native Q4 groups to width 4352,
      inside the kernel's bad regime, while the organic 17-27-cluster widths had run dozens of tokens
      across four loads without incident. The width rule lives inside cubecl's kernel selection and is
      not knowable from outside, so the cap now applies ONLY to groups that actually dequantize (the
      Q8 family), and native groups keep their proven organic widths. Upstream issue to file against
      burn: wgpu q_matmul, m=1, both `num_quants 4` and `num_quants 8` reproduced with exact shapes.

## Phases

### P0 — Workspace scaffold
- [ ] **Prepare the burn 0.22 migration** — v0.22.0-pre.1 (2026-07-29) is a breaking release aimed right
      at Mummu's core seams: (a) the `Tensor` **backend generic is removed** (a high-level `Device`
      struct replaces it) and (b) backends **lose associated element types** (`B::FloatElem` /
      `B::IntElem`) in favor of per-device defaults — which dissolves the `Gpu`/`GpuF16` alias split AND
      the type-level dtype pinning shipped 2026-07-30 (`backend::{float_dtype,int_dtype}` read
      `B::FloatElem`; under 0.22 the explicit-dtype path becomes the only correct one, likely
      simplifying the P6 mixed-precision story since dtype stops being a type parameter at all). Also
      relevant: burn-store ships its own `FloatCastAdapter` (evaluate replacing our `CastFloatAdapter`),
      burnpack moves to a `burn-pack` crate, and wgpu 30 lands with it (f16 beyond SPIR-V — the P6 note).
      Do NOT adopt a pre-release; when 0.22.0 stabilizes: migrate on a branch, re-run every parity gate +
      budget, and expect the backend aliases + dtype helpers + all loaders' `target_float` derivation to
      change shape. *(2026-07-30 research)* — https://github.com/Tracel-AI/burn/releases
      *(2026-08-06 research)* Still pre-release (0.22.0-pre.1 is the newest tag; 0.21.0 the latest stable),
      so this stays gated — but reading the pre-release notes properly turns it from a migration *cost*
      into the run's most valuable pending item, because **0.22 ships graph capture, explicitly to cut
      CPU-side launch overhead**. That is the exact bottleneck the dispatch-bound decode item has been
      chasing since 2026-07-11 (an f16-matches-f32 measurement, then SPIR-V, then this run's flash-attention
      A/B all pointing at per-dispatch cost rather than bandwidth), and it is the one lever that attacks it
      *generically* rather than one kernel at a time. Three more items arrive with it: **LoRA/QLoRA** land
      in-framework (P10 becomes wiring, not implementation), the **remote backend** gains multi-device +
      client-side operation-graph caching + async reads (P6 multi-GPU), and quantization gains
      dequant→op→quant fallbacks for slice/gather/select/expand plus BitNet b1.58 calibration (P9). Plan the
      migration around measuring graph capture on the decode loop first — if it lands the dispatch win, it
      reorders everything below it in the perf section.
      *(2026-08-20)* Still gated: crates.io now serves **0.22.0-pre.2**, so 0.21.0 remains the newest
      stable and the do-not-adopt-a-pre-release rule holds for another run. Checked as part of the
      dependency sweep, which is also why **wgpu 30 stays held**: `cargo upgrade --incompatible`
      offers it, but `cargo tree -i wgpu` shows exactly one wgpu in the graph (29.0.4, reached via
      `cubecl-wgpu 0.10`), so bumping our direct handle alone would put a second, non-Burn wgpu in
      the tree and the startup adapter probe would stop describing the device Burn actually runs on.
      wgpu 30 unblocks with the burn bump, exactly as the Stack note says — not before.
      *(2026-08-20 research)* One refinement to the migration shape: the associated element types are
      not simply deleted — they move off `Backend` onto a new **`BackendTypes`** trait, and the release
      notes steer callers to the **type aliases** (`Device<B>`, `FloatTensor<B>`) instead of naming
      associated types directly, specifically to dodge resolution problems. That is actionable now, at
      zero migration risk: every place we write `B::FloatElem` / `B::IntElem` by hand (`backend::
      {float_dtype,int_dtype}` and each loader's `target_float` derivation) is a place the 0.22 diff will
      land, so preferring the alias form where one already exists shrinks that diff before the bump.
      — https://github.com/tracel-ai/burn/releases
      *(2026-08-20, second run)* Two corrections and one addition, all from reading what is actually
      published rather than the migration note's summary. (1) **`burn-store` 0.21 does NOT ship a
      `FloatCastAdapter`** — its adapter set is `PyTorchToBurnAdapter` / `BurnToPyTorchAdapter` /
      `HalfPrecisionAdapter` / `ChainAdapter` / `IdentityAdapter`, and `HalfPrecisionAdapter` is
      f32<->f16 only, not the arbitrary target-dtype cast `CastFloatAdapter` does. So "evaluate
      replacing our `CastFloatAdapter`" is a 0.22 task, not something already available and skipped.
      (2) **wgpu 30's f16 story is bigger than "f16 beyond SPIR-V"**: `SHADER_F16` now works in WGSL
      and GLSL as well as SPIR-V passthrough (add `enable f16;` at the top of the shader), which
      matters because it means the f16 win stops being Vulkan-only — the `vulkan` feature's SPIR-V
      path is currently the only way we get f16 kernels, so a DX12/Metal consumer gets f32 today and
      would not after the bump. (3) wgpu 30 also lands **cooperative matrix load/store** (WGSL in;
      SPIR-V/Metal/WGSL out), gated on `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` and
      currently **8x8 f32 only** — too narrow to be the decode lever yet, but it is the seam a
      tensor-core matmul would eventually ride, so watch the supported configurations grow.
      — https://docs.rs/burn-store/latest/burn_store/ · https://github.com/gfx-rs/wgpu/releases
- [x] Silence the pre-existing `LNK4098` (LIBCMT defaultlib conflict) the 2026-07 nightly toolchain's
      new `linker_messages` lint now surfaces when linking the `mummu` lib-test binary — find which
      native dep object embeds the static-CRT directive (tokenizers' C++ deps are the suspects) and
      align it, rather than allowing the lint. *(2026-07-16, predates this run's changes — verified by
      an A/B build at HEAD.)* *(2026-07-16, later run) **Fixed at the source, no lint allow.** The
      culprit is `esaxx-rs`: its build.rs hardcodes `.static_crt(true)`, so `esaxx.lib` embeds
      `/DEFAULTLIB:LIBCMT` + `/FAILIFMISMATCH:RuntimeLibrary=MT_StaticRelease` while rustc-msvc links
      the dynamic CRT (`dumpbin -directives` confirmed; the sibling `onig.lib` correctly carries
      `/DEFAULTLIB:MSVCRT`). Upstream is aware and stalled — Narsil/esaxx-rs#11 open, its PR #19
      (`static_crt(false)`) unmerged — so the fix is ours: tokenizers now resolves with
      `default-features = false, features = ["onig", "progressbar"]`, dropping only `esaxx_fast`.
      That feature exists solely to accelerate the Unigram **trainer** (`models/unigram/trainer.rs`
      picks `esaxx_rs::suffix` over the pure-Rust `suffix_rs`), which a runner never runs; `onig` is
      load-bearing (`SysRegex`) and stays pinned. `cargo tree -e features` now resolves `esaxx-rs`
      with **no features** — its build.rs compiles nothing, so the C++ object is gone rather than
      merely tolerated. A/B proof: LNK4098 1 → 0 on a forced relink, zero warnings. Behaviour proof:
      BOTH GGUF tokenizer byte-identity tests still pass (`real_{qwen2,lfm2}_gguf_tokenizer_matches_the_hf_tokenizer`),
      and all three parity gates re-passed — Qwen2 max |Δlogit| 2.29e-5 + Ollama greedy byte-identical,
      LFM2 top-5 exact at 1.4879674843625068e-2, GGUF quantized leg unchanged.
      https://github.com/Narsil/esaxx-rs/issues/11 · https://github.com/Narsil/esaxx-rs/pull/19
- [x] The real-weights **GPU tests overflow the stack in the `dev` profile** — `real_gguf`'s
      `real_{lfm2,qwen2}_gguf_loads_and_decodes_on_gpu` die with `STATUS_STACK_OVERFLOW` on a CubeCL
      worker thread (`DSD-4-0`), while the identical tests pass in `--release` (7/7 in 76.9 s). Verified
      **pre-existing** by an A/B at unmodified HEAD, so it is not the tokenizers-feature change. Until
      it is root-caused, the real-model suites must be run `--release`; the fix is likely a bigger
      spawn-time stack for the CubeCL worker (or a debug-profile `opt-level` bump for the burn stack).
      *(2026-07-16)* *(2026-07-16, same run) **Fixed** — it was the `opt-level`, not the stack size:
      `[profile.dev.package."*"] opt-level = 2` optimizes **dependencies only**, so burn/CubeCL's deep
      generic tensor call chains stop keeping every inlinable frame live at once, while our own crates
      stay `-O0` and fully debuggable. The whole real-model suite now runs in `dev`: `real_gguf` 7/7
      (262 s vs 76.9 s in release — slower, but no longer impossible) and `real_inference` 4/4.
      `[profile.release]` is untouched, so parity and the perf budgets are unaffected by construction.
      Dropping `--release` from the real-model suites is now a choice, not a workaround.*
- [x] Cargo workspace: `crates/mummu` (the library), model code generic over `B: Backend`. `.gitignore`
      (Rust); commit `Cargo.lock` for reproducible builds/benchmarks. *(2026-07-09) Workspace + both
      crates; pinned combo burn 0.21 / wgpu 29 / tokenizers 0.22 / criterion 0.7; release profile fat-LTO.*
- [x] `cargo build` / `test` / `clippy --all-targets` green baseline; `mummu-bench` (criterion) crate stub.
      *(2026-07-09) All green; criterion harness wired via a smoke bench.*
- [x] The **`candle-core` entries in the workspace `Cargo.lock`** are correct and not prunable — root
      cause was misdiagnosed. They are NOT `tools/candle-probe` orphans (that is a separate workspace with
      its own lock); they come from **`burn`'s optional `burn-candle` backend**: `burn` declares
      `burn-candle` as a feature-gated optional dependency, `burn-candle → candle-core → tokenizers 0.22.2`,
      and Cargo.lock records a package's full *optional*-dependency closure regardless of which features are
      enabled. `cargo tree -i candle-core` prints "nothing to print" (nothing reaches it under our enabled
      features) precisely because it is an unenabled optional — that is expected, not an orphan. Proof it is
      inherent: a from-scratch `rm Cargo.lock && cargo generate-lockfile` re-adds candle-core (and the
      0.22.2 tokenizers it pulls). Nothing to fix — it is a normal, harmless artifact of Burn shipping a
      Candle backend behind a feature we never turn on; it costs zero compile time (never built) and would
      only disappear if Burn stopped declaring the optional dep. *(2026-07-17)*

### P1 — Backends & device *(ex-laurelane)*
- [x] Backend abstraction generic over `B: Backend`; one binary compiling BOTH `Wgpu` (Vulkan/DX12/Metal,
      no CUDA toolchain) and `NdArray` (CPU). *(2026-07-09) `backend::{Gpu, GpuF16, Cpu}` aliases; no
      feature splits.*
- [x] Runtime `use_gpu()` probe (`wgpu` adapter enumerate; `pollster::block_on` for the async probe),
      cached in a `OnceCell`; pick GPU else CPU. No feature-split builds. *(2026-07-09) `backend::inventory()`
      also records per-adapter/per-API `SHADER_F16` for the P6 planner; on the dev box: 4070 Ti SUPER
      f16=true on Vulkan, false on DX12, + an integrated AMD GPU (a real second adapter for multi-GPU).*
- [x] `fusion` + `autotune` on (`Wgpu` becomes `Fusion<Wgpu>`; needs `recursion_limit = 512`).
      *(2026-07-09) Workspace features + crate-level `recursion_limit`.*
- [x] Evaluate **burn-flex** (Burn 0.21's new pure-Rust CPU backend; `burn-ndarray` is now on a
      deprecation path) as the `Cpu` alias replacement — SIMD + gemm, no_std, and built-in per-tensor/
      per-block quantization (~40 quantized ops) that P9 could ride on. Gate on parity + a CPU decode
      bench — https://github.com/antimora/burn-flex · https://burn.dev/blog/release-0.21.0/
      *(2026-07-10) **Swapped in** (`Cpu = burn_flex::Flex<f32, i32>`, ndarray feature dropped): the
      MiniLM Candle-parity gate passes on Flex (cosine 0.99999994, max |Δcomponent| 1.3e-7 — equivalent
      to ndarray's 1.2e-7) and all 80 unit tests are green, incl. the cache-equivalence proofs that run
      on the CPU backend. A dedicated CPU decode tok/s bench still wants a CPU-tier model (0.5B) — next
      item.*
- [x] CPU decode bench for `bench/BASELINE.md`: pull Qwen2.5-0.5B (catalog entry exists) and record
      decode tok/s on the Flex backend, so CPU-only machines get a budget row and Flex regressions are
      caught like GPU ones. *(2026-07-10) 0.5B fetched through the registry/hub path (988 MB in ~13 s),
      decodes coherently on Flex at **11.7 tok/s** (7950X3D, f32); budget ≥ 6 tok/s gated by
      `mummu-bench/tests/budget_cpu.rs`.*

### P2 — Model zoo (from scratch, generic over `B`) *(ex-laurelane)*
- [x] Shared blocks: RmsNorm, GQA attention, RoPE (manual rotate-half), SwiGLU MLP, tied lm-head,
      depthwise causal conv (for hybrids). *(2026-07-09) `mummu::nn`: cache-aware `GqaAttention`
      (optional per-head q/k RMSNorm covers Qwen2 AND LFM2), `SwiGluMlp`, `ShortConv` (LIV) with rolling
      state, RoPE + causal mask; 18 unit tests incl. the prefill+decode ≡ full-forward equivalence for
      both the KV cache and the conv state. RmsNorm/tied-head come from burn::nn / the model files.*
- [x] **Qwen2 / Qwen2.5** decoder (1.5B / 0.5B tiers). *(2026-07-09) Ported (`models::qwen2`,
      config-driven, checked safetensors load); REAL GPU inference verified — Qwen2.5-1.5B on the 4070 Ti
      SUPER greedy-decoded "2+2 equals 4.", top-5 probe led by id 9707 "Hello"; toy-model cache-equivalence
      unit test.* *(2026-07-10) **Parity gate PASSED** (`tests/parity_qwen2.rs`): top-5 logits match the
      Candle f32 reference exactly by id with max |Δlogit| 2.7e-5 (bound 1e-3), and a 24-token greedy
      sequence matches `ollama qwen2.5:1.5b-instruct-fp16` byte-for-byte on the 4070 Ti SUPER.*
- [x] **LFM2.5-1.2B** hybrid (6 GQA-attention w/ per-head q/k RMSNorm + 10 double-gated short-conv "LIV"
      blocks, SwiGLU, tied head, conv-state cache; ChatML, EOS `<|im_end|>`). *(2026-07-09) Ported
      (`models::lfm2`, hybrid cache, LFM2→shared-block key remap); toy hybrid cache-equivalence test;
      greedy-vs-Ollama parity test written (`tests/parity_lfm2.rs`). REAL GPU inference verified — the
      1.2B greedy-decoded a correct, coherent primes list on the 4070 Ti SUPER. Stays `[ ]` until the
      gate passes: the local `ollama lfm2.5:latest` tag now resolves to the 8.5B **MoE** Q4 w/ thinking
      (verified via `ollama show` 2026-07-09) — not the same weights, so no valid local reference exists;
      see the P7 reference item.* *(2026-07-16) **Parity gate PASSED** (`tests/parity_lfm2.rs`, both
      legs) vs the llama.cpp same-weights reference the P7 item stood up: top-5 first-forward ids match
      exactly in order (max |Δlogprob| 1.49e-2 — the reference's own bf16-activation rounding; our Qwen2
      f32-vs-f32 comparison sits at 2.7e-5) and a 24-token greedy sequence is byte-identical on the
      4070 Ti SUPER.*
- [x] **all-MiniLM** BERT sentence-embedder (6-layer post-LN bidirectional attention + GeLU FFN,
      masked-mean-pool + L2-normalize). *(2026-07-09) Ported (`models::minilm`, ids+mask in → L2-normalized
      embedding out; tokenization stays caller-side); unit tests incl. padding-invisibility; real-weights
      semantic test (`tests/real_minilm.rs`).* *(2026-07-10) **Parity PASSED**: embedding matches the
      Candle f32 reference (`minilm-probe` fixture) at cosine 0.99999994 with max |Δcomponent| 1.2e-7
      (bound 1e-4); semantic sanity re-verified on real weights (paraphrase 0.556 vs cross-topic ≈ 0).*
- [x] **Qwen3.5 small tier** as the next zoo port: released Feb 2026 in 0.8B / 2B / 4B / 9B, with
      2026 GGUF re-releases specifically improving tool-calling (chat-template fixes) — the 4B/9B are
      the function-calling sweet spot BFCL identified (9B 66.1%), and unsloth ships ready GGUFs for the
      P3 import path to chew on; parity reference = Ollama `qwen3.5:9b` (already pulled locally) —
      https://unsloth.ai/docs/models/qwen3.5 · https://huggingface.co/unsloth/Qwen3.5-9B-GGUF
      *(2026-07-12 research)* *(2026-07-16 research)* Two refinements when this is picked up: the
      tool-calling chat-template fix landed across **all** quant uploaders (it improved nested-object
      parsing — re-pull any GGUF cached before it, ours included), and unsloth ships separate
      `-MTP-GGUF` repos for the 4B/9B whose **MTP speculative decoding** claims ~1.5–2× decode; MTP
      needs a draft-token verify loop our `decode` driver does not have, so treat it as a distinct
      P5 item rather than a free win of the port. *(2026-07-17) **Qwen3 dense architecture shipped**
      (`models::qwen3`) — the arch the Qwen3 AND Qwen3.5 dense tiers share, so this is the load-bearing
      half of this item. It reuses the shared `nn` blocks verbatim: the three deltas from Qwen2 (per-head
      q/k RMSNorm over `head_dim`, no q/k/v bias, a **decoupled** `head_dim` where `num_heads·head_dim`
      need not equal `hidden`) were all already covered by `GqaAttention`'s `qk_norm_eps` path + the
      independent `head_dim` in `GqaAttentionConfig` — HF Qwen3 orders qk-norm identically to the LFM2
      path we validated (`q_norm(q_proj(x).view(b,t,nh,hd)).transpose(1,2)`), so ZERO nn changes. Config
      (json + `qwen3.*` GGUF metadata, `key_length`→head_dim), safetensors + GGUF loaders (q_norm/k_norm
      key remaps, tied/untied head), `CausalLm` impl, 10 unit tests (decoupled-head_dim path,
      cache≡full-forward, gguf name map incl. qk-norms). Catalog: `qwen3-0.6b`, `qwen3-4b` (+ Q4_K_M
      GGUF specs). **REAL-GPU verified** on Qwen3-0.6B (28 layers, hidden 1024, 16h/8kv, head_dim 128
      decoupled, tied — `tests/real_qwen3.rs`): loads + greedy-decodes a correct answer from BOTH the
      bf16 safetensors AND the Q4_K_M GGUF alone (tokenizer-from-GGUF byte-identical to `tokenizer.json`
      on the prompt); the GGUF vs bf16 builds agree on the top first-token id (151667) at logit cosine
      0.989. **The arch is now parity-verified** (strict `[x]` leg below — byte-identical greedy vs
      llama.cpp on the same Q4_K_M). Stays `[ ]` only for the specific Qwen3.5-4B/9B **FC** target: a
      catalog run on those larger weights + tool-calling validation (the Hermes template machinery Qwen2.5
      already proved covers Qwen3, so this is a download + FC decode, not new architecture work).*
      *(2026-07-18) **The FC path is now real-GPU-proven on the Qwen3 dense arch** — the "download + FC
      decode" the target needs, demonstrated on the 0.6B tier so the only remaining variable is the larger
      weights. `tests/real_toolcall_qwen3.rs`: the checkpoint's imported `tokenizer_config.json` template is
      detected **Hermes** (`tok_config::tool_call_convention`), that selects `ChatMl::qwen2().render_with_tools`,
      and Qwen3-0.6B greedy-decodes (on the 4070 Ti SUPER) a `<think>…</think>` block followed by a clean
      `<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>` which `parse_tool_calls`
      round-trips — end to end from the model's own template metadata. What is left for the item proper is a
      catalog run on the 4B/9B weights (a download + the same decode).*
      Qwen3.6 (35B-A3B) is also out now but is MoE and
      well past the single-card tier this zoo targets —
      https://huggingface.co/unsloth/Qwen3.5-4B-GGUF · https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks
      *(2026-07-30)* **Premise correction — Qwen3.5 is NOT the qwen3 dense arch.** A header probe of
      `unsloth/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf` (HTTP-range fetch, metadata keys read directly)
      shows `general.architecture = "qwen35"` with `qwen35.ssm.{conv_kernel, state_size, group_count,
      time_step_rank, inner_size}`, `qwen35.full_attention_interval`, and `qwen35.rope.dimension_sections`
      — a **hybrid linear-attention/SSM + periodic full-attention architecture** (Qwen3-Next-style), not a
      dense decoder. So "a download + the same decode" does NOT cover Qwen3.5: it needs its own from-scratch
      port (new SSM/gated-delta blocks + the hybrid cache machinery the LFM2 port pioneered, plus the
      `qwen35.*` config/name maps) and a fresh parity gate. Re-scoped: the *FC-tier-at-4B* half of this item
      rides **Qwen3-4B dense** (catalog entry existed; FC decode proven this run — see below), and the
      Qwen3.5 hybrid port is now its own P2 architecture item beside the MoE one.
      *(2026-07-30, same run)* **FC tier at 4B PROVEN — item closed on the dense arch.**
      `tests/real_f16.rs::qwen3_4b_gguf_downloads_and_emits_a_tool_call_in_f16`: the catalog's
      `qwen3-4b-q4km` spec downloaded `Qwen3-4B-Q4_K_M.gguf` (2.50 GB) through the registry's
      resumable/hash-verified fetch, the header asserts arch `qwen3`, the one file loaded on **GpuF16**
      (the only precision a 16 GB card fits — ~8 GB resident), and from a `ChatMl::qwen3()` tools
      prompt it greedy-emitted a `<think>` block + a clean
      `<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>` that
      `parse_tool_calls` round-tripped — on the 4070 Ti SUPER, under heavy VRAM contention (15.7/16 GiB
      in use). The 9B tier stays out of reach on this card (18 GB in f16) until P9 keep-quantized or a
      P6 placement plan makes it fit.
      *(2026-07-22 research)* The mid-2026 function-calling field guides converge on the same picture: for
      ≤ 8 GB cards **Qwen3.5-9B** is the general-purpose FC pick and **Qwen3.5-4B** the CPU-only pick
      (reinforces the 4B/9B target of this item); the Qwen3.6 tier (27B dense / 35B-A3B MoE) ships a new
      `qwen3_coder` tool-call parser in vLLM/SGLang — if its emission format differs from Hermes ChatML,
      a Qwen3.6-class port would need a new `chat` convention entry, so check the template's markers before
      assuming Hermes covers it — https://insiderllm.com/guides/function-calling-local-llms/ ·
      https://www.popularai.org/p/best-cpu-only-local-llm-2026
- [x] **Qwen3 strict parity gate PASSED** — the Qwen3 dense arch is now through the P7 trust gate.
      `tests/parity_gguf.rs` gained a `qwen3` leg on the `llama_ref` harness: `llama-server` (Ollama's
      bundled binary) runs the SAME local Qwen3-0.6B Q4_K_M our loader loads, RAW `/completion` with
      token-id prompts + `n_probs`. On the 4070 Ti SUPER: top-5 first-forward ids match **exactly in
      order** (all 5: 151667, 151644, 151645, 99966, 131545) and the 24-token greedy sequence is
      **byte-identical** to llama.cpp — reproducing the `<think>…` reasoning tokens verbatim (both sides
      get the identical prompt-id array and greedy-decode, so thinking mode is irrelevant to parity; no
      template stack on either side per the LFM2 caveat). max |Δlogprob| **4.02e-1**, inside the shared
      7.5e-1 tolerance (a touch above Qwen2's 2.66e-1 / LFM2's 2.60e-1 — the reference's Q8_K
      activation-quant noise is proportionally larger for the narrow 0.6B). Run with
      `MUMMU_QWEN3_GGUF_PATH` + `MUMMU_LLAMA_SERVER`. *(2026-07-17)*
- [x] **LFM2.5-230M** as the CPU-tier hybrid zoo entry: shipped June 2026 with llama.cpp / GGUF support
      from day one — same `lfm2` architecture our loader + parity harness already cover, so this is a
      registry manifest entry + a run of the P3 quantized-reference leg (and a candidate to replace or
      join Qwen2.5-0.5B in the CPU decode budget row) —
      https://www.marktechpost.com/2026/06/27/liquid-ai-ships-lfm2-5-230m-with-llama-cpp-mlx-vllm-sglang-and-onnx-support-for-on-device-inference/
      *(2026-07-16 research)* *(2026-07-16, same run) Catalog entry + end-to-end proof landed
      (`real_hub.rs::hub_fetches_and_runs_lfm2_230m_on_cpu`): fetched through the registry (467 MB),
      checked-loads, greedy-decodes "2 + 2 equals 4." on the CPU backend. Its `config.json` uses the
      newer nested `rope_parameters` convention — `Lfm2Config` now reads both spellings (non-`default`
      rope types fail loudly; the 1.2B parity gate re-passed bit-identically after the change, max
      |Δlogprob| 1.4879674843625068e-2 unchanged).* *(2026-07-16, later run) **Parity gate PASSED**,
      both legs — the tier is now trusted. LiquidAI publishes `LFM2.5-230M-BF16.gguf`
      (`LiquidAI/LFM2.5-230M-GGUF`, 462 MB), the same-weights artifact the leg needed; `parity_lfm2.rs`
      is now written once and **parameterized by tier** (a `Tier` = tag + weights-dir env + BF16-GGUF
      env), so the 1.2B and 230M share both legs — the same hybrid loader covers both, which is the
      point of the architecture being config-driven. On the 4070 Ti SUPER: top-5 ids match the
      llama.cpp reference **exactly in order** and the 24-token greedy sequence is byte-identical, at
      max |Δlogprob| **3.244355439983515e-2**. That is ~2x the 1.2B's 1.49e-2 — expected, since a
      narrower model (1024 vs 2048 hidden) accumulates over fewer terms per dot product and so hides
      less of the reference's own per-dot bf16 activation rounding; it sits inside the existing 5e-2
      tolerance with ~1.5x headroom (tolerance unchanged, and the strict-order id match remains the
      real assert). Run legs with `MUMMU_LFM2_230M_DIR` + `MUMMU_LFM2_230M_BF16_GGUF`.*
- [x] **First MoE architecture** *(colibri parity)* — the zoo is dense-only and has so far deferred MoE
      ("Qwen3.6 35B-A3B … well past the single-card tier"), but colibri demonstrates the counter-thesis
      on our exact hardware class: frontier MoE (GLM-5.2 744B-A40B, Kimi K3 2.8T-A104B) on consumer
      boxes, because only the ~40B *active* params compute per token — dense parts (~10 GB at int4) stay
      resident and the 19k routed experts stream from NVMe on demand. The entry ticket for Mummu is a
      *small* MoE as a from-scratch Burn port — **OLMoE-7B-A1B** (colibri's own dev-tier model) or
      **Qwen3-30B-A3B** — new blocks: top-k router/gating + per-layer expert FFN banks, the GGUF
      `*.ffn_{gate,down,up}_exps` fused 3-D expert tensors in the name map, and the P7 parity gate vs
      llama.cpp like every port. This is the architecture prerequisite for the P6 expert-streaming item;
      resident-everything (no streaming) is a valid first cut for the small tier. *(2026-07-30 research)* —
      https://github.com/JustVugg/colibri
      *(2026-07-30, same run)* Port groundwork made concrete: **OLMoE-1B-7B** (64 experts, top-8 routing,
      1B active / 7B total) is the right first target — allenai publishes an official
      **`OLMoE-1B-7B-0125-Instruct-GGUF`** (Q2_K 2.6 GB … Q4_K_M **4.21 GB** … F16 13.8 GB, chat template
      in-metadata), llama.cpp runs it (arch `olmoe` — the parity reference leg is free), and the expert
      tensors follow the standard fused-3-D naming (`blk.N.ffn_{gate,down,up}_exps.weight`) our GGUF
      reader already parses structurally. Fit math for the first resident-everything cut: f32 dequant is
      28 GB — CPU-only (fits 128 GB RAM, slow) or **GpuF16 at ~14 GB** (marginal on the 16 GB card with
      ambient; the keep-quantized P9 leg or expert-CPU-offload makes it comfortable — expert tensors are
      the textbook offload candidates since only 8/64 fire per token) —
      https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct-GGUF ·
      https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide
      *(2026-08-03) **SHIPPED and parity-verified on the first run — the zoo is no longer dense-only.**
      New shared block `nn::SparseMoe`: a softmax top-k router (f32 island, HF `OlmoeSparseMoeBlock`
      math) over a **fused 3-D expert bank** `[experts, out, in]` — the exact row-major twin of ggml's
      `ffn_{gate,up,down}_exps`, so the 64 experts load as three tensors per layer rather than 192.
      Compute is **dense-mask**: every expert processes every token and the router's weight row (exactly
      zero off the top-k) scales the rest away — numerically identical to the sparse formulation, and it
      keeps the whole forward on-device with no data-dependent gather (see the measured follow-up below).
      `GqaAttention` gained OLMoE's q/k-norm placement — RMSNorm over the **whole projection** before the
      head split — inferred from the loaded gamma's width (`head_dim` = per-head, `n·head_dim` =
      projection), so the module shape is unchanged and every existing checkpoint loads identically.
      `models::olmoe` is config-driven off `olmoe.*` GGUF metadata (expert counts, either
      `feed_forward_length` spelling, `expert_weights_norm`), GGUF-only by design (HF ships the experts
      **unfused** as `mlp.experts.{i}.*` — a 64-way concat-on-import is the split item below); the
      tokenizer registry gained the `olmo` pre (stock GPT-2 regex + NFC). **Parity gate PASSED**
      (`parity_gguf.rs`, new `olmoe` leg — the harness is now generic over the backend): llama.cpp on the
      SAME allenai Q4_K_M file, top-5 first-forward ids match **exactly in order**
      (1992, 17833, 11202, 4943, 1394), the 24-token greedy sequence is **byte-identical**, max
      |Δlogprob| **3.687691131310693e-1** (inside the shared 7.5e-1 tolerance, right beside Qwen3's
      4.02e-1). Real-model proof (`real_olmoe.rs`): the registry spec fetched the 4.21 GB GGUF, the ONE
      file loaded 16 layers × 64 experts in 92 s and greedy-decoded "2 + 2 equals 4." at **1.15 s/token**
      on the CPU backend (~28 GB f32 resident — a 16 GB card is out of reach until the P9 keep-quantized
      leg), sanity spread 36.9; its GGUF-built tokenizer is **byte-identical** to the checkpoint's
      `tokenizer.json` across an 8-prompt battery. 202 unit tests; every prior gate re-passed unchanged
      after the attention refactor (Qwen3 GGUF parity bit-identical at 4.015608155114805e-1, Qwen2 both
      legs, template gate 10/10, budgets 104.4 ms / 13.2 tok/s GPU + 15.3 tok/s CPU). Run the legs with
      `MUMMU_OLMOE_GGUF_PATH` / `MUMMU_HUB_DEST` / `MUMMU_OLMOE_TOK_JSON`.*
- [ ] **MoE decode: make routed-expert compute actually pay** — the dense-mask forward computes all 64
      experts per token when only 8 are routed, so decode touches ~7B params instead of ~1B (baseline:
      **0.76 s/token** warm, `mummu-bench/tests/budget_moe.rs` — the number to beat). The obvious fix was
      tried and **measured a regression, so it was reverted, not shipped**: gathering the k routed
      expert slices with a device-side `select` off the router's own index tensor (no host sync, exact
      same math to summation-order rounding — a unit test confirmed the two paths agree to 1e-6) made real
      OLMoE decode **1.58 s/token vs the dense path's 1.15 s** end to end on burn-flex. The gather copies ~200 MB of
      expert weights per layer per token, and that copy costs more than the dense matmul it removes —
      i.e. `select` materializes where the dense path streams. Routes worth trying next, each gated on
      `bench/BASELINE.md` like this one was: (a) a fused dequant/gather matmul kernel that never
      materializes the gathered bank (the P9 keep-quantized kernel work is the natural host); (b) measure
      on the **GPU** backend, where the copy is far cheaper relative to compute — blocked until a MoE fits
      VRAM (P9 keep-quantized, or expert offload); (c) llama.cpp's own answer, `--n-cpu-moe`-style
      placement, which sidesteps the gather entirely by moving whole expert banks rather than slicing
      them. *(2026-08-03, measured this run.)*
      *(2026-08-20 research)* Two findings that sharpen the three routes above. (1) **Route (c) now has
      a measured implementation to copy rather than a slogan**: llama.cpp PR #25294 streams routed
      experts from disk behind a bounded per-layer device-side cache of expert *slabs* — top-k ids are
      remapped to cache slots on the CPU, a miss demand-loads asynchronously through an io thread pool,
      eviction is **decaying route hotness with an LRU tiebreak**, and reads use `O_DIRECT` so the page
      cache cannot thrash against a model far larger than RAM. Reported on GB10 for a ~254 GB model at a
      90-slot cache: **5.3x prefill / 2.4x decode vs `mmap`+`--n-cpu-moe`, at a 79 % cache hit rate**.
      That hit rate is the number that matters to us — it says a *small* resident expert set covers most
      tokens, which is exactly the premise the gather route needs to become cheap. It also ships
      **wave-partitioned prefill** (when a batch needs more experts than slots, run experts in waves of
      `(n_slots - n_expert_used)/2` and sum the masked outputs) — the trick that keeps a bounded cache
      from capping prompt length. (2) **Our gather regression is a batch-size regime, not a dead end**:
      llama.cpp only switches to the copy-experts-to-GPU path above a batch threshold, and ik_llama.cpp
      sets that threshold at `32 * total_experts / active_experts` — **256 tokens for OLMoE's 64/8**.
      Batch-1 decode is the worst possible point for a materializing gather, and prefill is the regime
      where it should win; the 2026-08-03 A/B measured only decode, so re-run it at prefill batch sizes
      before treating the gather as refuted. Both are gated on `bench/BASELINE.md` like the rest —
      https://github.com/ggml-org/llama.cpp/pull/25294 ·
      https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide
      *(2026-08-20, second run)* A fourth route, and the one that answers our specific regression:
      llama.cpp discussion #24528 proposes a **VRAM cache of hot experts with hybrid hit/miss
      execution** — the `MUL_MAT_ID` stays on the CPU, but thread 0 dispatches ONE batched matvec
      over the cached (hit) expert rows to the GPU while the remaining threads compute the miss rows
      exactly as they do today. The property that matters is that **a miss costs nothing extra**:
      performance degrades gracefully to the vanilla dense-CPU path instead of paying a gather that
      may not pay for itself, which is precisely how our 2026-08-03 A/B lost (1.58 s vs 1.15 s — the
      gather was unconditional). Eviction is LRU with an admission threshold, plus a "soft mode" that
      trades ~5-7 % of prefill to avoid evicting during decode; the cache **engages on decode only**.
      Reported: +25 % on GLM-5.1 754B, +7 % on Qwen3.5 397B, +28-46 % decode on 2x RTX 3090 — but a
      GTX 1080 Ti **regressed**, i.e. the win is a function of how much faster the GPU is than the CPU
      at the cached slice, which is exactly the hardware-dependence our own measurement found.
      Status: RFC, CUDA-only, unmerged. For us the shape translates directly — a bounded per-layer
      VRAM slab cache on the `Gpu` backend with the CPU dense path as the miss fallback — and it is
      the first route that does NOT require the whole MoE to fit VRAM first, so it unblocks route (b)
      without waiting on P9. Gate it on `bench/BASELINE.md` and the OLMoE 0.76 s/token warm number.
      — https://github.com/ggml-org/llama.cpp/discussions/24528 ·
      https://github.com/ggml-org/llama.cpp/issues/20757
- [x] **OLMoE from HF safetensors** — the port loads GGUF only because HF stores each expert separately
      (`model.layers.N.mlp.experts.{0..63}.{gate,up,down}_proj.weight`) while `MoeExperts` holds one fused
      `[experts, out, in]` tensor per projection. Needs a concat-on-import step (64 slices → one tensor,
      in expert order) in the safetensors path — mechanical, but it wants its own fixture (the bf16
      checkpoint is ~14 GB) and a byte-equality check against the GGUF-loaded weights. *(2026-08-03, split
      from the MoE item.)*
      *(2026-08-20) Shipped.* `crates/mummu/src/safetensors.rs` is a sharded-safetensors reader plus an
      **N:1 fusing rewriter** — the two things `burn-store` cannot do for us (it finds only a single
      `model.safetensors`, and its remapping is 1:1). It plans the whole output before reading one payload
      byte, so a group that is not exactly `count` members `0..count` is a loud `BadGroup` rather than a
      short bank that loads clean and computes wrong; members are ordered **numerically, not
      lexicographically** (`experts.10` sorts before `experts.2` as text — the trap this whole item exists
      to avoid). `olmoe::load_from_dir` maps `model.layers.N.mlp.experts.{0..63}.{gate,up,down}_proj.weight`
      onto the same fused `[experts, out, in]` banks the GGUF path lands on, then rides the ordinary
      `SafetensorsStore` + adapter-chain + `load_checked` pipeline.
      **REAL-WEIGHTS proof** (`tests/real_olmoe_safetensors.rs`, the 3-shard 13.84 GB bf16 checkpoint):
      the fuse builds `[64, 1024, 2048]` banks over 16 layers, checked-loads in **136.1 s** on the CPU
      backend, and emits a live distribution (sanity top 194, spread 20.6) that greedy-decodes 8 tokens.
      The check a mis-ordering could not survive: layer 5 / expert 37's `gate_proj`, read independently out
      of the raw shard bytes, is **bit-identical to slot 37 of the fused bank — 0 mismatches over 2 097 152
      values** (bf16 -> f32 is an exact 16-bit widening, so this is bit-equality, not a tolerance). Note the
      decode leg uses an arbitrary token probe, not a prompt, so it is a liveness check; the *weight*
      correctness evidence is the bit-exact one. 16 `safetensors` unit tests, incl. one pinning the
      safetensors fuse targets to the SAME module names the GGUF path renames to — both import paths must
      land on the same params or one of them is loading a different model.
      Two things the gate caught that the code did not survive first contact with:
      (a) `checkpoint_shards` planned straight off the index without checking the shards were **on disk**,
      so an interrupted download reported as a complete checkpoint, skipped the resume, and died later with
      a bare `os error 2` from inside the load — it now fails at planning time naming the missing shard, and
      says so explicitly when a `.part` sibling shows the download was interrupted;
      (b) the fuse materialized the payload in RAM **twice** (a `data` buffer, then a `blob` copy), and the
      second 13.8 GB allocation genuinely failed on this 128 GB box —
      `memory allocation of 13838346237 bytes failed`. `fuse_checkpoint_to_file` now streams header +
      payload straight to a temp file through one reusable per-part buffer (~4 MiB, one expert projection),
      so peak drops from ~42 GB (13.8 blob + ~28 model) to the model alone; a `FusedTemp` guard deletes the
      scratch file on every exit path, verified empty after the run. A unit test pins the two fuse paths as
      **byte-identical**, so the big-model path and the small-model path stay one importer.
- [x] **Give the GGUF dequant path the same streaming sink the safetensors fuse just got** — found while
      fixing the OLMoE safetensors OOM (2026-08-20). `gguf::dequant_to_safetensors` has the identical
      double-buffer: it fills a `data` Vec of `total_f32_bytes`, then copies it into a second `blob` Vec of
      `8 + header + data`, so peak is **2x the dequantized f32 payload** before the model is even built.
      For OLMoE-1B-7B that is ~28 GB f32 → ~56 GB peak, which is most of why that model reads as
      "CPU-only, and only on a big box". `safetensors::fuse_into` is the template: plan, then stream header
      + payload into an `impl Write` through one bounded per-tensor buffer, with a `*_to_file` variant for
      `SafetensorsStore::from_file`. Prove it the same way — a unit test pinning the in-memory and
      to-file outputs as byte-identical, then re-run the `real_gguf` + `parity_gguf` legs unchanged.
      *(2026-08-20, second run) Shipped.* `dequant_to_safetensors` is now a thin wrapper over a shared
      `dequant_into<W: Write>`, joined by `dequant_to_safetensors_file`. The double-buffer is gone from
      BOTH forms: the in-memory one writes header-then-payload into a single `Vec` (peak 1x the payload,
      down from 2x), and the file one holds nothing but the current tensor's f32 values plus a fixed
      1 MiB staging buffer. What made streaming possible also made the path stricter — a new
      `plan_dequant` decides every name, shape and offset **before the first payload byte is read**, so
      an unmapped name, a reshape that changes the element count, or a rename collision now fails in
      milliseconds instead of after N tensors have been dequantized (a unit test proves this by
      pointing a tensor's payload past the end of the file and checking the MAP error still surfaces).
      `olmoe::load_from_gguf` takes the file variant, reusing the same `FusedTemp` guard the HF path
      uses, so the scratch file is deleted on every exit path.
      **REAL-WEIGHTS proof**, the OLMoE-1B-7B Q4_K_M leg against llama.cpp on the identical file:
      top-5 ids exact in order, the 24-token greedy sequence **byte-identical**, max |Δlogprob|
      3.687691131310693e-1 (tolerance 7.5e-1), 130.5 s. Measured peak while it ran: **26.5 GB private
      commit** — the model alone — against a 51.7 GB working set, i.e. ~25 GB of the load is now
      file-backed mmap rather than charged to commit, which is the binding limit on this box. The old
      shape was ~28 GB blob + ~28 GB model ≈ 56 GB of commit. Nothing else moved: qwen2 GGUF parity
      2.6614442413586614e-1 and qwen3 **4.015608155114805e-1, bit-identical to the recorded value**,
      both greedy sequences byte-identical; the four `real_qwen2_gguf_*` legs pass (44.1 s). 234 unit
      tests (+2). The stale "~60 GB free RAM" gates on the three OLMoE test legs are corrected to
      ~30 GB free commit + ~28 GB of scratch disk.
- [x] **`gguf.rs` still has `.expect()` on production paths** — `dequant_to_safetensors` and the metadata
      readers carry `usize::try_from(..).expect("bounded above")` (4 sites), the same pattern removed from
      `safetensors.rs` on 2026-08-20 in favour of a fallible `to_usize` that returns `OverBound`. Mechanical,
      and it keeps the no-panic-on-production-paths rule true across the whole import suite rather than in
      the newest module only.
      *(2026-08-20, second run) Done — eight sites, not four.* `gguf::to_usize` mirrors
      `safetensors::to_usize` exactly. The four header readers (`Reader::{string,value,read_file,
      tensor_table}`) go through it — `tensor_table` now converts its count once instead of twice —
      and so does `dequant_to_safetensors`' payload capacity. Three more were hiding outside the
      `try_from` pattern: the tensor-name JSON encode (now reports through `BadTensor`, so it names
      the offending index), and `dequantize`'s two block-geometry casts plus its F32 4-byte block
      cast, all mapped into the `String` error that function already returns. No behaviour change by
      construction — every replaced site was unreachable on a 64-bit target — proven by the four
      `real_qwen2_gguf_*` legs on the Q4_K_M checkpoint (header parse, dequant-vs-true-weights,
      tokenizer byte identity, GPU load + decode).
- [x] **Decide the dequant sink per model, not per code path** — `olmoe::load_from_gguf` takes the new
      streaming `dequant_to_safetensors_file`; qwen2 / qwen3 / lfm2 still take the in-memory
      `dequant_to_safetensors` (which is now 1x the payload rather than 2x, so they already got half the
      win for free). That split is a guess, not a measurement: the file variant trades a spike in commit
      for an f32 write plus mmap-back, which should LOSE on a 1-2 GB model and win on anything that is a
      meaningful fraction of RAM. Measure GGUF load wall-clock both ways on Qwen2.5-1.5B (~2.2 GB f32)
      and Qwen3-0.6B, then either pick per-model or — better — pick automatically from
      `total_f32_bytes` against the device inventory's free-RAM figure (P6 already probes it), so a
      consumer never has to know. *(2026-08-20, discovered shipping the streaming sink.)*
      *(2026-08-21) Measured, and the guess was wrong.* A/B on Qwen2.5-1.5B Q4_K_M (~6 GB of f32),
      the same `real_qwen2_gguf_loads_and_decodes_on_gpu` load each way: **in-memory 33.90 / 35.93 /
      42.15 s vs scratch file 36.58 / 37.14 s** — the scratch numbers sit *inside* the in-memory run's
      own spread, so at any payload of real size the disk round-trip is free and the doubled peak was
      being paid for nothing. (n=2 on the scratch side: a concurrent routine held the shared cargo
      target's lock and the third run never started.) `burn-store` is why it is free —
      `SafetensorsStore::from_file` **mmaps** and materializes tensors lazily
      (`safetensors_to_snapshots_lazy_file`), so the read-back is page faults during a load that was
      already reading, not a second pass. So the choice is automatic, not per model:
      `import::DequantSink` is `Memory` / `Scratch` / `Auto`, `Auto` keeps a payload in RAM only up to
      `MAX_IN_MEMORY_DEQUANT_BYTES` (512 MiB — a ~1 GiB peak, free anywhere) and streams everything
      else, and all four ports pass `Auto` so no consumer sees the knob. Proof the new default changed
      nothing: qwen2 and qwen3 now take the scratch path they did not take before and their parity
      numbers are **bit-identical** to the recorded ones (2.6614442413586614e-1 / 4.015608155114805e-1),
      greedy sequences byte-identical, no scratch files left behind. Note the RAM-aware variant this
      item floated (compare against the device inventory's free-RAM figure) is deliberately NOT built:
      once streaming is free, a size threshold answers the question and a planner dependency would be
      complexity bought for nothing.
- [x] **`FusedTemp` is import-suite machinery living in one model file** — `models/olmoe.rs` owns the
      scratch-file guard (create beside the weights, unique per process + counter, `Drop`-delete on
      every exit path), and it now serves BOTH import paths. Any other model large enough to want a
      streaming sink has to reach into `olmoe` or copy it. Promote it to `import.rs` when a second
      model needs it — not before, since a one-caller abstraction moved early is just churn.
      *(2026-08-20.)* *(2026-08-21) The second caller arrived the same night* — `import::gguf_store`
      needs the guard for every port, not just OLMoE's — so it is now `import::ScratchFile`, with a
      unit test pinning the two properties that matter (two live guards never name the same file;
      `Drop` deletes on every exit path).
- [ ] **The f32 scratch file is a symptom, not the design** — both streaming sinks exist because
      `SafetensorsStore` is the only checked-load pipeline we have, so every import must first become
      f32 safetensors on disk. A GGUF **keep-quantized** load (P9) would skip the temp file entirely:
      no ~28 GB write, no 4x dtype widening, and the 1B-7B would stop being CPU-only. Worth stating
      here so the scratch-file machinery is understood as a bridge with a known end, not a fixture.
      *(2026-08-20.)*
- [ ] **Qwen3.5 hybrid (`qwen35`) architecture port** — split from the Qwen3.5-tier item when the
      2026-07-30 header probe showed Qwen3.5-4B/9B are a hybrid **linear-attention/SSM + periodic
      full-attention** arch (`qwen35.ssm.*` metadata: conv_kernel/state_size/group_count/time_step_rank/
      inner_size; `full_attention_interval`; `rope.dimension_sections` partial-rotary), not qwen3 dense.
      A port needs: the SSM/gated-delta block (recurrent state cache — the LFM2 conv-state machinery
      generalizes), the interval-scheduled full-attention layers, `qwen35.*` config + GGUF name maps, and
      the P7 parity gate vs llama.cpp (which runs it — the GGUFs ship working). Worth it when picked up:
      the 4B/9B are 2026's local-FC sweet spot (BFCL 9B 66.1%) and the 4B ships an `mmproj` vision
      projector (P11 candidate). *(2026-07-30 research)* —
      https://huggingface.co/unsloth/Qwen3.5-4B-GGUF
      *(2026-08-20 research)* The mechanism has a name and a ratio, which makes the port scopeable:
      the linear-attention half is **Gated DeltaNet**, interleaved with periodic full-attention at
      roughly **75 % linear / 25 % full** — so `full_attention_interval` is expected to read 4, and the
      recurrent-state cache (the LFM2 conv-state machinery generalizes) carries three quarters of the
      layers. Two adjacent facts: the family ships **MTP** variants as their own GGUF repos
      (`unsloth/Qwen3.5-{2B,9B,35B-A3B}-MTP-GGUF`), which is the concrete draft-model artifact the P5
      speculative-decoding item needs rather than a hypothetical; and llama.cpp renamed the flag
      `--spec-type mtp` to `--spec-type draft-mtp` (2026-05-13), worth knowing before wiring a
      reference leg against it. —
      https://huggingface.co/unsloth/Qwen3.5-9B-MTP-GGUF · https://sebastianraschka.com/llm-architecture-gallery/hybrid-attention/
      *(2026-08-21 ground truth)* Read straight from a real header (`gguf-info`, new `examples/`
      tool, on `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_S` — arch string is literally **`qwen35`** even for
      the 3.8 generation): `full_attention_interval = 4` **confirmed**; the linear-attention side is
      keyed as `ssm.*` (conv_kernel 4, inner_size 6144, state_size 128, group_count 16,
      time_step_rank 48); `nextn_predict_layers = 1` rides IN the main GGUF as an unused 66th block
      (`blk.65` `nextn.*` tensors — llama.cpp logs them "unused", so a port can skip them);
      `rope.dimension_sections = [11, 11, 10, 0]` (MRoPE — the family is multimodal, mmproj sidecar);
      head_count 24 / kv 4, key/value_length 256, embedding 5120, ffn 17408, 65 blocks, vocab
      248320, ctx 262144. Fit reality for the 27B: mummu's dequant-to-f32 path would need ~109 GB —
      the P9 keep-quantized item is a hard prerequisite, not an optimization, for this tier.
      Its UD quant mixes seven formats: Q4_K/Q5_K/Q6_K/Q8_0/Q3_K (have), **IQ4_XS** (shipped
      2026-08-21, 172 of 866 tensors), and small counts of IQ4_NL/IQ3_S/IQ3_XXS/IQ2_XS/IQ2_S
      (29 tensors, still missing — the IQ codebook-grid family).
      *(2026-08-21, later the same day)* **The full IQ family shipped** (IQ4_NL + the
      codebook-grid four, tables regex-extracted verbatim from `ggml-common.h` into
      `gguf_iq_grids.rs`, per-format hand-computed block tests) — `gguf-verify` (new example)
      streams **all 866 tensors of the 27B UD file to finite, plausible f32**. And the
      **`models::qwen35` port landed**: Gated DeltaNet (sequential recurrence, rolling conv
      state) + gated attention with partial RoPE every `full_attention_interval`-th layer,
      GGUF-only import (NextN/MTP tensors explicitly skipped via the new `GgufMap::Skip`),
      catalog entries `qwen3.5-2b` (BF16, the parity build), `qwen3.5-2b-q8`, and
      `qwen3.8-27b-ud-q4ks` (loud-size-error until P9). Toy cache-equivalence tests green;
      the llama.cpp parity gate is `tests/parity_qwen35.rs` (`MUMMU_QWEN35_GGUF` +
      `MUMMU_LLAMA_SERVER`; the local Ollama 0.32.15's bundled llama-server speaks qwen35).
- [ ] **RoPE scaling + sliding-window attention** *(mistral.rs parity)* — the shared blocks compute plain
      RoPE and a full causal mask; there is no `rope_scaling` handling (YaRN / linear / dynamic-NTK /
      `rope_type: llama3`) and no sliding-window mask anywhere in `nn` (grep across `src` is clean).
      Every current zoo model is fine — their configs ship `rope_scaling: null` and no window inside
      native context, which is why this never bit — but it is the **silent-wrong-answer** kind of gap:
      a checkpoint that *does* carry `rope_scaling` (Qwen2.5 past 32k via YaRN) or `sliding_window`
      (Mistral-family; Gemma 2/3 alternating layers) would load cleanly and degrade numerically far
      beyond the parity probes' short prompts. Two halves, the cheap one first: (1) **parse the fields
      and reject unsupported modes loudly** — that alone converts silent degradation into an error
      naming the mode; (2) implement scaled `rope_tables` + a windowed mask as config-driven variants
      of the shared blocks, no new module shapes. Gate: a *long-context* parity leg vs llama.cpp on a
      rope-scaled checkpoint — the short-prompt gates cannot see this by construction. *(2026-08-21,
      mistral.rs scan.)*
- [ ] **Llama-family decoder port (`llama`)** *(mistral.rs parity)* — the loader that multiplies
      checkpoint coverage most per unit of new surface: Llama 2/3.x and the wide Mistral/TinyLlama-style
      fine-tune space share one architecture shape, and it is strictly a subset of blocks Mummu already
      has (GQA without q/k-norm + SwiGLU + RoPE + RmsNorm + tied-or-untied head — expected zero new `nn`
      code; config + key-remap + chat renderer + parity gate). The zoo grows on consumer demand, not
      breadth-for-breadth — recorded here because when the next port is picked, this one buys the most,
      and Llama 3.x is the natural `rope_type: llama3` test vector for the scaling item above.
      *(2026-08-21, mistral.rs scan.)*
- [ ] **Qwen3-Embedding on the existing qwen3 port** *(mistral.rs parity)* — mistral.rs serves
      `Qwen/Qwen3-Embedding-0.6B` through its ordinary Qwen3 loader; the checkpoint IS the qwen3 dense
      arch Mummu already runs and parity-verifies, used as an embedder: last-token (EOS) pooling over
      the final hidden state instead of MiniLM's masked mean, an instruction-prefixed query format,
      L2-normalize. A modern 32k-context multilingual embedder for Nanna's memory `embed_fn` at the
      cost of a pooling fn + a fixture; MiniLM stays as the tiny tier. Gate: cosine parity vs the HF
      reference, same discipline as `real_minilm`. *(2026-08-21, mistral.rs scan.)* —
      https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- [x] A `Model` trait so new architectures (Hermes-class function-callers, Gemma, Qwen3, …) slot in.
      *(2026-07-10) `models::CausalLm<B>` — associated `Cache` type; a port supplies `new_cache` /
      `forward` / `is_eos` and inherits `generate` / `greedy_generate` / `first_token` from the shared
      driver (static dispatch, single code path). Both LLMs now implement it; the parity gate re-ran
      green through the trait, and the real-inference suite shares one `ModelSlot` (4 GPU tests, one
      3.1 GB load — found and fixed a 2-models-in-VRAM blowup in the old per-test loads).*

### P3 — Model import suite (any source / any format → a running model)
The subsystem that turns "a model on HuggingFace or on disk" into a loaded, parity-checked Mummu model.
**Data-driven** — adding a model is a manifest entry, not new code. All import is Burn-native (`burn-store` /
`burn-import`).
- [x] **Sources** — HuggingFace Hub (repo id + revision), local paths, and a bundled resources dir (checked
      first). Streaming download into a per-user cache: resumable (`.part`), integrity-checked, and
      **sharded-checkpoint aware** (read `*.index.json`, fetch + merge shards). *(2026-07-10) `mummu::hub`
      (ureq 3, https-only): `fetch_file` streams through `.part` with HTTP-Range resume + Content-Length
      verification + cache-first; `fetch_model` pulls config/tokenizer/weights with the
      `model.safetensors.index.json` shard fallback; `Progress` callback per chunk (feeds P8). Real-network
      proof: all-MiniLM (90.8 MB) downloaded → checked-load → unit-norm embedding; a half-seeded `.part`
      resumed at byte 249,507/466,247 and finished byte-identical. Local paths are already first-class
      (`load_from_dir`); bundled-resources-dir precedence is app wiring.*
- [x] Stronger download integrity: verify the Hub's LFS sha256 (`X-Linked-ETag`) instead of length-only;
      re-verify on cache hits behind a flag. *(2026-07-11) Every download stream-hashes (sha2, SHA-NI)
      against the sha256 a redirect-stopped HEAD reads from `X-Linked-ETag`; resumes fold the `.part`
      prefix into the hash; a mismatched `.part` is deleted, never resumed. `FetchOptions::verify_cached`
      re-hashes cache hits and self-heals once (delete + verified refetch). Real-network proof on the
      90.8 MB MiniLM weights: a flipped byte mid-file (invisible to the length check) was caught and
      healed; a 45.4 MB-seeded resume re-verified whole. Non-LFS files (no announced sha256) stay
      length-verified.*
- [x] **safetensors** *(ex-laurelane)* — `burn-store` `SafetensorsStore` + `PyTorchToBurnAdapter`; the primary path.
      *(2026-07-09) `import::{CastFloatAdapter, load_checked}`: bf16→backend-float cast + fail-loud load;
      proven by loading the real 3.1 GB Qwen2.5-1.5B and 2.3 GB LFM2.5 checkpoints with zero missing keys.*
- [x] **PyTorch state dicts** (`.pth` / `pytorch_model*.bin`) — for models not shipped as safetensors.
      *(2026-07-11)* `burn-store`'s `PytorchStore` wired through the shared checked-load path:
      `import::weights_file` picks `model.safetensors` first, falls back to `pytorch_model.bin`;
      `load_checked` is now generic over any `ModuleStore`; MiniLM loads either format through one
      remap table. REAL-WEIGHTS proof (`tests/real_pytorch.rs`): the Hub's actual MiniLM
      `pytorch_model.bin` embeds **byte-identically** (max |Δ| = 0) to the safetensors copy of the same
      weights. Remaining follow-ups: sharded `.bin` indexes, a bf16-cast on this path (PytorchStore has
      no adapter chaining; `.bin`-era checkpoints are f32), decoder loaders adopt `weights_file` when a
      real `.pth` decoder checkpoint exists to verify against, and `hub::fetch_model` learning the
      `pytorch_model.bin` fallback. *(2026-07-30)* The fetch-fallback follow-up is **deprioritized after a
      fixture hunt**: the Hub's safetensors-conversion backfill has made "has `tokenizer.json` + only
      `pytorch_model.bin`" repos effectively extinct (every candidate checked — tiny-gpt2,
      hf-internal-testing, cross-encoder — either got safetensors or predates fast tokenizers and lacks
      `tokenizer.json`, which `fetch_model` requires first). A real proof would need relaxing the
      tokenizer.json contract too — do it only if a consumer actually hits such a repo.
- [x] **GGUF** (llama.cpp) — parse the GGUF container (metadata KV + tensor table), map tensors to modules,
      and **dequantize** Q4/Q5/Q8/K-quant blocks into Burn tensors (or hand keep-quantized to P9). GGUF is how
      most small models are distributed — this makes the whole ecosystem importable. *(2026-07-10 research)*
      K-quant superblocks are 256 values: Q4_K = fp16 d + fp16 dmin + 12 B of 6-bit sub-scales/mins + 128 B
      of 4-bit q (144 B total), `x = d·scale_i·q − dmin·min_i`; Q6_K = 128 B low-4 + 64 B high-2 + 16×i8
      sub-scales + fp16 d (210 B). Rust references: llama.cpp ggml-quants + the `rage-quant` crate
      (Q8_0/Q4_K/Q6_K dequant + SIMD dot) — https://haroldbenoit.com/notes/ml/llms/quantization/llama.cpp/k-quants-implementation ·
      https://crates.io/crates/rage-quant
      *(2026-07-11 research)* `pmetal-gguf` 0.5 (May 2026, MIT/Apache-2.0, standalone — no candle/burn
      deps) is the most complete Rust GGUF implementation yet: read/write + dequant for the K-quants
      (Q2K–Q8K) AND IQ-quants, SIMD-optimized, importance-matrix support, HF-compatible config
      generation — evaluate as dependency-or-reference before hand-porting ggml-quants —
      https://docs.rs/pmetal-gguf/latest/pmetal_gguf/
      *(2026-07-12) **Container reader shipped** (`mummu::gguf`, no new deps): magic/version (v2/v3 LE),
      typed+bounded metadata KVs (strings ≤ 1 MiB, arrays ≤ 4M, nesting ≤ 2), tensor table with
      per-entry validation (known dtype, aligned offset, whole blocks, unique names), K-quant block
      layouts recorded (`block_size`/`bytes_per_block` for Q4_0…Q8_K), fail-loud error taxonomy.
      REAL-FILE proof (`tests/real_gguf.rs`): the local Qwen2.5-1.5B **Q4_K_M** parses — v3, 26 kvs,
      339 tensors, Q4_K `token_embd [1536, 151936]`, 198 K-quant tensors, ~1.04 GiB payload located
      (3B file cross-checked: 435 tensors). 7 unit tests over a synthetic-bytes builder.*
      *(2026-07-12, same run) **Dequant shipped** for the Q4_K_M set — F32/F16/BF16, Q8_0, and the
      K-quant superblocks **Q4_K/Q6_K** (exact ports of ggml-quants' reference dequantizers, incl. the
      packed 6-bit scale/min encoding) + `GgufFile::read_tensor_f32`. Proof against the model's TRUE
      weights (`real_gguf.rs` — the same checkpoint exists locally as bf16 safetensors AND Q4_K_M GGUF):
      the GGUF's F32 `output_norm.weight` is **bit-exact** vs the bf16 originals, and dequantized Q4_K
      embedding rows hit cosine **0.9975** vs truth (garbage layout ⇒ ≈ 0) — 5 hand-computed-block unit
      tests (121 total). NEXT slice: remaining dequants (Q4_0/Q5/Q2_K/Q3_K/Q5_K), GGUF→model load
      (name remap + ggml dim-order transpose), tokenizer-from-GGUF-metadata.*
      *(2026-07-13) **GGUF→running model shipped.** Every storage dtype now dequantizes (added
      Q4_0/Q4_1/Q5_0/Q5_1 + K-quants Q2_K/Q3_K/Q5_K, exact ggml ports, hand-computed block tests —
      133 unit tests total; *2026-08-21:* + **IQ4_XS**, the workhorse of unsloth's UD dynamic quants
      — the codebook-grid IQ family IQ4_NL/IQ3_S/IQ3_XXS/IQ2_XS/IQ2_S is still unimplemented and is
      what blocks parsing a UD file end-to-end, see the qwen35 note in P2); `GgufFile::dequant_to_safetensors` bridges a GGUF onto the SAME checked-load
      pipeline as safetensors (dims reversed = row-major HF layout, unmapped tensor names are loud
      errors); `Qwen2Config::from_gguf` reads hyperparameters from `qwen2.*` metadata (vocab from the
      embedding tensor, EOS from tokenizer metadata); `qwen2::load_from_gguf` = one file → running model.
      The Qwen2 module gained an optional **untied lm_head** (llama.cpp GGUFs materialize the tied head
      as a separate higher-precision `output.weight` — the real Q4_K_M carries it as Q6_K; also unlocks
      untied safetensors like Qwen2-7B). REAL-GPU proof (`real_gguf.rs`): the Q4_K_M file alone
      greedy-decodes "2+2 equals 4.", first-token top-1 identical to the bf16 build, top-5 overlap 4/5,
      logit cosine 0.977 (28 layers of Q4_K drift; a layout bug reads ≈ 0). Parity gate re-passed
      byte-identically after the lm_head change (max |Δlogit| 2.670e-5, Ollama greedy exact); f16 +
      budget gates green (GPU 107.4 ms / 13.6 tok/s, CPU 14.8 tok/s).*
- [x] **Tokenizer-from-GGUF metadata** — build the HF `tokenizers` pipeline from `tokenizer.ggml.*`
      (tokens, merges, token types, BPE pre-tokenizer regex) so a GGUF needs no sibling
      `tokenizer.json`; byte-verify token ids against the HF tokenizer of the same checkpoint.
      *(2026-07-13, same run) Shipped (`mummu::tokenizer::tokenizer_from_gguf`): NFC → Split(the
      per-family `tokenizer.ggml.pre` regex, llama.cpp-style registry — unknown ids are loud errors) →
      ByteLevel → BPE; token id = array index; CONTROL(3)/USER_DEFINED(4) types become special/plain
      added tokens with post-build id verification, UNUSED(5) `[PADn]` entries skipped. REAL-FILE
      proof: **byte-identical ids vs the checkpoint's `tokenizer.json` on an 8-prompt battery**
      (ChatML + specials, unicode/CJK/emoji, whitespace runs, contractions, empty) and identical
      decodes; the end-to-end GPU test now runs tokenizer + config + weights from the ONE .gguf file.
      Only `gpt2`-model/`qwen2`-pre is registered so far — new families add a regex entry.*
- [x] **LFM2 GGUF name map** — extend `load_from_gguf` to the LFM2/LFM2.5 hybrid (llama.cpp `lfm2`
      arch: `shortconv.*` tensor names, `lfm2.*` metadata keys) once a same-weights GGUF is validated.
      *(2026-07-13, same run) Shipped (`lfm2::load_from_gguf`): `Lfm2Config::from_gguf` derives
      `layer_types` from llama.cpp's per-layer `head_count_kv` array (0 = conv, nonzero = attention;
      i32 in real files), `feed_forward_length` arrives pre-adjusted; the name map covers the hybrid's
      per-head q/k norms + `shortconv.{conv,in_proj,out_proj}`, with the depthwise conv kernel as the
      one shape special-case (`GgufMap::Reshape` — llama.cpp squeezes `[C,1,K]` to ggml `[K,C]`, same
      bytes). Tokenizer registry gained the `lfm2` pre (digits-≤3 regex, no NFC, BOS post-processor
      from `add_bos_token`; low-id added tokens ride BPE-vocab id reuse). REAL-FILE proof against the
      official LiquidAI Q4_K_M (697 MB, downloaded this run) vs the local bf16 safetensors of the same
      checkpoint: 5 F32 tensors incl. both conv kernels **bit-exact**; tokenizer byte-identical on a
      6-prompt × 2-mode battery; REAL-GPU end-to-end — the one file greedy-decodes "2 + 2 equals 4.",
      first-token top-1 identical to the bf16 build, logit cosine **0.9914**. (Still not the P2/P7
      strict parity gate — that needs the llama.cpp same-quant reference leg.) 138 unit tests; parity
      (2.670e-5, byte-identical) + budget gates re-passed.*
- [x] **Quantized-reference parity leg for GGUF loads** — the end-to-end test compares against the bf16
      build (quantization drift bounded, not exact); a strict leg needs llama.cpp itself running the
      SAME quantized file (`llama-server` raw `/completion`, `n_probs` logprobs — see the P7 LFM2.5
      reference item's caveats) to assert our dequant matches ggml's compute path token-for-token.
      *(2026-07-16) Shipped (`tests/parity_gguf.rs`, on the P7 `llama_ref` harness): llama.cpp runs the
      SAME local Q4_K_M files our loader loads — 23/24-token greedy sequences **byte-identical** for
      BOTH Qwen2.5-1.5B and LFM2.5-1.2B; top-3 first-forward ids exact in order, top-5 overlap ≥ 4/5,
      max |Δlogprob| 2.7e-1 (the reference's own Q8_K *activation* quantization in its integer Q4_K
      kernels — an order above the BF16 leg's 1.5e-2; our f32 path doesn't quantize activations).*
- [x] **`fetch_model` fetches the sibling files the import gates read** — found while installing the OLMoE
      safetensors checkpoint through the registry (2026-08-20). `hub::fetch_model_with` fetched exactly
      `config.json`, `tokenizer.json`, and the weights, so a checkpoint installed through Mummu's OWN
      registry arrived with no `tokenizer_config.json` — and the 2026-07-19/20/21 gates all fail **open**:
      `validate_checkpoint_dir` returns `Ok(None)`, the EOS-agreement + added-token-id + tool-call-convention
      checks silently no-op, and the `tokenizer_config` the loaders surface is always `None`. The gates
      appeared to work only because every locally-cached fixture had been populated by hand. Fixed: both
      `tokenizer_config.json` and `chat_template.jinja` (the standalone-template checkpoints of the
      2026-07-20 item) are now fetched as **optional** files — one HEAD each, and only a 404 counts as
      absent, so a 403 on a gated repo or a 5xx still reaches the real fetch and is reported rather than
      swallowed as "the repo just doesn't ship it". `config.json` / `tokenizer.json` stay required. They are
      fetched BEFORE the weights because the single-file branch returns early — placing them after it would
      have fetched them for sharded checkpoints only, which is exactly the kind of half-fix this item exists
      to avoid. Proof (`tests/real_hub.rs::a_registry_install_arrives_with_the_files_the_import_gates_read`):
      a catalog model installed into a **clean** dir — no hand-populated fixture can satisfy it — now has
      `tokenizer_config.json` on disk AND `validate_checkpoint_dir` returns `Some`, i.e. the gate has
      something to check instead of quietly passing. *(2026-08-20)*
- [ ] **GPTQ / AWQ** (HF safetensors) — import the calibration-quantized int4/int8 layouts most "quantized on
      the Hub" models ship as (a `.safetensors` payload + a quant config), dequant or keep-quant into Burn.
      *(2026-07-13 research)* Both are quantization *algorithms*, not formats — the artifact is ordinary
      safetensors shards (packed int4 `qweight`/`qzeros`/`scales`, group size 128 is the de-facto
      standard) plus `quantization_config` in `config.json`; vLLM's **compressed-tensors** is the
      emerging unified on-disk convention to target (one reader covers GPTQ/AWQ/INT8/FP8 exports) —
      https://github.com/vllm-project/compressed-tensors ·
      https://www.digitalapplied.com/blog/gguf-vs-awq-vs-gptq-vs-mlx-llm-quantization-formats-2026
- [ ] **Architecture auto-detection (any checkpoint by path alone)** *(mistral.rs parity)* — mistral.rs
      dispatches any checkpoint off `config.json`'s `architectures[0]` (HF) or `general.architecture`
      (GGUF) with `--arch` as a rarely-needed override; Mummu *parses* both today but only *dispatches*
      through the curated registry catalog, so a Qwen2/Qwen3/LFM2/OLMoE/MiniLM checkpoint that is not
      one of the 12 catalog entries needs the caller to know its architecture. The North-Star reading
      ("one pathway, no config, it just works") makes this the import suite's missing top:
      `registry::detect(dir_or_gguf) -> Architecture` off the checkpoint's own metadata, so any
      supported-arch checkpoint loads by path alone, and an unknown architecture is a loud, readable
      error naming the arch string it found — never a guess. The catalog keeps its job (a curated,
      parity-verified set with known-good URLs); it just stops being the only door. *(2026-08-21,
      mistral.rs scan.)*
- [ ] **ONNX** (optional) — `burn-import` ONNX→Burn for models distributed as ONNX graphs.
- [x] **Dtype handling** — a `CastFloatAdapter` (bf16→f32/f16); quantized→dequant on import; keep-quantized
      handed to P9. *(2026-07-17)* All three legs are in place and proven across the zoo: `CastFloatAdapter`
      (`import.rs`) casts bf16 (HF's shipping dtype) to the backend's float on load — f32 on the `Gpu`/`Cpu`
      aliases, **f16** on `GpuF16` — chained after `PyTorchToBurnAdapter` on every safetensors AND GGUF
      load path; quantized→dequant-on-import is the `load_from_gguf` pipeline (every storage dtype → f32
      via `dequant_to_safetensors`); keep-quantized-in-VRAM stays a P9 item (the actual fit lever). Newly
      verified on the Qwen3 arch: it loads + decodes coherently on `GpuF16` with the same f32-softmax
      attention island Qwen2/LFM2 use (`tests/real_qwen3.rs::real_qwen3_decodes_coherently_in_f16` — no
      overflow to NaN, sanity-smoke spread 49.4 identical to f32, "2 plus 2 equals 4").
- [x] **Weight-name remapping + checked load** — per-architecture key-remap tables (checkpoint naming →
      Mummu module names); **fail loudly** on missing/unexpected keys with a readable diff, never silently zero-init.
      *(2026-07-09) Remap tables live in each model's `load_from_dir` (Qwen2: strip `model.` + RmsNorm→gamma;
      LFM2: + `out_proj`/`q_layernorm`/`w1-w3` onto the shared blocks); `load_checked` errors carry the
      store's readable report. A declarative table registry lands with the manifest item below.*
- [x] **Config import** — parse `config.json` → model hyperparameters (layers, hidden, heads, kv-heads,
      rope-theta, vocab, tie-word-embeddings, …) so a model is **config-driven**, not hardcoded per checkpoint.
      *(2026-07-09) Per-architecture serde configs with validation (`Qwen2Config`, `Lfm2Config` incl.
      `layer_types` + auto-adjusted ff_dim); both real checkpoints parse and drive the build.*
- [x] **Tokenizer + chat-template import** — HF `tokenizer.json` (fast), SentencePiece `tokenizer.model`, BPE
      merges/vocab; special-tokens map + the chat template from `tokenizer_config.json`. *(2026-07-18)*
      **`tokenizer_config.json` import shipped** (`mummu::tok_config::TokenizerConfig`, no new deps): parses
      the *conventions* HF keeps beside `tokenizer.json` — `add_bos_token`/`add_eos_token`, the BOS/EOS/PAD/UNK
      special-token slots (each `null` | bare string | `{content,special}` object, id resolved from
      `added_tokens_decoder`), the full id-sorted `added_tokens_decoder` map, `model_max_length`, and the raw
      Jinja `chat_template` (string, or the `default`/first entry of a `[{name,template}]` list). Total +
      bounded (4 MiB file cap, 1M added-token cap; malformed key / missing content / duplicate numeric id /
      non-object are loud `ImportError::Parse`, never a panic), `eos_id`/`bos_id`/`pad_id` accessors. It does
      **not** render Jinja — prompt wrapping stays the byte-verified `chat` renderers; the imported template
      is the check-against + tool-style-detect source, the imported ids are a config↔tokenizer cross-check.
      `tool_call_convention()` detects the template's tool-call style from its marker tokens
      (`ToolCallConvention::{Hermes, Lfm}` — LFM's unambiguous `<|tool_call_start|>` checked first, else
      Hermes' `<tool_call>`, else `None`), so an app picks the matching `chat::render_with_tools` style from
      the checkpoint instead of hardcoding it. 10 unit tests + a REAL-FILE gate
      (`tests/real_tokenizer_config.rs`, cached qwen3-0.6b): EOS `<|im_end|>`→151645, PAD
      `<|endoftext|>`→151643, `add_bos_token` false, 4168 B ChatML template detected **Hermes**, and **all 26
      added-token ids agree byte-for-byte with `tokenizer.json`** (`Tokenizer::token_to_id`).
      *(2026-07-18, same run)* **Consistency validators shipped**: `check_ids_against(token_to_id)` promotes
      the config↔tokenizer cross-check to a first-class fn (every `added_tokens_decoder` id must equal the
      real tokenizer's id — resolved BOS/EOS/PAD/UNK slots are subsumed, each carries an id only because its
      content was found in the added map; returns a bounded `Vec<IdMismatch>`), and `check_eos_agrees(&[u32])`
      cross-checks `config.json`'s `eos_token_id` set against the resolved EOS id (catches the repackaging bug
      where the two files disagree on which token ends a turn). Both take plain data (closure / slice), so
      `tok_config` stays free of a models/tokenizer dependency; a loader just calls them. REAL-FILE proof on
      qwen3-0.6b: all 26 ids pass `check_ids_against`, and `config.json` eos 151645 agrees with the resolved
      `<|im_end|>`. Remaining on this item: SentencePiece `tokenizer.model` import, and *calling* these
      validators from `load_from_dir` (config-driven EOS + template-vs-renderer consistency) — split below.
      *(2026-08-21) Closed — every remaining piece it named is `[x]` below, and the whole surface was
      re-verified on real files this run rather than taken on the sub-items' word:*
      `real_tokenizer_config` (qwen3-0.6b config↔tokenizer ids agree), `real_spm` **both** legs
      (Unigram via flan-t5-small and BPE-type via tinyllama, ids byte-matching each checkpoint's own
      `tokenizer.json`), the **10-case template BYTE gate** (qwen2 / qwen3 / lfm2, plain + history +
      tools), and `imported_render` under `--features jinja-template` (the fallback renderer's output
      byte-identical to the family renderer at 142 / 748 / 324 B for qwen3, 157 / 379 B for lfm2, plus
      the no-family-renderer fallback at 57 B). Anything further on tokenizers is its own item, not
      this one.
- [ ] **The real-file gates panic on a missing env var instead of reporting a skip** — already recorded
      operationally for `parity_gguf` (its lfm2 leg has no local GGUF, so the whole binary reports
      FAILED even when every other leg passes); `imported_render` did the same thing this run, failing
      on an unset `MUMMU_LFM2_DIR` while both other legs were byte-identical. The tension is real and
      the current behaviour is the safer half of it — a gate that silently skips is a gate that does
      not exist, which is exactly how a parity suite rots. So the fix is NOT "skip quietly": make the
      missing-fixture case a distinct, *summarized* outcome — collect the unrunnable legs and print
      one "N legs skipped for missing fixtures: …" line, so the run summary distinguishes "no fixture"
      from "wrong answer" without ever letting the second hide inside the first. *(2026-08-21,
      discovered re-verifying the tokenizer gates.)*
- [x] **SentencePiece `tokenizer.model` import** — the `.model` proto tokenizer (Llama/Gemma/T5 family) that
      HF ships instead of a `tokenizer.json`; build the equivalent HF `tokenizers` pipeline (or convert), and
      byte-verify ids against a `tokenizer.json` of the same checkpoint where one exists. *(2026-07-18, split
      from the tokenizer-import item.)* *(2026-07-18 research)* Route: the HF `tokenizers` crate we already
      depend on carries a **Unigram** model (`tokenizers::models::unigram`) — a SentencePiece proto is a
      Unigram vocab + scores + a `precompiled_charsmap` normalizer, so importing it is (a) parse the protobuf
      (`google/sentencepiece`'s `ModelProto` — no serde; a tiny hand-rolled varint reader like `gguf.rs`, or
      the `prost`/`quick-protobuf` crates) into pieces+scores, (b) feed the charsmap through HF's
      **`spm_precompiled`** crate (purpose-built to use `sentencepiece`'s `precompiled_charsmap` inside
      `tokenizers`), (c) assemble a `Unigram` + `Precompiled` normalizer. `guillaume-be/rust-tokenizers` loads
      the same `.model` proto directly and is a reference. No SentencePiece checkpoint is cached locally yet,
      so this needs a fixture fetch (a small Gemma/Llama `.model` + its `tokenizer.json` for the byte-verify) —
      https://github.com/huggingface/spm_precompiled · https://github.com/guillaume-be/rust-tokenizers
      *(2026-07-30)* **Shipped exactly along the researched route, Unigram leg — and the byte gate passed
      first run.** `tokenizer::tokenizer_from_spm(path)`: a bounded hand-rolled protobuf wire reader (the
      `gguf.rs` approach — varint/length-delimited/fixed32, unknown fields skipped, 64 MiB file / 1M piece
      caps, truncation is a loud error) parses `ModelProto` pieces(+scores+types), `trainer_spec.model_type`
      /`unk_id`, and `normalizer_spec.{precompiled_charsmap, add_dummy_prefix, remove_extra_whitespaces}`;
      assembly mirrors HF's `convert_slow_tokenizer`: `Precompiled` charsmap (via the `spm_precompiled` the
      `tokenizers` dep already carries — **zero new dependencies**) + a regex `" {2,}"→" "` collapse →
      Metaspace(`▁`, Always/Never by `add_dummy_prefix`) as pre-tokenizer AND decoder → `Unigram(pieces,
      unk_id, byte_fallback = any BYTE piece)`; CONTROL/UNKNOWN pieces become special added tokens,
      USER_DEFINED plain, every added id verified post-build like the GGUF path. BPE-type protos (Llama-2
      family) are a loud not-yet-supported error (split below). Proof: 4 new unit tests over synthetic proto
      bytes (assembly + metaspace/collapse behavior, BPE/truncation/bad-unk rejection, unknown-field skip),
      and `tests/real_spm.rs` on the real **flan-t5-small** fixture (`spiece.model` vs its shipped
      `tokenizer.json`): a 10-prompt battery — whitespace runs, leading/trailing space, `™`/`½`/ligature/
      fullwidth (the charsmap leg), CJK+emoji, contractions, newlines, empty — is **byte-identical in ids
      AND decode round-trips**, and `<unk>`/`</s>`/`<pad>` resolve to the same ids. Tokens beyond the proto
      (T5's 100 `<extra_id_*>`) are sibling-metadata territory by design (`tokenizer_config.json` /
      `added_tokens_decoder`), documented on the fn.
- [x] **BPE-type SentencePiece protos** (Llama-2 family: `trainer_spec.model_type = BPE`) — the other half
      of the SPM import: same proto reader, but the pieces feed a BPE model (scores encode merge ranks)
      instead of Unigram; needs a byte-verify fixture with both files (TinyLlama ships `tokenizer.model` +
      `tokenizer.json`, non-gated). Currently a loud not-yet-supported error in `tokenizer_from_spm`.
      *(2026-07-30, split from the SentencePiece item.)* *(2026-07-30, same run) **Shipped, byte gate
      passed first run.** `tokenizer_from_spm` now dispatches on `model_type`: the BPE leg reconstructs
      the merge list from vocab + scores exactly as HF's `SentencePieceExtractor` does (every piece's
      valid splits, local `(id_l, id_r)` order, stable global sort by score DESCENDING — higher score =
      earlier merge), builds `BPE { fuse_unk, byte_fallback }`, with the Llama-family pipeline its own
      `tokenizer.json` pins: `Prepend(▁)` + `Replace(" "→"▁")` normalizers (driven by
      `add_dummy_prefix`/`escape_whitespaces`, now parsed), NO pre-tokenizer, and the
      `Replace(▁→" ")`/`ByteFallback`/`Fuse`/`Strip` decoder chain. Proof: a synthetic BPE toy proto
      whose merge chain must fire in score order (unit test, 185 total), and `tests/real_spm.rs` gains a
      TinyLlama-1.1B leg — the reconstructed 61 249-merge tokenizer is **byte-identical in ids and
      decode round-trips** to the checkpoint's shipped `tokenizer.json` across the 10-prompt battery,
      byte-fallback cases included; `<unk>`/`<s>`/`</s>` ids agree. Both SPM legs (Unigram + BPE) now
      cover the `.model`-shipping families end to end.*
- [x] **`tok_config` reads a standalone `chat_template.jinja`** — recent `transformers` `save_pretrained`
      (and the v5 tokenizer split) writes the chat template to a separate **`chat_template.jinja`** file in the
      tokenizer dir rather than the `chat_template` key of `tokenizer_config.json`; some checkpoints ship it
      *only* there (Gemma4 — transformers #45205). Mummu's `TokenizerConfig::from_dir` read the template from
      `tokenizer_config.json` alone, so for such a checkpoint `chat_template` was `None`, `tool_call_convention()`
      `None`, and the load-time convention check (the 2026-07-19 gate) silently no-op'd — a foreign-template
      mismatch went uncaught. *(2026-07-20)* **Fixed:** `from_dir` now falls back to reading
      `dir/chat_template.jinja` when the JSON key is absent (a present JSON key wins, never overridden;
      absent/non-file/empty → `None`; oversized or non-UTF-8 → a loud `ImportError::Parse`, bounded by the same
      4 MiB cap as the JSON). So `has_chat_template()`/`tool_call_convention()` — and thus the loader's
      convention gate — keep working for these checkpoints. 3 new unit tests (fallback picks up the file +
      detects its convention; JSON key wins over the file; a whitespace-only file is treated as absent) and a
      `tests/load_gate.rs` end-to-end case: a Qwen3 dir with an LFM-convention template in `chat_template.jinja`
      (no JSON key) is now rejected `Inconsistent` at load rather than passing silently. — 
      https://github.com/huggingface/transformers/issues/45205 ·
      https://huggingface.co/docs/transformers/chat_templating_writing
- [x] **Wire `tok_config` into `load_from_dir`** — have the safetensors loaders read `tokenizer_config.json`
      for the config-driven EOS/BOS ids (today each model hardcodes `EosIds`) and assert the imported
      `chat_template`'s markers are consistent with the model's byte-verified `chat` renderer, so a
      checkpoint whose template silently disagrees with our renderer fails loudly at load. *(2026-07-18,
      discovered building `tok_config`.)* The validators this needs now exist —
      `TokenizerConfig::check_eos_agrees` (config.json ↔ tokenizer_config EOS) and `check_ids_against` (vs the
      loaded tokenizer); what remains is the loader *calling* them (a behavior-affecting change — the loaders
      don't currently open the tokenizer, and a fail-loud EOS mismatch changes load semantics, so it wants a
      deliberate decision + a re-run of the real-model suite, not a drive-by).
      *(2026-07-19)* **Shipped, tokenizer-free half.** A fail-loud consistency gate now runs inside all three
      safetensors `load_from_dir`s (`qwen2`, `qwen3`, `lfm2`), right after `config.json` parses and **before**
      any weight bytes are read: `tok_config::validate_dir(dir, config_eos_ids, expected_convention)` reads the
      sibling `tokenizer_config.json` **if present** (a GGUF-derived / minimal dir has none → `Ok(None)`, no
      behavior change), a *present-but-malformed* file propagates its parse error, and a well-formed file that
      **disagrees** becomes a new loud `ImportError::Inconsistent`. Two checks, both needing no tokenizer:
      (a) `check_eos_agrees` — the resolved `tokenizer_config` EOS must be in `config.json`'s `eos_token_id`
      set (catches the repackaging bug where the two files name different turn-enders → the model never stops
      or stops wrong); (b) `TokenizerConfig::check_consistency` — if the imported template *declares* a
      tool-call convention (`tool_call_convention()` is `Some`), it must not contradict the family renderer's
      (`Hermes` for Qwen2/Qwen3, `Lfm` for LFM2.5) — catches e.g. an LFM template dropped into a Qwen dir; a
      tool-less base template declares nothing and is not forced to match. `EosIds::to_vec()` feeds the EOS
      set. Proof: 3 new `tok_config` unit tests + a `tests/load_gate.rs` that drives the **real**
      `qwen3::load_from_dir` on a temp dir (zero-byte `model.safetensors` + a mismatched
      `tokenizer_config.json`) and confirms it returns `Inconsistent` *before* touching weights, while an
      agreeing checkpoint clears the gate and fails only later on the empty weights (173 unit + 3 gate tests
      green); the real Qwen3-0.6B safetensors GPU load+decode re-passed unchanged with the gate live (the
      cached checkpoint's `tokenizer_config` EOS `<|im_end|>`→151645 agrees with `config.json` and its ChatML
      template is Hermes). What remains is split to the item below.
- [x] **Loaders open the tokenizer for `check_ids_against` + config-driven BOS** — the 2026-07-19 gate is
      tokenizer-free (it cross-checks `tokenizer_config.json` ↔ `config.json` only). The remaining half of the
      wiring needs the loader to actually *open* the HF `tokenizer.json`: run `check_ids_against(token_to_id)`
      (every added-token id vs the real tokenizer) at load, and *drive* the model's EOS/BOS ids from
      `tokenizer_config.json` instead of only cross-checking the hardcoded `EosIds`. That is a larger change —
      the loaders don't currently construct a `Tokenizer` (tokenization is caller-side by design), so it adds a
      tokenizer dependency to the load path and shifts where the source-of-truth EOS lives; wants its own
      deliberate decision + real-model re-run. *(2026-07-19, split from the wiring item above.)*
      *(2026-07-20)* **`check_ids_against` half shipped — the loaders now open the tokenizer.** New
      `tokenizer::validate_checkpoint_dir` wraps the tokenizer-free 2026-07-19 gate and, when a sibling
      `tokenizer.json` is present, loads it (the `tokenizers` crate the crate already depends on) and runs
      `TokenizerConfig::check_ids_against(|t| tok.token_to_id(t))`: every added-token id `tokenizer_config.json`
      declares must equal the real tokenizer's id, else a loud `ImportError::Inconsistent` naming the
      disagreeing tokens — fired *before* any weight bytes are read. All three safetensors loaders
      (`qwen2`/`qwen3`/`lfm2`) now call it in place of `tok_config::validate_dir`; `tok_config` stays
      tokenizer-free (the closure seam does the opening in `tokenizer.rs`). Both sibling files stay optional —
      a GGUF-derived dir (no `tokenizer.json`) skips the id check, the EOS/convention checks still run. Proof:
      `tests/load_gate.rs` gains a mismatch case (a real `tokenizer.json` built via the `tokenizers` BPE model
      declares `<|extra|>` at id 999 while the config says 6 → rejected, error names `tokenizer.json` + `999`)
      and an agree case (ids match → gate clears, load then fails on the empty weights); 5/5 load-gate tests
      green, and the real Qwen3-0.6B safetensors GPU load re-passed with the id check live (its 26 added-token
      ids all agree). **Remaining (kept `[ ]`): config-driven EOS/BOS** — driving the model's EOS/BOS *from*
      `tokenizer_config.json` rather than the hardcoded `EosIds` was deliberately NOT done here: `config.json`'s
      `eos_token_id` is already the source of truth and is now cross-checked to agree, and BOS/tokenization
      stays caller-side by design; surfacing the parsed `TokenizerConfig` on the `Loaded*` structs for a
      consumer to read is the intended shape, left to a dedicated decision.
      *(2026-07-21)* **Config-driven EOS/BOS shipped — the intended shape, exactly as scoped.**
      `validate_checkpoint_dir` already parsed and *returned* the sibling `TokenizerConfig`, but all three
      safetensors loaders discarded it; they now capture it and surface it as a new public
      `tokenizer_config: Option<TokenizerConfig>` field on `LoadedQwen2`/`LoadedQwen3`/`LoadedLfm2`. A consumer
      reads config-driven EOS/BOS/PAD straight off the loaded model (`m.tokenizer_config.as_ref().and_then(|c|
      c.eos_id())`, `.bos_id()`, `.add_bos_token`) instead of hardcoding. Deliberately *additive*: the model's
      internal `is_eos` still rides `config.json`'s `eos_token_id` (already the cross-checked source of truth per
      the 2026-07-19 gate), so decode/parity behavior is byte-unchanged; a debug-assert in each safetensors
      loader upholds the invariant that any surfaced EOS agrees with `config.json`. GGUF loads surface `None`
      (self-contained — that path reads no sibling `tokenizer_config.json`; EOS rides GGUF metadata). REAL-GPU
      proof (`tests/real_qwen3.rs`): the safetensors leg asserts the surfaced config's `eos_id()` == 151645
      (`<|im_end|>`) and agrees with `config.json`; the GGUF leg asserts `tokenizer_config.is_none()`. 176 unit
      tests + parity + budget gates unmoved (additive field, off every hot path).
- [x] **Evaluate `hf-chat-template` to render the imported `chat_template`** — Mummu's prompt wrapping is
      hardcoded, byte-verified `chat` renderers (one per family); the `hf-chat-template` crate (built on
      **minijinja** + a transformers compatibility layer) renders an arbitrary HF `chat_template` Jinja string
      **byte-identically to `transformers.apply_chat_template`**, tools included. Two payoffs to weigh: (1) a
      test that renders the *imported* template via `hf-chat-template` and asserts it byte-matches our
      hardcoded `ChatMl::{qwen2,lfm2}()` output — turning "template-vs-renderer consistency" into a real gate
      instead of a marker check; (2) a general fallback renderer for a checkpoint whose family has no
      hardcoded renderer yet (any model becomes chat-able from its own template). Gate on: it must reproduce
      the parity-committed prompts byte-for-byte before it is trusted, and it adds a minijinja dep (weigh
      against the from-scratch ethos — likely a dev-dependency for the consistency test first). *(2026-07-18
      research)* — https://docs.rs/hf-chat-template · https://github.com/mitsuhiko/minijinja
      *(2026-07-19 research)* Now concrete: `hf-chat-template` is at **0.2.0** and its `RenderInput` carries
      `tools`/`documents`/the generation-prompt flag with **tool-calls supported** (assistant `tool_calls` +
      `tool` responses), claiming byte-identical output to `transformers.apply_chat_template`. This is the
      natural upgrade to payoff (1): the 2026-07-19 gate above is the *cheap, marker-based* consistency check
      (it only asserts the template's tool-call **convention** matches the family renderer); a dev-dependency
      test that renders the **imported** template via `hf-chat-template::RenderInput` and asserts byte-equality
      with `ChatMl::{qwen2,lfm2}().render_with_tools(...)` on the parity-committed prompts would turn
      "template-vs-renderer consistency" into a true byte gate. Still gate on reproducing the committed prompts
      byte-for-byte before trusting it. — https://lib.rs/crates/hf-chat-template
      *(2026-07-22)* Crate is now at **0.2.1** (June 20, 2026 — a same-day patch over the 0.2.0 evaluated
      above; same `RenderInput` surface). Version to use when this is picked up.
      *(2026-07-23)* **Adopted as a dev-dependency; payoff (1) SHIPPED as `tests/template_gate.rs` — and it
      caught two real divergences.** The gate renders the cached Qwen3-0.6B checkpoint's own imported
      template through `hf-chat-template` 0.2.1 (default features on — real HF templates call minijinja's
      Python-compat string methods, `startswith` included) and byte-compares against `ChatMl::qwen2()`:
      **all four legs are byte-identical** — plain (142 B), multi-turn (201 B), full Hermes tools block
      (748 B), and FC history (`<tool_call>` turn + tool response, 324 B). Getting there surfaced: (a)
      `hf-chat-template` hard-requires `serde_json/preserve_order`, so as a dev-dep it silently flipped
      test builds to insertion-order JSON while production builds stayed alphabetical — prompt bytes would
      have differed between what tests verify and what consumers ship. Resolved by making `preserve_order`
      a first-class workspace feature (insertion order is what Python/transformers renders — the training
      distribution). (b) our tool JSON was serde-compact (`{"a":1}`) where transformers' `tojson` emits
      Python `json.dumps` spacing (`{"a": 1}` — the spacing models emit back in their own tool calls);
      fixed with a ~30-line `python_json` serializer (custom `serde_json::ser::Formatter`) now used for
      every prompt-JSON site (Hermes `<tools>` block, LFM `List of tools:` line, `<tool_call>` history
      blocks). Live re-proof after both changes: Qwen3-0.6B greedy-emitted a clean parseable
      `<tool_call>{"name": "get_weather", ...}</tool_call>` from the new prompt bytes on the 4070 Ti SUPER.
      Payoff (2) — a general fallback renderer for checkpoints without a hardcoded family renderer — is
      split below.
      *(2026-07-24, merged from the parallel nightly)* **Gate coverage extended beyond Qwen3**: the same
      in-process harness now byte-verifies **Qwen2.5-1.5B** (plain / tools / FC history) and **LFM2.5-1.2B**
      (plain±system, tools±system in the LFM bare-JSON convention, history think-stripping, pythonic
      `<|tool_call_start|>` + `tool` role turns — LFM's legs also exercise the standalone
      `chat_template.jinja` import fallback on the real checkpoint, whose template file was fetched into
      the local cache). Known family divergences are PINNED to their exact deltas so any other drift still
      fails: Qwen2.5's no-system branding preamble ("You are Qwen, …") vs our neutral one (with tools AND
      the plain injected default turn), Qwen3's no-system no-preamble, Qwen3's history think-stripping.
- [x] **General fallback chat renderer via `hf-chat-template`** — payoff (2) of the evaluation above: for a
      checkpoint whose family has no hardcoded `chat` renderer, render prompts from its own imported
      `chat_template` (the byte gate proved fidelity on Qwen3). Weigh promoting the dep from dev to optional
      runtime feature vs the from-scratch ethos; needs the P8/consumer-facing API decision of when to prefer
      the imported template over a family renderer. *(2026-07-23, split from the evaluation.)*
      *(2026-08-06) **Shipped as `mummu::template`, behind the non-default feature `jinja-template`.** Both
      open questions decided: **(a) the dep** is promoted from dev-only to an *optional* runtime dependency
      — same crate, same 0.2.1 the byte gate already trusts as the transformers-equivalent reference — so a
      default build still carries no Jinja engine and the from-scratch ethos holds for the zoo, while a
      consumer that must run an un-ported checkpoint opts in. **(b) the selection rule** is a value, not a
      convention: `Renderer::for_checkpoint(family: Option<ChatMl>, dir)` takes the family renderer when the
      caller has one and falls back to `ImportedTemplate` otherwise, and it deliberately does **not**
      second-guess a family renderer by reading the template (the gate pins those bytes; a checkpoint
      repackaged with a foreign template is caught at *load* by the `tokenizer.rs` consistency gate, not
      silently obeyed at render). API mirrors `ChatMl`: `render` / `render_with_tools`, plus
      `render_with_tools_json` because the `tools` shape is genuinely open — the mainstream templates unpack
      the `transformers` `{"type":"function","function":{…}}` wrapper, LFM2.5's wants the signature bare.
      Bounded + fail-loud throughout (`Absent` / `Jinja` / `TooLarge` at 8 MiB / `BadTool`; a runaway
      template trips the byte bound rather than returning an untokenizable prompt — unit-tested with a
      render bomb). The one model change: `chat::Turn` gained an additive `tool_calls: Vec<ToolCall>` field
      that `assistant_tool_calls{,_lfm}` now populate **beside** the rendered content — the family renderers
      never read it (prompt bytes unchanged by construction *and* by the gate), but the imported path passes
      calls as data so the template writes its own markers instead of inheriting Hermes' `<tool_call>`
      wrapping. Proof: 7 unit tests over a toy Jinja template (no fixture needed) + `tests/imported_render.rs`
      on real checkpoints — byte-identical to `ChatMl::qwen3()` on plain 142 B / tools 748 B / FC history
      324 B and to `ChatMl::lfm2()` on plain 157 B / tools 379 B (that leg also exercising the standalone
      `chat_template.jinja` fallback and `bos_token` injection, which the gate had to hand-inject); all 10
      template-gate legs re-passed unchanged, 209 unit tests with the feature (202 without), clippy clean in
      both configurations, and Qwen3-0.6B still greedy-emits a parseable `<tool_call>` on the 4070 Ti SUPER.*
- [x] **`ChatMl::qwen3()` with history think-stripping** — the byte gate documented that Qwen3's template
      strips `<think>…</think>` reasoning from assistant turns at/before the last user query while our
      shared `ChatMl::qwen2()` renderer re-renders history verbatim (fine for fresh prompts + tool loops,
      wrong for long multi-turn chats with a thinking Qwen3). A `qwen3()` constructor wants the LFM-style
      strip (the machinery exists — `turn_content` already does it for `Lfm`) but keyed to Qwen3's
      "at/before the last user query" rule rather than LFM's "every but the last assistant turn"; gate it
      on the template byte gate's think case flipping from documented-divergence to byte-equal.
      *(2026-07-24, found by the byte gate.)* *(2026-07-30) **Shipped, and the gate flipped** — plus the
      other pinned Qwen3 divergence for free. `ChatMl::qwen3()`: think-strip is now a per-family
      `ThinkStrip` policy on `ChatMl` (`Keep` Qwen2.5 / `PastAssistant` LFM / `BeforeLastUserQuery`
      Qwen3), implemented byte-for-byte from the template's Python chain — strip = text after the final
      `</think>` `lstrip('\n')` (newlines only, NOT LFM's full trim), "last user query" excludes
      pre-wrapped `<tool_response>` user turns, and an assistant turn AFTER the last query (mid tool
      loop) keeps its reasoning re-emitted in the normalized `<think>\n…\n</think>\n\n` shape. The
      no-system tools preamble also became per-family (`qwen2()` keeps the neutral default; `qwen3()`
      injects none, per its template). Byte gate: the think case AND the no-preamble case flipped from
      pinned-divergence to **byte-equal** (think-strip 249 B, tools-no-system 724 B, and a new
      think+tool_call normalization leg 361 B — all byte-identical vs the imported template through
      hf-chat-template; every other leg unchanged, qwen2/LFM divergences still pinned exactly). 5 new
      unit tests (181 total). REAL-GPU proof: `real_toolcall_qwen3` now renders with `qwen3()` — from
      the no-preamble prompt, Qwen3-0.6B greedy-emitted `<think>…</think>` + a clean parseable
      `<tool_call>{"name": "get_weather", …}</tool_call>` on the 4070 Ti SUPER.*
- [x] **Model registry / manifest** — a declarative `ModelSpec` (repo, architecture, weight format, dtype,
      tokenizer, chat template, size tier) + a small built-in catalog of known-good models (Qwen2.5, LFM2.5,
      MiniLM, …); adding a model = a manifest entry. *(2026-07-10) `mummu::registry`: `ModelSpec`
      (name/repo/revision/architecture/size, validated incl. traversal-safe names, serde round-trip) +
      `spec.fetch(models_root, progress)` onto the hub downloader; built-in catalog: Qwen2.5-1.5B/0.5B,
      LFM2.5-1.2B, all-MiniLM (repo ids match laurelane's validated constants); the network proof now
      fetches spec-driven. Weight-format/dtype/chat-template fields accrete as those import paths land.*
      *(2026-07-13) **Weight-format field landed**: `ModelSpec.format` (`WeightFormat::Safetensors` |
      `Gguf { file }`, serde-defaulted so old manifests still parse); GGUF specs fetch the one file via
      the resumable hub path, `gguf_path()` names where it lands, validation rejects unsafe file names.
      Catalog gains single-file Q4_K_M entries for Qwen2.5-1.5B and LFM2.5-1.2B (quarter the download of
      the safetensors). REAL-NETWORK proof (`real_hub.rs`): the LFM2.5 GGUF spec installed end-to-end —
      697 MB fetched, header parses as `lfm2` (148 tensors), tokenizer built from its metadata.*
- [x] **Import validation** — checked load + a post-import liveness smoke, with a clear error taxonomy.
      *(2026-07-17)* Two taxonomies now cover the pipeline end to end: **`ImportError`** for the file→module
      stage (missing file, parse, load, `Incomplete` with a per-tensor missing/errored diff — key mismatch
      and unsupported dtype surface here, never a silent zero-init), and a new **`SanityError`** for the
      *runtime* stage a checked load can't see — `NonFinite` (NaN/Inf logits: bad dtype or corrupt bytes
      that still deserialize to the right shape), `WrongVocab` (logits width ≠ tokenizer/config vocab), and
      `Degenerate` (spread < 1e-4: a dead/zero-init forward reported as fully applied). `import::logit_sanity`
      is the pure check (6 unit tests over the taxonomy, incl. the width-before-index-access ordering);
      `CausalLm::sanity_check(probe_ids, expected_vocab, device)` is the model-level gate an app calls right
      after `install`. On real Qwen3-0.6B it reports a healthy live distribution (spread 49.4). The
      *"first-token parity smoke against a reference"* is, for catalog models, the P7 parity gates (Qwen2 /
      Qwen3 / LFM2.5 / MiniLM — all passing); an arbitrary user import has no reference, so the liveness
      smoke is its general trust check.

### P4 — Tokenizer & chat templates *(ex-laurelane)*
- [x] HF `tokenizers` (pinned); explicit chat templates (ChatML + per-model), correct special/EOS tokens.
      *(2026-07-10) `mummu::chat`: `Turn`/`Role` + a `ChatMl` renderer with per-model constructors
      (`qwen2` plain, `lfm2` with `<|startoftext|>` BOS); byte-verified — the Qwen2 parity gate now
      renders its prompt through the template and still matches the Candle fixture and the Ollama fp16
      greedy leg exactly. EOS stays config-driven (`EosIds`); tool-use templates are the next item.*
- [x] Hermes-style tool-use chat template (the format Qwen2.5/Qwen3 ship in `tokenizer_config.json`) —
      function calling is why the apps want a local runner —
      https://qwen.readthedocs.io/en/latest/framework/function_call.html
      *(2026-07-11) `chat`: `ToolSpec` → `render_with_tools` (byte-matches the Qwen template's `# Tools`
      /`<tools>`/`<tool_call>` wording), `Turn::{assistant_tool_calls, tool_response}` (consecutive tool
      results merge into one user turn, per the template), and a bounded `parse_tool_calls` extractor
      (calls + prose, loud error taxonomy). REAL-GPU proof (`tests/real_toolcall.rs`): Qwen2.5-1.5B
      greedy-emitted `<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>`
      from a rendered prompt and the parser round-tripped it. 10 new unit tests.*
- [x] LFM2.5 bracket-notation tool-call template + parser (`<|tool_list_start|>` special tokens,
      Python-ish call syntax) — with Hermes/Qwen2.5 (0.880 agent score) done, LFM2.5-1.2B (same score,
      fastest at ~1.5 s on 2026's 21-model local tool-calling benchmark) is the other target —
      https://mikeveerman.be/blog/github-2026-02-06-tool-calling-benchmark/
      *(2026-07-12) Shipped to LFM**2.5**'s actual wire format (its `chat_template.jinja` + model card —
      NOT the gen-1 `<|tool_list_start|>` wrapping, which 2.5 dropped): tools as bare JSON on a
      `List of tools: […]` system line (no default preamble), tool results as real `tool` role turns,
      `</think>` reasoning stripped from all but the last assistant history turn, calls emitted as a
      Pythonic list in `<|tool_call_start|>…<|tool_call_end|>`. `chat`: style-split `render_with_tools`,
      `Turn::assistant_tool_calls_lfm`, and a bounded recursive-descent parser (`parse_tool_calls_lfm`,
      depth ≤ 8, ≤ 64 calls, Python AND JSON literal spellings, byte-offset error taxonomy). REAL-GPU
      proof (`tests/real_toolcall_lfm.rs`): LFM2.5-1.2B greedy-emitted exactly
      `<|tool_call_start|>[get_weather(city="Paris")]<|tool_call_end|>` from our rendered prompt and the
      parser round-tripped it; the Qwen2 parity gate re-passed both legs after the template refactor
      (max |Δlogit| 2.670e-5, Ollama greedy byte-identical). 16 new unit tests (109 total).*
      *(2026-07-23/24) Template-embedded tool JSON now spells `json.dumps` separators (`python_json` —
      `{"a": 1}`, not compact `{"a":1}`) in BOTH conventions, with insertion-order keys (serde_json
      `preserve_order`, a workspace feature): the P3 template byte gate proved the checkpoints' own
      templates (Jinja `tojson`) and the models' own emissions use that spelling, and the renders are
      now byte-identical to `transformers.apply_chat_template` — see the P3 gate item.*
      *(2026-07-10 research)* 2026 community numbers back the plan: Qwen3-8B keeps tool-calling score
      through Q4_K_M (0.919 quantized vs 0.933 full — quant does NOT cost tool reliability, good news for
      P9); BFCL shows a capability cliff below ~7B (Qwen3.5-9B 66.1% vs 4B 50.3%), so the zoo's
      function-calling tier should target the 7–9B class once quant lands; Hermes 4 (Qwen3-14B fine-tune)
      emits `<tool_call>` tags after an explicit reasoning step — easy to parse with the same template
      machinery — https://www.promptquorum.com/power-local-llm/best-local-models-tool-calling-2026 ·
      https://localaimaster.com/blog/best-ollama-models-for-agents

### P5 — Decode engine *(ex-laurelane)*
- [ ] **Async, multithreaded runtime — retire the fully-sync library surface** — Mummu is sync by
      construction today, deliberately and consistently: `decode::generate_loop` is a blocking driver
      that hands tokens to an `FnMut(u32) -> ControlFlow<()>` callback, `ureq` is blocking HTTP ("sync
      like the rest of the library surface", per its own `Cargo.toml` note), `backend.rs` wraps wgpu's
      async adapter enumeration in `pollster::block_on`, and `ModelSlot::with` serializes every
      inference behind a `Mutex`. The whole crate contains **zero `async fn`, zero `.await`, and one
      `thread::spawn` — in a test**. That was the right first cut; it is now the ceiling on three
      separate things (streaming ergonomics, multi-GPU, concurrent requests), so the transition is
      worth planning properly rather than bolting a runtime on.
      **Why it is a real lever here, and where it is not.** Decode on this stack is **dispatch-bound,
      not bandwidth-bound** — established three independent ways (f16 initially matching f32 exactly;
      ~114 GB/s effective against the card's ~672; a **~30 % swing from host CPU load alone** on an
      otherwise-idle GPU). The host thread submitting kernels *is* the bottleneck, so moving
      tokenization, logits readback, detokenization and the consumer's own per-token work **off the
      submit thread** attacks the measured bottleneck directly, which is unusual — most async wins are
      about hiding I/O latency, and this one is not. State the converse in the same breath so nobody
      expects magic: async makes **no single decode step faster**, and a naive multithreaded executor
      makes things *worse* by scheduling the submit thread onto whichever core is cold.
      **What the type system already permits (probed 2026-08-21, burn 0.21).** The naive expectation is
      backwards. Every model in the zoo — `Qwen2` / `Qwen3` / `Lfm2` / `Olmoe`, on both `Gpu` and `Cpu`
      — is **`Send` AND `Sync`**, and so is the KV cache (`Vec<LayerKv<Gpu>>`); a compile probe with a
      deliberate `Rc<u8>` control confirms the check is real and only the control fails. So inference
      state may cross threads freely, and **one owning worker thread per device** is available today.
      What is *not*: `burn_store::TensorSnapshot` holds an `Rc<dyn Fn() -> Result<TensorData, _>>`
      (`burn-store-0.21.0/src/tensor_snapshot.rs:53`), so the **entire import pipeline is `!Send`** and
      cannot cross a thread boundary or be held across an `.await` on a multithreaded executor. A load
      must run to completion on one thread; a public `async` load can only mean "hand the work to that
      thread and await the result". Consequence for P3: this is a constraint upstream owns, so re-check
      it at the burn 0.22 bump before designing around it.
      **Correct a stale premise first.** `cache.rs`'s header says "Burn's `Param` is not `Sync`, so the
      loaded value lives behind a `Mutex`" — the probe above says `Param` **is** `Sync` at 0.21, so the
      `Mutex` is no longer justified by the type. Serializing work on a single GPU is still a perfectly
      good *policy*, but the comment must say that instead, or the next person reads a type-level
      prohibition that does not exist and designs around a phantom.
      **The app-agnostic constraint is the main design risk.** Mummu is shared by laurelane and Nanna.
      A library that picks `tokio` and spawns its own tasks **imposes that runtime on both apps** — the
      exact app-coupling the North Star forbids, and worse than an app concept leaking in because it is
      unfixable downstream. So: return `impl Future` / `impl Stream` and **never spawn**; keep the sync
      API as the base rather than a wrapper over an async one (sync-over-async deadlocks; async-over-sync
      does not); put any runtime dependency behind a non-default feature, the `jinja-template` precedent.
      Confirm each consumer's runtime before committing to a `Stream` shape (Nanna is Tauri, i.e. tokio;
      laurelane unverified).
      **Slices, smallest first, each shippable alone:**
      (a) fix the stale `cache.rs` premise and land the `Send`/`Sync` facts as compile-time assertions,
      so a burn bump that revokes them fails the build instead of the design;
      (b) give `generate_loop` a pull-based streaming shape (an iterator/`Stream` of tokens) beside the
      `on_token` callback, adopting no runtime — pure API ergonomics, and the piece both apps touch;
      (c) **move each device onto an owning worker thread with a channel API** — the real change, and
      the same shape P6's *Multi-GPU execution* item needs, so build it once and share it;
      (d) async I/O for the P3 downloader, the one genuinely I/O-bound path, where classic async pays;
      (e) concurrent request serving / continuous batching — the only slice that raises *throughput*
      rather than moving latency around, and big enough to be its own item when (c) lands.
      **Gate:** every parity leg byte-identical, `bench/BASELINE.md` held, and specifically that
      **seeded sampling stays reproducible** — a multithreaded executor is exactly where determinism
      dies, and `real_inference`'s `qwen2_sampled_streaming_is_seeded_deterministic_and_cancellable`
      is the test that must not become flaky. *(2026-08-21, requested.)*
- [x] Per-layer KV cache (+ conv-state cache for hybrids); prompt prefilled once, then one token/step.
      *(2026-07-09) `nn::{LayerKv, ConvState}` + per-model `new_cache`/`forward(past)`; prefill+decode ≡
      full-forward proven by unit tests at block AND whole-model level, then by real GPU decode.*
- [x] On-GPU argmax (sync only the winning index); single-token decode skips the causal mask.
      *(2026-07-09) `decode::argmax_id`; `t == 1` builds no mask.*
- [x] Sampling beyond greedy (temperature / top-p); **token streaming** via a callback/channel;
      cooperative interrupt/cancellation between tokens. *(2026-07-10) `decode::{SamplerOptions,
      sample_id, Pcg32, generate_loop}` + per-model `generate(…, on_token)`: temperature/top-k/top-p over
      an O(vocab) partial select, deterministic in-house PCG32 (no rand dep), `ControlFlow` callback =
      streaming AND cancellation; greedy stays on-GPU argmax and re-passed the Qwen2 parity gate; real-GPU
      proof: seeded sampled stream replays identically and cancels at 8 tokens (10 new unit tests).*
- [x] Process-lifetime model + tokenizer cache (per backend; behind a `Mutex` since Burn `Param` isn't `Sync`).
      *(2026-07-10) `cache::ModelSlot<T>` — one static slot per (model, backend); `with(key, load, f)`
      loads once, reuses on key match, drop-then-swap on a different checkpoint dir (the P8
      active-model-switch primitive), `clear()` frees VRAM; real-GPU proof: two decodes through a static
      slot performed exactly one 3.1 GB load (6 unit tests + threaded-static test).*
- [ ] **Speculative decoding (draft-verify loop)** *(colibri parity; promotes the Qwen3.5 MTP note from P2)* —
      the decode driver is strictly one-token-per-forward; a draft-verify loop (draft k tokens cheaply,
      verify in one batched forward, accept the agreeing prefix) is the standard decode-latency lever and
      is byte-identical to greedy *by construction* — exactly the kind of speedup our parity ethos allows.
      Two draft sources, in order of reach: (a) **native MTP heads** — Qwen3.5-4B/9B ship `-MTP-GGUF`
      repos claiming ~1.5–2×, and colibri measures GLM-5.2's built-in MTP head at 2.2–2.8 accepted
      tokens/forward *end-to-end* with a disable switch (`SPEC_PIN=1`/`DRAFT=0`) — the gate discipline to
      copy: measure acceptance end-to-end, keep a kill switch, assert byte-equality vs non-speculative
      greedy; (b) a small same-tokenizer zoo model as drafter (0.5B drafts for 9B). Needs: batched-verify
      forward through the existing KV cache (rollback on reject), driver support in `generate_loop`.
      *(2026-07-30 research)* — https://github.com/JustVugg/colibri
      *(2026-08-06 research)* **Calibration that changes this item's expected value: on consumer GPUs a
      small-draft-model speculation is frequently a net LOSS, not a 1.5–2× win.** The 2026 measurements to
      plan against: a 7B target + 0.5B draft on an RTX 5060 Ti ran at **0.27×** (i.e. ~4× slower) and only
      flipped to 1.4× when the target grew to 14B — the draft's cost only amortizes when the target is
      expensive enough per token; and a public 19-configuration llama.cpp study on Qwen3.6-35B-A3B with a
      vocab-matched Qwen3.5-0.8B drafter found **no variant achieving a net speedup** on a single RTX 3090
      (ngram-cache, ngram-mod and classic draft all lost), while vLLM on the same hardware got +27.5 % —
      i.e. the engine's verify-batching quality, not the idea, decides it. Consequences for Mummu: (a)
      route (a) **native MTP heads** is the one to build — no second model to pay for; (b) route (b) small-
      model drafting must be gated on an end-to-end measurement per (target, draft, hardware) triple, never
      shipped on the literature's headline; (c) the batched-verify forward has to be genuinely batched
      through the KV cache, since that is where the engines that win differ from the ones that lose. Our
      own dispatch-bound decode makes (c) harder and the win smaller: a k-token verify is one forward
      either way, so speculation trades dispatches for compute — which is the right direction here, and
      worth re-checking after graph capture (P0) moves the dispatch baseline. —
      https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090 ·
      https://inventivehq.com/blog/llama-cpp-speculative-decoding-consumer-gpu
      *(2026-08-20, second run)* Route (a) now has a merged reference implementation to read and a
      number to aim at: llama.cpp PR **#22673** landed MTP-head support (tested on Qwen3.6-27B and
      Qwen3.6-35B-A3B, but written for any MTP model), driven by `--spec-type mtp` plus
      `--spec-draft-n-max <k>`. Reported steady-state **acceptance ~75 % at k=3 for >2x end-to-end**,
      rising past 80 % on code/math/reasoning, and the community finding that **k=2 often beats k=3**
      because acceptance falls off as the draft window widens — so `n_draft` is a tunable to measure,
      not a constant to pick. Two things to carry into our design: MTP needs no second model in VRAM
      (the heads ride the target checkpoint), which is what makes it the route worth building on a
      16 GB card; and the flag rename noted above means a reference leg should probe BOTH
      `--spec-type mtp` and `--spec-type draft-mtp` rather than assume either. —
      https://github.com/ggml-org/llama.cpp/pull/22673 ·
      https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
- [ ] **Grammar-constrained decoding** *(colibri parity)* — colibri forces structured output via `.gbnf`
      grammars (llama.cpp's GBNF convention) and even uses grammar-forced *drafts* to speed structured
      generation. Mummu's tool-calling currently *trusts* the model to emit parseable `<tool_call>` JSON
      and catches failure only in `parse_tool_calls`. A constrained-sampler stage — mask invalid next
      tokens against a grammar/JSON-schema automaton before `sample_id` — makes tool-call emission
      *guaranteed*-parseable, which is worth more to the apps (agent loops) than raw tok/s. Rust
      substrates to evaluate before hand-rolling: `llguidance` (the guidance-ai engine, token-level
      masks), GBNF-port crates. Gate: greedy path byte-unchanged when no grammar is armed; masked decode
      re-passes the FC real-GPU tests with the parser's error leg now unreachable. *(2026-07-30 research)*
      *(2026-07-30, same run)* `llguidance` evaluated concretely: **1.7.6** (June 2026), MIT, pure Rust —
      JSON Schema + Lark-variant CFGs, ~50 µs/token masks on a 128k vocab (Earley + derivative-based lazy
      lexer + a tokenizer trie), the engine OpenAI credited for Structured Outputs. Integration shape for
      our driver: build a `TokenParser`/`Constraint` from the grammar + vocabulary (the sibling
      `toktrie_hf_tokenizers` crate bridges HF `tokenizers`, which we already load), then in
      `generate_loop`: `compute_mask()` → mask the logits before `sample_id`/argmax → `commit_token()`.
      Weigh the dep tree (toktrie, derivre, rayon) against a hand-rolled JSON-schema automaton; 2026
      benchmarks note XGrammar edges it on repeated-schema caching, but llguidance's Rust-native API is
      the fit here — https://lib.rs/crates/llguidance · https://github.com/guidance-ai/llguidance
      *(2026-08-21, mistral.rs scan)* Production corroboration for the llguidance pick: mistral.rs ships
      its structured output (JSON Schema, regex, Lark CFGs) AND grammar-enforced tool calling on
      llguidance — the exact integration shape sketched above, running in the closest comparable Rust
      engine. The pick is settled; only the wiring remains. —
      https://github.com/EricLBuehler/mistral.rs
- [ ] **Sampling breadth: min-p, repetition/frequency/presence penalties, logit bias** *(mistral.rs
      parity)* — `SamplerOptions` is temperature/top-k/top-p/seed; the OpenAI-shaped requests the
      consumers will forward (and every llama.cpp / mistral.rs config in the wild) also speak min-p,
      the three penalty families, and per-token logit bias. All are host-side transforms over the
      candidate set that slot between the existing O(vocab) top-k select and `sample_id` — no new
      device code (penalties need a small recent-token ring the driver already implies). Defaults keep
      every one of them off; the gate is that an options struct with the new fields at their defaults
      produces **byte-identical** streams to today (greedy AND seeded-sampled replay), so no existing
      parity leg moves. *(2026-08-21, mistral.rs scan.)*
- [ ] **Batched forward (N sequences, one dispatch)** *(mistral.rs parity — the honest subset)* — the
      decode driver is strictly batch-1. Full continuous batching is a serving-stack feature and stays
      a non-goal (single-user consumers, no request queue to schedule), but a fixed-N batched forward
      through the KV cache is the shared prerequisite for two things already wanted: the
      speculative-decoding **verify** step (item above — "genuinely batched through the KV cache is
      where the engines that win differ from the ones that lose") and **bulk embedding** (Nanna memory
      backfills embed texts one forward at a time today). Our dispatch-bound decode profile makes this
      unusually attractive: N sequences per dispatch amortizes exactly the cost the Performance section
      measures as dominant. Gate: batch-of-1 byte-identical to the unbatched path; batch-of-N equal to
      N independent runs. *(2026-08-21, mistral.rs scan.)*
- [ ] **In-memory prompt-prefix KV reuse** *(mistral.rs parity; the warm sibling of P9's KV-cache
      persistence)* — agent loops re-send system prompt + growing history every turn and Mummu
      re-prefills from token zero each time. Keep the last (or LRU-few) prefill's KV in the `ModelSlot`
      keyed by prompt-token prefix; on the next generate, longest-common-prefix match resumes prefill
      at the divergence point — the multi-turn case converts each turn's prefill cost from
      O(whole conversation) to O(new tokens). Same gate as persistence: a resumed decode must be
      byte-identical to the uninterrupted one (the cache-equivalence proofs already cover the
      mechanism; this is bookkeeping, not new math). And per the North Star, no knob: it is simply on
      once proven, like the KV cache itself. *(2026-08-21, mistral.rs scan.)*

### P6 — Hardware planner: precision, placement & full utilization
The "use all the hardware" phase — inventory the machine, then pick the precision and the device placement
that fits the model AND uses every device to the fullest.
- [ ] **Device inventory** — enumerate every GPU (`wgpu` adapters: name, backend, VRAM) and the CPU (cores,
      RAM); a stable device set cached at startup, reported so the apps can show it in settings.
      *(2026-07-11)* Everything but true VRAM shipped: `DeviceInventory` now carries per-adapter
      `max_buffer_bytes` (the planner's hard per-tensor bound; dev box: 4 GiB Vulkan / 2 GiB DX12) and
      `CpuInfo` (logical cores + total RAM — `GlobalMemoryStatusEx` on Windows, `/proc/meminfo` on
      Linux; dev box: 32 cores / 127 GiB). Remaining: per-adapter VRAM capacity, which wgpu does not
      expose portably — needs per-API `wgpu-hal` queries (Vulkan memory heaps / DXGI) — and a macOS RAM
      sysctl. *(2026-07-16)* **True VRAM shipped on Windows**: `GpuAdapter.vram_bytes` from a DXGI 1.1
      adapter walk (hand-bound frozen COM ABI — `windows-sys` 0.60+ dropped COM bindings and the
      `windows` crate is heavy for two vtable calls), matched to wgpu adapters by name with per-API
      decoration tolerance; wgpu still exposes nothing portable (gfx-rs/wgpu#2447 open). Dev box:
      15.7 GiB reported for the 4070 Ti SUPER on BOTH its Vulkan and DX12 rows. Remaining: Linux
      (Vulkan memory heaps), macOS (+ RAM sysctl).
- [ ] **Precision selection** — pick a per-device dtype (f64 /f32 / **f16** / int8 / int4) that fits: f16 via
      `Wgpu<half::f16, i32>`; drop to int8/int4 (P9) when f16 still won't fit. *(2026-07-11) The f16
      backend itself is now **fully validated** (all 3 claims — see the islands item below); what remains
      here is the *picking* logic, which rides the placement-plan item + P9.*
      *(2026-08-09) **The float half of the picking logic shipped: `mummu::plan`.***
      `pick_precision(&ModelShape, &DeviceBudget) -> Option<Fit>` returns the **highest** precision that
      fits one adapter, or `None` — which is the honest answer "no float tier fits; this needs
      quantization or a multi-device plan", never a silently-worse tier. `ModelShape::from_decoder`
      takes the `config.json` numbers (params, layers, kv heads, head_dim, context) and derives the KV
      geometry itself; `DeviceBudget::from_adapter` reads straight off `backend::inventory()` and
      returns `None` when `vram_bytes` is unknown (every non-Windows adapter today) rather than
      guessing; `Fit` carries the projected and usable byte counts plus `headroom_bytes()`, which is
      already the shape the `plan`/`doctor` introspection item will render. Two constants carry the
      judgement and are calibrated against `bench/BASELINE.md` rather than first principles:
      `OVERHEAD_BYTES` (1 GiB for activations/workspaces/CubeCL pools — the residual between measured
      runner VRAM and weights+KV, ~0.5–1.8 GiB depending on dtype) and `USABLE_VRAM_FRACTION` (0.75,
      because the reference box runs 3.5–6.5 GiB of desktop ambient on the same card and a plan that
      ignores it fails at load). 7 unit tests pin the decisions against real hardware and real models:
      Qwen2.5-1.5B projects 7.0 GiB f32 / 3.9 GiB f16 against the measured 8.0 / 3.6; the 15.7 GiB
      reference card gets f32 and an 8 GiB card gets f16; the dev box's own **DX12 rows (no
      `SHADER_F16`) never get an f16 plan**; a 64k context pushes a 12 GiB card from f32 down to f16;
      and OLMoE-1B-7B on a 16 GiB card returns `None`, matching what `bench/BASELINE.md` records as
      "GPU is out of reach until keep-quantized VRAM (P9)". Item stays `[ ]` for the int8/int4 tiers,
      which extend `Precision` downward once P9 lands.
- [x] **f16 mixed-precision islands** — Qwen2.5-1.5B in pure f16 NaNs out (overflow in the
      softmax/RmsNorm/logit reductions; f16 max is 65 504). Keep weights + matmuls f16 but compute the
      numerically hot reductions (attention softmax, RmsNorm accumulation, final logits) in f32, then
      re-run the f16 gate and the parity harness. *(2026-07-11) **One island sufficed**: the q·kᵀ
      attention scores overflow f16 — scores + mask + softmax now compute in f32 (per-tensor
      `cast(DType::F32)`, Burn 0.21 multi-dtype; llama.cpp pins the same matmul to f32), probs return to
      the ambient dtype; Burn's RmsNorm already reduces in f32 upstream, and the logit path needed
      nothing. The f16 gate passes all 3 claims: no crash, **6.75 GiB whole-card / ~3.6 GiB runner**
      (vs ~7.9 GiB f32), coherent greedy output ("2+2 equals 4."). Casts are no-ops on f32: the parity
      gate re-passed both legs (max |Δlogit| 2.670e-5, unchanged; Ollama greedy byte-identical), and f32
      perf *improved* (TTFT 100.5 → 88.4 ms, decode 13.3 → 14.1 tok/s). f16 benches recorded in
      `bench/BASELINE.md`: 88.0 ms TTFT, 14.1 tok/s — speed parity with f32, VRAM halved.*
      *(2026-08-06 correction)* The islands themselves are unaffected — they were validated by
      `real_f16.rs`, one dtype alias per process, genuinely f16 (no NaN, 6.75 GiB, coherent output). But
      the **"speed parity with f32" bench line above is withdrawn**: that row came from the two-alias
      bench process and was an f32 run mislabelled f16 (bisected this run — see the closed drift item in
      the perf section). Real f16 decode is 20.5 ms/token vs f32's 60.0. The same withdrawal applies to
      the SPIR-V item's "+30 % on BOTH dtypes" — only its f32 half was ever measured.
- [x] Evaluate burn-wgpu's **`spirv` compiler feature** on Vulkan (CubeCL SPIR-V backend instead of
      WGSL/naga): claims significantly faster matmul incl. TensorCores at f16 — could be the cheapest
      decode-tok/s lever on the dev GPU; gate on the parity harness + `bench/BASELINE.md` —
      https://github.com/tracel-ai/burn/blob/main/crates/burn-wgpu/README.md
      *(2026-07-12) **Adopted** (burn `vulkan` feature; runtime reports `fusion<cubecl<wgpu<spirv>>>`,
      auto-selected on Vulkan adapters only — other APIs keep WGSL, no code changes): decode
      **70.7 → 54.3 ms/token (14.1 → 18.4 tok/s, +30%)** on f32 AND f16; TTFT 88.4 → 96.7 ms (well
      under the 150 ms ceiling); VRAM peak unchanged (11.5 GiB whole-card). Parity gate byte-identical
      (max |Δlogit| 2.670e-5, Ollama greedy exact); f16 island + CPU budget gates re-passed
      (108.4 ms / 11.7 tok/s GPU gate, 13.2 tok/s CPU). `bench/BASELINE.md` re-baselined.*
- [x] **Per-device default-dtype policy when mixing precisions in one process** — Burn 0.21 resolves
      unspecified-dtype tensor creation against a per-DEVICE settings policy (`get_device_settings` /
      `set_default_dtypes`), not the backend type alias: a `GpuF16` client and a `Gpu` client sharing the
      same device inside one process flip each other's ambient float dtype. *(2026-07-23, found live)*: with
      the three `real_qwen3` GPU tests in one binary, running the f16 leg before the f32 GGUF leg made the
      GGUF load's zeros-probed `target_float` come back **F16**, so the whole f32 model loaded/ran in f16 and
      the strict f32 logits readback died `TypeMismatch("expected F16, got F32")` — deterministic
      sequentially, racy in parallel; A/B-confirmed pre-existing at f1e547a. Two mitigations landed
      (2026-07-23): every loader now takes `target_float` from the TYPE (`<B::FloatElem as Element>::dtype()`,
      never a probe tensor), and all `GpuF16` legs live in their own test binary (`real_f16.rs`) = separate
      process. What remains for the P6 planner (which will legitimately run f32 and f16 models side by side):
      decide the policy explicitly — call `set_default_dtypes` per device at model-load/planner level (or pin
      every runtime tensor creation site's dtype) so in-process mixed-precision is defined behavior, and add a
      two-alias regression test once it is.
      *(2026-07-30)* **Decided and shipped: pin every runtime creation site; the policy registry is not the
      tool.** Reading burn-backend 0.21 source settled it — `set_default_dtypes` is ONE-SHOT per device
      (`OnceLock` semantics: first tensor touch permanently locks the defaults, later calls error
      `AlreadyInitialized`), and the registry key is the *device* type, which `Gpu`/`GpuF16` share — so a
      runtime per-model policy flip is impossible by design, and pinning is the only correct shape. New
      `backend::{float_dtype, int_dtype}::<B>()` expose the TYPE-level dtypes; all 7 policy-dependent
      runtime creation sites now pass an explicit dtype (`from_data(…, (device, dtype))`): rope cos/sin
      tables, the causal mask, MiniLM's ids+mask, and the three decoders' token-id tensors (the f32-softmax
      island was already input-derived via `q.dtype()`; load paths were type-pinned 2026-07-23). Proof:
      new `tests/real_mixed_dtype.rs` (own binary — its policy pollution is the experiment) runs the
      historically poisonous order on the real GPU: an f16 Qwen3-0.6B forward locks the shared device
      policy to F16, then the f32 model in the SAME process loads, forwards with `logits.dtype() == F32`,
      survives the exact strict-f32 readback that died `TypeMismatch` on 2026-07-23, agrees with the f16
      leg on the greedy top token (151667, spreads 49.4), and greedy-decodes "2 + 2 = 4." coherently.
      Parity unchanged by construction AND by measurement: the Qwen3 GGUF-vs-llama.cpp strict leg re-passed
      bit-identically (top-5 exact in order, max |Δlogprob| 4.015608155114805e-1 unchanged, 24-token greedy
      byte-identical). The one-alias-per-process test convention stays for the OTHER suites (defense in
      depth), but is no longer load-bearing for pinned code paths.
- [ ] **Placement plan** — given model size + KV-cache + display headroom and the device set, choose a
      **fit-and-fill** plan: single GPU when it fits; **shard layers across multiple GPUs** (pipeline/
      layer-parallel over Burn's multi-device tensors — Burn gives the multi-device *primitives*, not automatic
      tensor-parallel, so we place modules on devices ourselves); **spill cold layers to CPU** (GGUF-style
      hybrid) when total VRAM is short. Largest-model-that-fits, every device busy.
      *(2026-07-24, design settled)* **Heterogeneous per-device precision comes from ONE source file —
      no new format.** The checkpoint stores weights once (bf16 safetensors, or a GGUF); each pipeline
      stage *derives* its own in-memory representation at load: cast bf16→f32 for the big GPU and
      bf16→f16 for a SHADER_F16-capable iGPU (both = the existing `CastFloatAdapter` path, quality-free
      casts), and quantize int8/int4 for the CPU stage (the P9 keep-quantized leg; naive round-to-nearest
      from bf16 is worse than a calibrated GPTQ/K-quant artifact, so prefer an on-the-fly *block-wise*
      Q4_K-style quant — llama.cpp's offline K-quants are data-free, same math). The inverse also holds
      and already runs: one Q4_K_M GGUF can serve f32/f16 stages by dequant/upcast (at Q4 quality) and
      the CPU stage keep-quantized — so "one file" is a quality-vs-disk choice (bf16 = quality-max,
      GGUF = size-min), never a format question. The only format-adjacent addition is a later
      **derived-artifact cache** (don't re-quantize 30 layers per launch): a per-user cache dir of
      ordinary safetensors/GGUF shards keyed by (source hash, dtype, layer range) — a cache, not a
      format. The real work is runtime, in dependency order: (1) test the dtype-policy hazard above
      *across* devices first — the flip is per-device, so f32-on-discrete + f16-on-iGPU in one process
      is expected to work but is exactly the unproven experiment; (2) the stage-composed model type —
      `CausalLm<B>` is generic over ONE backend, a GPU+iGPU split can stay one `Wgpu` backend with
      per-tensor dtypes (Burn 0.21 multi-dtype, the f32-softmax island already does per-tensor casts),
      but the CPU stage is a different backend *type* (`burn-flex`), so the GPU→CPU seam is a
      host-memory transfer between two backend generics; (3) activations cast at stage seams (small
      tensors, cheap); (4) per-stage KV-cache shards + the micro-batch schedule (the multi-GPU item).
      Expectation to encode in the planner: pipeline throughput = the slowest stage, so iGPU/CPU stages
      exist to make a model FIT, not to make a fitting model faster — fit-and-fill, per-device precision
      picked by what fits + what the device advertises (`inventory()` already records SHADER_F16 +
      max_buffer_bytes + true VRAM per adapter).
- [ ] **Multi-GPU execution** — run the sharded plan: per-device sub-modules, activations handed across the
      device boundary between stages, KV-cache per shard, and a micro-batch/pipeline schedule so the GPUs
      overlap rather than idle. *(Tensor-parallel within a layer is the stretch goal; layer/pipeline split is
      the tractable first cut.)*
- [ ] **Third tier: stream weights from NVMe** *(colibri parity)* — the placement plan above stops at
      GPU→CPU-RAM spill; colibri's core result is that **disk is a usable third tier**: VRAM/RAM/NVMe as
      one hierarchy ("a JIT for weights") with a per-layer LRU residency cache, a *learned* pinned hot
      set driven by recorded routing history (`.coli_usage`), one-layer-ahead prefetch off a lookahead
      thread (71.6 % next-layer predictability measured), and an async I/O pool overlapping reads with
      compute. For dense models the payoff is modest (every layer is touched every token — streaming =
      bandwidth-bound layer paging, only worth it when RAM is also short); the real unlock is **MoE
      expert streaming** (P2 MoE item is the prerequisite), where per-token active weights are a tiny
      fraction of the total so an LRU + prefetch hides most staging latency. Mummu shape: extend
      fit-and-fill with a `Disk` placement class — keep-quantized expert blocks paged into a bounded
      RAM/VRAM pool, residency keyed by routing frequency, prefetch keyed by the router's early output.
      Gate like everything: parity unaffected by placement (colibri's own invariant — "placement only
      affects speed, never precision"), and a bench proving the streamed model beats the
      *(2026-08-20 research — read with the P2 MoE note, which carries the detail)* The colibri-shaped
      design above now has an independent implementation to measure against: llama.cpp PR #25294 does
      per-layer bounded expert-slab caching + async demand-load + hotness/LRU eviction + `O_DIRECT`, and
      reports 5.3x prefill / 2.4x decode over `mmap`+`--n-cpu-moe` at a 79 % hit rate. Two design points
      worth stealing outright when the `Disk` placement class is built: eviction on **decaying route
      hotness with an LRU tiebreak** (not plain LRU — MoE routing is skewed, and plain LRU throws away
      a hot expert after one cold burst), and **`O_DIRECT`/unbuffered reads**, because the OS page cache
      actively hurts once the model exceeds RAM. Its stated limitation is also a design constraint for
      us: single-context only — concurrent decodes sharing one streamed model corrupt each other, so the
      residency pool has to be owned per-session or locked. — https://github.com/ggml-org/llama.cpp/pull/25294
      largest-fitting resident one on task throughput. *(2026-07-30 research)* —
      https://github.com/JustVugg/colibri
      *(2026-08-03 research)* Prior art to mine when this is picked up: llama.cpp's **`--n-cpu-moe N`**
      flag (core algorithm in its PR #15077) keeps attention on-GPU and moves the first N layers'
      expert FFN weights to CPU RAM — community numbers show 12–24 GB cards running 35B-class MoE at
      50–60 tok/s, which calibrates what expert-CPU-offload alone (no NVMe tier) buys; and an open
      llama.cpp feature request (#20757) sketches the **two-tier GPU+RAM expert cache with pluggable
      eviction** — the same LRU-residency design colibri proved, upstreamed. Mummu's OLMoE port
      (2026-08-03) makes both concrete here: expert banks are single fused 3-D tensors per layer, the
      natural offload/eviction unit — https://github.com/ggml-org/llama.cpp/issues/20757 ·
      https://openclawdc.com/blog/llama-cpp-moe-offload-flags-explained/
- [ ] **Planner introspection (`plan` / `doctor`)** *(colibri parity)* — colibri ships `coli plan`
      (print the placement decision without running) and `coli doctor` (readiness checks). Mummu's
      planner should expose the same as *API*, consumer UIs render it: a `Plan` report (per-device
      layers/dtype/VRAM-projection + why) computable without loading weights, and a `doctor`-style
      preflight (weights present + hash-verified, VRAM/RAM headroom vs plan, backend/adapter features
      like SHADER_F16) with a clear error taxonomy. Cheap to build once the placement plan exists —
      it *is* the plan, printed instead of executed. *(2026-07-30 research)*
- [ ] **`[hardware]` config + overrides** — auto by default; explicit overrides (device list, per-device layer
      counts, precision, CPU-offload cap) for power users.

### P7 — Parity & performance harness *(ex-laurelane)*
- [x] Parity gate: single-forward top-k logits + a short greedy sequence must match a reference (Candle,
      or a local Ollama of the same model) — the trust gate every port passes. *(2026-07-09) Greedy leg
      exists (`tests/parity_lfm2.rs`, Ollama raw-mode temperature-0 via curl).* *(2026-07-10) Both legs
      live and passing for Qwen2.5-1.5B (`tests/parity_qwen2.rs`): logits leg vs a committed
      `tools/candle-probe` fixture, greedy leg vs Ollama fp16. LFM2.5 still lacks a same-weights
      reference (see the P2 item); candle-transformers has no LFM2, so its logits leg needs another
      route (llama.cpp logprobs, or an HF transformers dump).*
- [x] Stand up same-weights references: a Candle-based logits probe (dev-dependency or a small side
      harness, as laurelane's Qwen2 validation did) + pull `qwen2.5:1.5b-instruct-fp16` in Ollama for the
      Qwen greedy leg. *(2026-07-10) `tools/candle-probe` (out-of-workspace bin, Candle =0.9.1 CPU f32)
      prints top-k (id, logit) JSON for the fixed prompt; fixture committed under
      `crates/mummu/tests/fixtures/`; fp16 Ollama tag pulled and validated.*
- [x] LFM2.5 same-weights reference for the parity gate: no Candle port exists — candidate routes are
      llama.cpp `logprobs` on the fp16 GGUF, or a one-shot HF `transformers` logits dump matched to the
      safetensors revision. *(2026-07-16) **Shipped** exactly as researched: `tests/llama_ref` spawns a
      user-supplied `llama-server` (`MUMMU_LLAMA_SERVER` — Ollama installs bundle one, no separate
      install) on LiquidAI's official **BF16** GGUF (bit-identical to the local bf16 safetensors), RAW
      `/completion` with prompts as **token-id arrays** (no template stack, no BOS injection —
      `tokens_evaluated` is asserted) and `n_probs` top-logprobs. Both P2 legs pass — see the P2 item.* *(2026-07-11 research)* Liquid officially documents running LFM2.5-1.2B
      GGUFs under `llama-server`; its completion API returns per-token top logprobs via `n_probs`
      (temperature 0 for the greedy leg) — that plus an fp16 GGUF of the same revision is a workable
      logits leg without Python — https://docs.liquid.ai/deployment/on-device/llama-cpp ·
      https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
      *(2026-07-12 research)* Caution for that route: llama.cpp's own chat/tool layer mishandles
      LFM2.5 (issue #23838 — its parser rejects the documented `<|tool_call_start|>[…]` format), so
      drive `llama-server` in RAW completion mode (`/completion`, no chat template) and render prompts
      with our byte-verified `ChatMl::lfm2()` — never through llama.cpp's template stack —
      https://github.com/ggml-org/llama.cpp/issues/23838
      *(2026-07-13 research)* That parser bug is now FIXED upstream (PR #24178 merged ~June 2026 —
      `is_lfm2_template()` detected only the gen-1 `<|tool_list_start|>` tags); raw completion mode
      remains the right choice for the logits leg regardless (no template stack in the loop). Also:
      LiquidAI officially publishes LFM2.5-1.2B GGUFs (incl. F16) — the same-weights reference artifact
      this leg needs — https://github.com/ggml-org/llama.cpp/issues/23838
- [x] **Every gate compiles again: the burn 0.22 migration had left 35 of them behind.**
      *(2026-08-25)* `cargo check --all-targets` was **red on `main`** — the library and
      `mummu-serve` had been migrated off the 0.21 backend generic, but **every parity gate, every
      budget gate, every real-model proof, the load gate and three examples had not**. So the whole
      shipping discipline this file rests on — "parity or it didn't ship", "never ship red", the
      `bench/BASELINE.md` budgets — had been unenforceable since the migration, and nothing said so,
      because a target that does not compile reports nothing at all. Four API shifts, all mechanical
      once named: the `Gpu`/`Cpu`/`GpuF16` aliases became `Device` **values** (`backend::gpu_device()`
      / `cpu_device()`), `Tensor<B, D>` lost its backend parameter, `generate` / `first_token` /
      `sanity_check` / `warm_up` / `decode::argmax_id` became futures (gates are `#[tokio::test]`,
      the library's own idiom), and two structs had grown fields (`DeviceExpert::native_ok`,
      `TierDevice::preload_units`). Three places needed a real decision rather than a rename:
      `ModelSlot::with` takes a *sync* closure and generation is a future, so the gates that held a
      model across a decode moved to the `acquire` guard the cache already exposes for exactly this;
      criterion's closures are sync, so the bench blocks on a runtime built once per group (the wait
      measured is still the device's fence); and `gguf_store` / `compare_against_llama_cpp` now take
      the device explicitly, which is what moved the OLMoE parity leg back onto the host where its
      ~28 GB f32 build actually fits.
      **Both restored gates were then RUN on real weights**, on `--no-default-features --features
      vulkan-spirv` (see the fusion item above for why not the default set): `parity_qwen2`'s logits
      leg matches the committed Candle fixture top-5 **exactly in order** (785, 16, 32, 1249, 8420) at
      max |Δlogit| **1.0681152e-4** against a 1e-3 bound, and `parity_gguf`'s qwen3 leg matches
      llama.cpp on the identical Q4_K_M file — top-5 exact in order (151667, 151644, 151645, 99966,
      131545), the 24-token greedy sequence **byte-identical** including the `<think>` tokens, max
      |Δlogprob| **4.0152457458654034e-1** against the recorded **4.015608155114805e-1**. Those two
      agreeing to four decimals is the evidence that reading the dtype off the device changed nothing
      at F32; the fifth-decimal drift is burn 0.21 -> 0.22, not this change. 312 unit tests green.
      `parity_qwen2`'s Ollama greedy leg could not run — the `qwen2.5:1.5b-instruct-fp16` tag is not
      pulled on this machine (own `[ ]` below).
- [x] **...and the migration had silently dropped f16 — the gates that would have caught it were
      among the 35.** `backend::float_dtype()` returned the constant `DType::F32` and 28
      tensor-creation sites in the library called it, so a device configured for f16 would have been
      overridden back to f32 at every creation site. burn 0.22 keeps the element type on the device
      as a runtime setting (`Device::configure`), so the fix is to read it there:
      **`float_dtype(&device)` / `int_dtype(&device)`**, plus **`backend::gpu_device_f16()`** as the
      `GpuF16` alias's replacement. Device settings lock on first use and cannot be changed, which is
      the 0.21 one-alias-per-process rule now enforced by burn itself — `gpu_device_f16` is therefore
      idempotent within a process, and **returns an error rather than an f32 device wearing an f16
      label**, the exact shape of the 2026-07-11 mislabelling bug. The bench arm carries the same
      guard: it asserts the device's dtype against the row's name before believing a number, and runs
      the f16 arm FIRST because whichever arm runs first locks the process.
      Default behaviour is unchanged by construction (a wgpu device defaults to F32), which is what
      the re-run f32 parity numbers are evidence for.
- [ ] **`cargo fmt --check` is red across the tree — ~330 hunks in ~50 files**, concentrated in the
      newest modules (`nn/moe.rs` 41, `pack.rs` 39, `mummu-serve/src/engine.rs` 45, `models/qwen35.rs`
      24). There is no `rustfmt.toml`, so this is default-rustfmt drift accumulated over many runs,
      not a house style. Deliberately NOT bundled into the 2026-08-25 migration commit — a 6 800-line
      reformat would have buried a 57-file semantic change. Land it as its OWN commit, touching
      nothing else, and then the guardrail is checkable again on every later run. *(2026-08-25.)*
- [ ] **The parity and budget gates need a compile-only CI leg.** The 35-target breakage above
      survived several runs precisely because `cargo test` (which compiles only what it runs) and
      `cargo build` (default targets) both stayed green while `--all-targets` was red. A cheap
      `cargo check --all-targets --keep-going` in the run's verify step would have caught it the day
      it landed; it is now the first thing this routine runs. Worth making it a checked-in CI job so
      it does not depend on a routine remembering. *(2026-08-25.)*
- [ ] **Run every gate on BOTH feature sets — `fusion` on and off.** The 2026-08-25 parity re-run
      found the default (`fusion`) build cannot complete a single forward on burn 0.22.0-pre.2 while
      the `--no-default-features --features vulkan-spirv` build passes byte-for-byte. A gate that only
      ever runs one feature set cannot see that, and `fusion` is the set consumers get by default. The
      cheap version is a second invocation of the same gates with the flag flipped; the honest version
      records both numbers in `bench/BASELINE.md` so a divergence between them is itself a regression.
      *(2026-08-25.)*
- [ ] **The Ollama greedy leg has no fixture on this machine** — `parity_qwen2`'s second leg fails
      with `model 'qwen2.5:1.5b-instruct-fp16' not found` while its logits leg passes. That is the
      missing-fixture-vs-wrong-answer confusion the P3 item above is about, hit in the field: the run
      summary says FAILED for a tag that was never pulled. Pull the tag (`ollama pull
      qwen2.5:1.5b-instruct-fp16`) or let the leg report a summarized skip — but not both silently.
      *(2026-08-25.)*
- [ ] Wire the perf suite (above) into the parity harness so a correctness *or* budget regression fails CI.

### P8 — Model management API
- [x] Download progress · disk usage · switch/remove models — an app-agnostic API the consumers' settings
      UIs call. *(laurelane has disk-usage + remove; add progress + active-model switch.)*
      *(2026-07-09) Disk usage + traversal-safe removal validation shipped as `manage` (5 unit tests).*
      *(2026-07-10) Composed into `manage::ModelManager`: catalog listing, `is_installed`,
      `install(name, on_progress)` (resumable hub fetch with per-chunk progress), traversal-safe
      `remove`, `disk_report`; active-model switch = a consumer `ModelSlot` keyed by
      `manager.model_dir(name)` (drop-then-swap + `clear()` already proven on the real GPU). 4 new
      unit tests.*

### P9 — Quantization (fit any model to the hardware)
The VRAM lever the P6 planner pulls to make the largest useful model fit the user's actual devices.
- [ ] **Stage 3 — the native multi-precision pack + tiered multi-device MoE** *(directive 2026-08-22:
      "run any model optimally on any hardware")*. Two parts, in order:
      **(a) `.mummu` pack format.** Import converts ONCE into mummu's own artifact: a directory with
      `manifest.json` (source + sha, architecture, config, tokenizer, per-tensor role/shape/offsets)
      and one blob per precision (`q4.bin`, `q8.bin`, `f16.bin`, `f32.bin`; every tensor — and every
      MoE **expert** separately — at every level), so a device memory-maps only the precision and
      the tensors it needs, and the planner picks precision per tensor group without re-importing.
      f64 is widening, not information: weights born bf16/Q4 gain nothing from f64 storage, so f64
      is a *compute* option derived from f32 where a backend supports it — stored levels stop at f32.
      The quantized blobs carry our own layout (i8 / nibbles + f32 block-32 scales); loaders build
      backend tensors via `q_from_data` where the layout matches, else dequant→quantize per tensor.
      **(b) tiered MoE execution.** Experts as an `ExpertPool` spread across devices (CPU efficiency
      cores / performance cores / iGPU / dGPU), each shard at the precision that device runs best,
      activations crossing devices as `[tokens, hidden]` host copies per layer (a few KB per
      token); a scheduler assigns experts→(device, precision) from the P6 inventory and can
      **hot-swap** precision per expert (the pack makes every level available instantly). v1:
      static assignment by capability; v2: live re-tiering on load/latency/energy signals.
      Status *(2026-08-22)*: **(a) implemented** — `mummu::pack` (manifest + `header.gguf` +
      per-precision blobs; block-32 Q4/Q8 in burn's canonical quantized `TensorData` layout so
      loading is a copy on every backend; nibble-packed Q4 on disk; `quantize_blocks` tested
      incl. the canonical round-trip), `qwen35::{pack_actions, load_from_pack}`,
      `olmoe::{pack_actions, load_from_pack}` (experts as separate pack entries — the per-expert
      tiering hook), and mummu-serve **auto-imports on first use** (`<model dir>/pack`,
      `MUMMU_PACK=off` to disable, `MUMMU_PACK_PRECISIONS` default all four; temp-dir + rename so
      a crash never leaves a half pack) then loads from the pack at the planner's level.
      **Gated** on the 2B BF16 fixture (`tests/real_qwen35_pack.rs`): pack F32 and F16 reproduce
      the GGUF f32 logits with Δ = 0, pack Q8 ≡ streaming Q8 and pack Q4 ≡ streaming Q4 with
      Δ = 0 (same quantizer, stored once). Two importer bugs fell out of the gate: the conv
      kernel's dims order (ggml `ne` vs row-major) and burn 0.21's `TensorData::quantized`, which
      only handles 8-bit values under the default `PackedU32` store (its Q4 reader unpacks nibbles
      the constructor never packs) — the pack now builds the canonical packed-u32 + scales bytes
      itself (`pack::quantized_tensor_data`), which both flex and cubecl ingest. `examples/pack-import`
      pre-imports a GGUF (the 27B's pack is 4 levels ≈ 220 GB — 4.4 TB free on the models drive).
      **(b) implemented** — `mummu::tier` (`plan_tiers`: admission at the cheapest slot so no
      expert is silently dropped, then hottest-first promotion to faster device / higher level
      within budgets; `diff` for hot-swap lists; `smooth_hotness` EMA; 6 tests),
      `nn::{ExpertExec, DeviceExpert<B>, ExpertPool, Routing}` (router on the main backend,
      routed rows cross devices as host f32, every in-service expert runs concurrently on its
      own device, host scatter-add; hit counters; `swap` = atomic per-expert hot-swap; toy test:
      pooled ≡ local, hot-swap preserves outputs), `olmoe::{load_trunk_from_pack,
      load_expert_from_pack, pack_expert_costs, LoadedOlmoeQ::with_pool}`, and mummu-serve's
      tier runtime (`MUMMU_TIERS`, devices = CPU [Q8, Q4] + CUDA [F32, Q8, Q4] or wgpu [F32, Q8];
      budgets = backend budget minus the trunk on the main backend; re-plan after every request
      from routing hits, ≤ 16 moves per pass on a background thread; expert bytes on a backend
      are subtracted from that backend's fit budget). **Gated** on the real OLMoE-1B-7B
      (`tests/real_olmoe_tiered.rs`, 300 s, 2026-08-22): pack import 161 s (1024 experts — Q4 3.9 MB /
      Q8 7.1 MB / F32 25 MB each); all-Q8 pool ≡ same-backend Q8 with Δ = 0; the mixed plan put
      128 experts at f32 on the wgpu GPU + 586 at int8 + 310 at int4 on the CPU (5.0 GiB CPU /
      3.0 GiB GPU resident, loaded in 22 s), first token = f32's, greedy "2 + 2 equals 4." (6 tokens,
      8.0 s — host round trips + CPU dequant fallback; the speed lever is the CUDA/cubecl q_matmul
      tiers); the hit-driven re-plan promoted hot experts (561 moves), 64 hot-swapped live in 1.1 s
      (~17 ms/expert) and the model kept answering. The 27B pack: 851 tensors, 192 GiB, 19.4 min.
      **Deployed** (image rebuilt + `docker compose up -d --no-deps mummu`): the 27B answered from
      its pre-imported pack on the default path (Cpu @ Q4, 24 tokens, 19.3 min wall incl. load);
      OLMoE auto-imported in the container (529 s over the bind mount) and was placed 495 experts
      f32 on CUDA + 529 int8 on CPU — then the **Docker VM's OOM killer** took the process (46 GB
      anon RSS: the 27B still resident in the CPU slot at ~40 GiB + OLMoE trunk/experts + the
      import's dirty page cache). Fixes: the importer `sync_data`s every GiB per blob and
      `sync_all`s at the end (dirty pages no longer pile up against the process); the tier loader
      calls `ensure_host_room` (MemAvailable vs worst-case CPU tier + 6 GiB slack, evicting the CPU
      slot's other model) before placing experts; compose sets `MUMMU_GPU_BUDGET_GB=11` for the
      16 GB card shared with the desktop (lowered to 9 after the re-verified run still showed 15.6 of
      16 GiB in use). **Re-verified after the fix**: OLMoE placed 324 experts f32 + 1 int8 on CUDA
      (7.6 GiB after the trunk) + 699 int8 on the CPU, all 1024 resident in 139 s (cpu 4.61 GiB,
      cuda 7.60 GiB); first request "2+2 equals 4." (47 s incl. prefill — the host-round-trip
      expert path is correct but slow under WSL2 GPU-PV), warm request 32 tokens in 19 s
      (1.7 tok/s), background re-tier moved 16 experts in 6.6 s with 390 queued for later requests.
      Next perf lever: batch a device's in-service experts into one dispatch and keep activations
      on-device when the expert's device is the main backend (no host hop), then the CUDA q_matmul
      int8/int4 tiers.
      **(c) dense models as clusters — implemented and gated (2026-08-22).** `mummu::partition`:
      MoEfication-style *parameter* clustering (balanced k-means on each neuron's gate‖up vector
      under a deterministic JL projection — no calibration data, so it runs on import for any
      size) permutes every layer's FFN intermediate dim so clusters are contiguous and rewrites
      gate/up/down **in place at every stored level** (same bytes, same names: the dense loader is
      unchanged, nothing is duplicated); crash-safe per layer via a journal with pre-rewrite
      fingerprints. `Pack::{tensor_cols, tensor_rows}` slice clusters straight from the quantized
      bytes (block-aligned, no re-quantization); `qwen35::{load_from_pack_partitioned,
      load_ffn_clusters, ffn_names}`; `LoadedQwen35::{with_ffn_pool, with_ffn_skip}`; the layer
      forward = local slab (typed, on the main device) + `ExpertPool::run_dense` (remote clusters,
      host f32, concurrent), exact when everything runs. **Skipping** (stage "MoE-fy for real") is
      a training-free gate-energy router: clusters whose `Σ silu(x·g)²` falls below `tau` × the
      row's total energy are not computed — opt-in only, behind a measured skip table
      (`examples/pack-calibrate`: hotness prior + tau → max |Δ log-prob| / argmax agreement /
      kept fraction, written into the manifest) and `MUMMU_FFN_SKIP_TOLERANCE`. Serve: qwen35
      packs are partitioned on import / first load (`ensure_partition`), `build_partitioned_qwen35`
      plans clusters per-cluster with the calibrated hotness (main device's ladder pinned to the
      trunk level), keeps ≥1 local cluster per layer, then **groups the remote clusters by device**
      into one executor per (layer, device) — in exact mode every cluster runs, so this is one
      matmul per device per layer, not one per cluster (the 27B: ~128 grouped executors, not 2048).
      Placement is static (all clusters always run), so there is no runtime re-tier for dense; the
      fit planner fits only the **trunk** on the preferred backend when units tier out
      (`pack_trunk_bytes`). **GPU tiers are F32/Q8 only** — burn 0.21's CUDA `q_matmul` panics in
      autotune on the pooled f32-input × Q4-weight matmul (`Cast element count must match`) and
      returns garbage, the same GPU-Q4 breakage the wgpu path already refuses; Q4 units run on the
      CPU (flex dequant fallback, correct). Verified in the container: the 27B partitions (64×32
      clusters, 45 min once), the trunk lands on CPU@Q8 (the 27B trunk > the 9 GiB GPU budget),
      FFN clusters group to ~128 executors and load in 837 s (8.7 GiB on CUDA); with the Q4-on-GPU
      fix the mixed CPU-trunk / GPU-cluster plan is exact. Throughput is CPU-trunk bound (~100 s/tok
      for the 65-layer DeltaNet on flex) — the trunk-on-GPU lever needs the trunk to fit VRAM.
      **RUNNING (2026-08-23, commit aacca4e):** the dense 27B now generates coherently through the
      tiered path in the container — trunk on CPU@f32, 1793 FFN clusters on CUDA @ Q4 (8.72 GiB),
      255 on CPU; "2+2 equals…" with a correct `<think>` trace, no panic, no OOM (~29 s/tok,
      CPU-trunk + host-hop bound). Three fixes closed the gap that the 2026-08-22 runs hit: (1)
      `nn::compute_weight` dequantizes a pooled expert's quantized weight **on-device** before the
      matmul — burn 0.21's CUDA `q_matmul` panics on the f32-input × quantized-weight path ("Cast
      element count must match") and wgpu computes it wrong, so we never take it, and storage stays
      Q4 (the GPU holds ~4× more clusters than at F32, which is what makes the model fit VRAM
      instead of spilling to CPU RAM and OOMing); (2) Q4 re-enabled on the CUDA ladder — at F32-only
      the GPU held too few clusters and the planner spilled the model onto the CPU until admission
      failed loudly ("expert N fits no device"); (3) `run_dense` runs a layer's executors
      **sequentially** on the calling thread — a spawned thread makes cubecl-cuda open a new CUDA
      stream/client with its own VRAM pool, and 64 layers of that exhausted the card
      (CUDA_ERROR_OUT_OF_MEMORY "Can create a new stream"). Follow-up perf levers: trunk-on-GPU
      (needs the trunk to fit VRAM), batch grouped clusters, keep activations on-device.
      **MEASURED 6.2x (2026-08-23)**: 29.7 s/tok in the container -> 10.4 cold -> **4.8 s/tok
      warm** running natively on the host, same weights, same correct output. Three causes,
      separated by the cold/warm pair: (1) the container had **no persistent autotune cache at
      all** — CubeCL walks up from CWD (`/` in a runtime image) and defaults its root to the
      Cargo `target/` tree, so every start re-tuned; fixed by shipping `/cubecl.toml` pointing
      at the `/models` bind mount, and the cold->warm gap (10.4 -> 4.8) is that cost amortized
      over 24 tokens. (2) no WSL2 GPU-PV natively. (3) a 45 GiB host budget instead of ~19
      changes placement (1469 CPU clusters at F32 + 579 on GPU at mixed Q8/F32). The native
      server also enumerates the AMD iGPU, invisible inside the container — the iGPU tier is
      only reachable off Docker. **Correction to an earlier entry**: the "GPU is ~25x slower for
      decode (96 ms vs 3.9 ms)" measurement was a COLD autotune cache — autotune running inside
      the timing loop, not throughput. Warm it is 2.5-3.3 ms, on par with the CPU and
      launch-latency bound (44 MB in 2.5 ms), so it amortizes as work per launch grows. Warm
      every probe before believing a number (`examples/sync-probe.rs`).
      Gate on the 2B (`tests/real_qwen35_partition.rs`, 186 s): partition
      61 s (24 layers × 32 clusters of 192 neurons); dense(partitioned) vs dense(original)
      Δ = 2.8e-5; half the clusters remote at f32 Δ = 3.1e-5 (exact up to summation order);
      f32-local / int8-remote Δ = 0.149 logit, same first token, correct answer; skip tau=0.02
      Δ = 4.8 logits on the 2B (SwiGLU is not sparse — skipping is a real trade, which is why it
      is opt-in and measured). `examples/{pack-partition, pack-calibrate}`; `pack-import`
      partitions qwen35 packs automatically. Scope notes: the 27B (`Qwen3.8-27B`) is **dense** (65 layers, no expert
      tensors) — it benefits from the pack's per-tensor level choice, not the expert pool; dense
      multi-device tiering = layer-wise placement with per-device caches (next). CPU core classes
      (efficiency vs performance) are one `Cpu` tier until burn-flex can pin threads; f64 compute
      tiers need an f64-typed backend instance (not wired). Device list comes from the serve
      backends in use, not yet the P6 inventory (iGPU as a separate wgpu adapter).
- [x] **Stage 1 — keep-quantized weights on the single path** *(2026-08-21)*. `mummu::quant`:
      `QuantPolicy::{Off, Q8, Q4}` (burn `Q8S`/`Q4S`, block-32 scales, `MUMMU_QUANT` env), applied by
      the new **streaming GGUF importer** (`qwen35::load_from_gguf_quantized` — one tensor at a time:
      dequant whatever the source stored → device → **re-quantize** → assign; peak = model + one f32
      tensor, never the 109 GB whole-model blob). The SAME modules and forward serve both precisions:
      a quantized `Param` executes through the backend's `q_matmul` (burn-cubecl runs the mixed
      float×quantized matmul natively; flex falls back to per-op dequantize). Embeddings/norms/convs
      stay float deliberately (no `q_select` kernel; precision anchors). **Measured** (`quant-probe`
      example, 2048² matmul): Q8S correct on flex CPU + wgpu + CUDA (~0.04% err); Q4S correct on CPU
      + CUDA (~0.8%) but **wgpu's Q4 kernel returns garbage (~98%)** — consumers refuse Q4-on-wgpu
      loudly (upstream issue to file). Gates: parity_qwen35 re-passed BYTE-IDENTICAL over the
      streaming loader (same 5.02e-2 Δlogprob, same greedy bytes), and the Q8 2B answers the smoke
      correctly with its first-token argmax equal to f32's (real_qwen35). **The 27B tier RAN**
      *(2026-08-21)*: Qwen3.8-27B UD-Q4_K_S → streaming import + Q4 re-quantize in **445.5 s**,
      ~33.7 GB resident (vs ~109 GB f32 — impossible), 24 correct greedy think-stream tokens on
      flex CPU at **73.5 s/token** (the fallback re-dequantizes every weight per matmul — the
      LlamaWeb-style in-kernel path is the throughput follow-up; cubecl's native `q_matmul` already
      does this on CUDA, where the 27B is VRAM-bound instead: ~18 GB > 16 GB). Two burn 0.21 bugs
      found and worked around en route: (a) `Linear::forward`'s weight unsqueeze reshapes packed
      quantized storage and crashes (flex AND wgpu) — qwen35 routes every projection through a
      `qlinear` helper that flattens the FLOAT input instead; (b) block-level `compute_range`
      crashes on rows not divisible by the block width — `QuantPolicy::eligible` now requires
      `dims[1] % 32 == 0` (the 27B's 48-wide β/α projections stay float).
      *(2026-08-22)* **Stage 2 — the default path + MoE.** mummu-serve gained a **fit planner**
      (P6-lite): per request it estimates resident bytes from the GGUF header under each policy
      and picks the first `(backend, quant)` that fits — preferred backend first with the
      `MUMMU_QUANT` ladder Off→Q8→Q4, then the CPU (budget = min(3/4 total, 85% MemAvailable) —
      a shared VM's total lies, learned when a Q8-27B plan wiped the 40-GB-free Docker VM), never
      Q4-on-wgpu; logs every fit check. So `qwen3.8-27b-ud-q4ks` is an ordinary catalog entry
      now: installed in the container's models dir, it plans `Cpu @ Q4` while the 2B stays on
      CUDA f32 — **verified end to end in the deployed container** (2026-08-22): the planner's
      fit log (Cuda@Off 142 GiB … Cpu@Q4 35 GiB vs 42 GiB budget), ~33 GB RSS during import,
      8 streamed tokens through `/api/chat` in 472 s (≈59 s/token on the 12-core VM — the
      flex dequant-fallback bound; the in-kernel path is the throughput follow-up). **MoE:** `nn::SparseMoePerExpert` — experts as independent weight triples,
      each **quantized separately** (own block scales), **routed** compute so only the top-k are
      in service per token (host readback of `[tokens, k]` routing ids; `select` →
      three 2-D matmuls → `select_assign(Add)`), proven EXACTLY equal to the dense-mask path in
      `nn::moe` tests; `olmoe::load_from_gguf_quantized` streams each fused bank once and splits
      it into its 64 members (`LoadedOlmoeQ`, a separate struct so the parity-proven fused path
      is untouched; attention/router/embed/head stay float). **Real-weights gate green**
      (`tests/real_olmoe_quant.rs`, OLMoE-1B-7B Q4_K_M → per-expert Q8 on flex CPU): first-token
      argmax identical to the fused f32 model, greedy answer "2 + 2 equals 4." (234 s incl. both
      loads). Remaining: N-resident expert
      **paging** to disk (needs a fetch-on-demand store; all experts resident-quantized already
      fit everywhere OLMoE runs), other families onto the streaming path, GPTQ/AWQ, the three
      upstream burn reports (wgpu Q4 kernel, packed-weight reshape, block-range on non-divisible
      rows).
- [ ] **Burn-native quant** — Burn's `Quantizer` int8 / int4 (block-wise, ~4–8× weight reduction) on the wgpu
      + CPU paths; quantize on import or on the fly, keyed to the fit target from P6.
- [ ] **Import pre-quantized** — run **GGUF K-quants** (Q2_K–Q8_0, per-layer precision) and **GPTQ / AWQ** int4
      layouts directly (dequant to the compute dtype, or a keep-quantized matmul where the kernel exists), so a
      model already quantized on the Hub loads as-is. *(2026-07-13) The dequant-to-f32 leg of the GGUF
      path shipped in P3 (`load_from_gguf` — every storage dtype); what remains here is **keep-quantized
      in VRAM**, which is the actual fit lever (Q4_K_M currently dequants to the same f32 footprint).*
- [ ] **Steal LlamaWeb's kernel split for the keep-quantized path** — a 2026 paper (LlamaWeb) does
      exactly the P9 target on **WebGPU** (our wgpu substrate) and reports the design that worked:
      keep every format as flat `u32` buffers in VRAM and dequantize *in-kernel*, but split by op —
      matmul/prefill dequants a tile into **shared memory** and reuses it across outputs, while the
      matvec that dominates decode (they measure **80–90 %** of decode time) dequants **straight into
      registers**. One format-agnostic representation carried 21 quant formats without per-format
      kernels. Results: peak memory **−29–33 %** and decode throughput **+45–69 %** vs WebLLM /
      Transformers.js, though prefill lost 21–51 %. Two things to weigh against our own numbers before
      adopting: (a) they conclude decode is **bandwidth**-bound (f16 → q8_0 bought 20–53 %), which is
      the *opposite* of what our f16-exactly-matches-f32 measurement says about our path — so the win
      here may be VRAM-only for us until the dispatch gap closes; (b) their gains lean on subgroup
      matrix ops where available, with portable fallbacks. — https://arxiv.org/html/2605.20706v1
      *(2026-07-16 research)*
      *(2026-08-06 research)* Corroborated from the CUDA side, which matters because it means the design is
      the *general* answer and not a WebGPU workaround: production int4 kernels (Marlin and descendants)
      fuse dequantization into the matmul so the int4 values unpack to f16 **in the register file**, never
      materializing a full-size intermediate — the same "straight into registers" split LlamaWeb measured
      for the decode matvec. Also worth copying from LlamaWeb is its *interface*: a new quantization scheme
      is an **unpacker + a dequant routine**, and every downstream kernel (matmul, attention) stays
      format-agnostic — the shape that let one representation carry 21 formats, and the natural fit for our
      GGUF reader, which already parses every K-quant block layout structurally. One caveat sharpens with
      this run's numbers: LlamaWeb's headline decode win came from being bandwidth-bound, and our f32 path
      is not — but our **f16** path now decodes 2.9× faster than f32, so measure the keep-quantized win
      against f16, not f32, or it will look better than it is. —
      https://www.tensortonic.com/llm-internals/quantization
      *(2026-08-09 research)* Third independent corroboration, this time from a shipped WebGPU product
      rather than a paper: PrismML's 1-bit 27B WebGPU runner describes the same split — hand-written
      WGSL matmul kernels with **fused dequantization in shared memory, weights never materialized as
      fp16 in VRAM**. Three sources (LlamaWeb on WebGPU, Marlin-class CUDA int4 kernels, this) now
      agree on the same shape, so the design question for our P9 kernel is settled and only the
      *substrate* choice is open (hand-written CubeCL kernel vs the CubeCL quantization primitives in
      the item below). Also worth watching for the KV half above: llama.cpp's TurboQuant discussion
      (#20969) is the current state of extreme KV-cache quantization —
      https://essamamdani.com/blog/prismml-bonsai-27b-1-bit-27b-model-runs-phone-webgpu-july-2026 ·
      https://github.com/ggml-org/llama.cpp/discussions/20969
- [ ] Evaluate **CubeCL's quantization primitives** for the keep-quantized matmul: recent CubeCL ships
      block-scaled MMA, global quantization for matmul, quantized tensor views, and FP4/FP2 formats —
      the kernel substrate a Q4-weights × f16-activations decode path would ride (vs hand-writing a
      dequant-fused kernel); gate any adoption on the parity harness + `bench/BASELINE.md` —
      https://github.com/tracel-ai/cubecl/releases · https://burn.dev/blog/release-0.21.0/
- [ ] **KV-cache quantization (FP8/e4m3)** — quantize the KV cache (and optionally the QK/ScoreV attention
      matmuls) to 8-bit, halving per-token cache footprint — the *other* VRAM lever besides weights, and the
      one that grows with context length. vLLM shipped exactly this (April 2026) and published the lessons
      that transfer: uncalibrated per-head e4m3 scales recover 97%+ on reasoning tasks and 94–98% AUC at
      1M-token contexts; **two-level accumulation is critical** (intermediate f32 writes on long contexts —
      the same failure our f32-softmax island guards); layer-selective beats uniform (sliding-window layers
      pay overhead for no benefit); head_dim 256 loses at prefill (~1.6× register pressure) while 64/128 win.
      For Mummu: our KV cache is f16 on `GpuF16` — an e4m3-quantized cache would halve it again; gate on the
      parity harness + budgets like every numeric change. — https://vllm.ai/blog/2026-04-22-fp8-kvcache
      *(2026-07-24 research)*
- [ ] **KV-cache persistence (warm session reopen)** *(colibri parity)* — colibri persists compressed KV
      state to disk (`.coli_kv`) so reopening a conversation skips the whole prefill; its 57× compression
      comes from MLA, which is architecture-specific (DeepSeek-family latent attention — not our GQA
      models), but the *persistence* transfers as-is and compounds with the FP8 item above (an e4m3
      cache is already half the bytes to write/read). Mummu shape: serialize the per-layer KV (+
      conv-state for hybrids) cache keyed by (model id + revision, dtype, prompt-prefix token hash);
      on reopen, load + verify the key and resume decode at the suffix. For the apps this converts
      long-system-prompt agent restarts from a full prefill (~100 ms/KB-of-prompt and growing with
      context) into a file read. Gate: a resumed decode must be byte-identical to the uninterrupted
      one (the cache-equivalence proofs extend naturally to a serialize/deserialize round-trip).
      *(2026-07-30 research)* — https://github.com/JustVugg/colibri
- [ ] **Auto-quantize-to-fit** — the planner picks the *highest* precision that fits the detected VRAM
      (f16 → int8 → int4), reports the quality/size trade, and never silently ships a worse tier than asked.

### P10 — Training & adapters
*(2026-08-21, fleshed out by the mistral.rs scan — was one line.)*
- [ ] **LoRA adapter inference** *(mistral.rs parity)* — load a HF PEFT adapter
      (`adapter_config.json` + `adapter_model.safetensors`) onto a zoo model. Two application modes,
      both wanted: **merge-on-load** (W ← W + (α/r)·B·A at import — zero per-token cost; the default
      "no config" path) and **unmerged** (keep A/B beside the base matmul — a small per-token cost
      that buys instant swap, and it is the representation the fine-tune loop below trains in). Gate:
      merged-load logits byte-identical to a reference `peft` `merge_and_unload` export of the same
      adapter; unmerged ≡ merged to summation-order tolerance.
- [ ] **Adapter hot-swap** *(mistral.rs parity)* — mistral.rs loads/unloads named adapters at runtime
      and routes per-request without dropping the base model. The Mummu shape already exists:
      `ModelSlot`'s drop-then-swap generalizes to swapping only the delta while the base weights stay
      resident — unmerged adapters swap by tensor replacement, merged ones pay a re-merge. Per the
      North Star this is a library call (`with_adapter`), not a server route; per-request LoRA
      *serving* stays consumer glue.
- [ ] On-device fine-tune loop (Burn supports training) — learn-by-format / personalization
      on-device, producing exactly the PEFT-shaped artifacts the two items above consume.

### P11 — Vision & OCR (retire Candle)
- [ ] Port a vision/OCR model (DeepSeek-OCR — currently Candle in laurelane, `physics515/deepseek-ocr.rs`)
      to Burn on the same runner, so Candle can be dropped from consumers entirely.
- [ ] **Qwen3.5-4B is multimodal** — `unsloth/Qwen3.5-4B-GGUF` ships an `mmproj-BF16.gguf` (a CLIP-style
      vision projector) beside the text GGUF, and its text tower is the **`qwen3` dense arch Mummu now
      runs + parity-verifies**. That makes it a strong second vision candidate whose LLM half is already
      done: a P11 port would be the mmproj vision encoder + the projector, feeding embeddings into the
      existing Qwen3 decoder — far less new surface than a from-scratch OCR model. Weigh against
      DeepSeek-OCR once the P3 GGUF-mmproj parse + a vision block land. *(2026-07-17 research)*

## Colibri parity scan *(2026-07-30)*

[colibri](https://github.com/JustVugg/colibri) is a pure-C, zero-dependency inference engine whose thesis
is running frontier **MoE** models (GLM-5.2 744B-A40B reference; Kimi K3 2.8T) on consumer hardware by
treating VRAM/RAM/**NVMe** as one memory hierarchy — int4 dense parts resident, routed experts streamed
from disk behind an LRU + learned hot-pinning + one-ahead prefetch. Same North-Star hardware class as
Mummu (a 128 GB box with one consumer GPU), opposite bet: Mummu shrinks the model to fit the fast tiers
(quantize-to-fit), colibri keeps the model huge and moves the working set. The scan spawned one item per
gap, each gated by the usual parity + budget discipline:

- **P2 — first MoE architecture** (OLMoE-7B-A1B / Qwen3-30B-A3B): router + expert blocks + GGUF expert
  tensors; the prerequisite for everything expert-shaped. **Closed 2026-08-03** — OLMoE-1B-7B ported and
  parity-verified vs llama.cpp; the expert-streaming and routed-compute items below now have their
  architecture.
- **P5 — speculative decoding** (MTP heads / small-model drafts, byte-identical by construction) and
  **grammar-constrained decoding** (guaranteed-parseable tool calls — worth more to the apps than tok/s).
- **P6 — NVMe as a third placement tier** (expert streaming; the headline feature) and **planner
  introspection** (`plan`/`doctor` as API).
- **P9 — KV-cache persistence** (warm conversation reopen; MLA's 57× is architecture-specific, the
  persistence isn't).

**Already at or past parity:** token-exact validation (our P7 gates are the same discipline, run per
port), quantized import (P3 GGUF dequant covers colibri's int4 tier; keep-quantized is the open P9 half
on both sides' terms), CPU path (burn-flex ≈ their OpenMP), model-format-agnostic loading (P3 is
data-driven by design). **Non-goals, deliberately:** the `serve`/`chat`/`web` surfaces (OpenAI-compatible
server, TUI, dashboard, Tauri wrapper) are consumer glue per the contract below — Nanna/laurelane own the
app surface, Mummu stays the library; and no C rewrite — parity here means *capabilities on Burn*, never
a second runtime. Dual-SSD mirroring, `O_DIRECT`, and NUMA interleave are streaming-tier tuning knobs to
revisit only if/when the P6 disk tier exists and measures I/O-bound.

## mistral.rs parity scan *(2026-08-21)*

[mistral.rs](https://github.com/EricLBuehler/mistral.rs) (v0.8.2) is the closest Rust analog in ambition —
"run any HF model, quantized, on whatever hardware" — and the opposite architecture in nearly every
choice: Candle + per-vendor builds (CUDA/Metal/MKL feature splits, NCCL tensor parallelism) where Mummu is
Burn/wgpu one-binary; a serving stack (OpenAI **and** Anthropic HTTP APIs, web UI, a server-side agentic
loop with sandboxed code/shell execution, web search, skills, MCP client + server) where Mummu is a
library; and a large knob surface (`--arch` overrides, per-layer topology TOML, `tune`/`doctor` CLIs, ISQ
flags) where Mummu's North Star is **one pathway, no config, auto-optimized for the hardware it lands
on**. So parity here is *capability* parity, translated: every feature worth having arrives as something
the importer or the P6 planner decides automatically and the parity harness gates — never as a flag the
caller must know to pass. mistral.rs's `tune` CLI *recommends* settings; Mummu's planner is the same
judgement *applied*. The scan spawned one item per gap, each tagged `*(mistral.rs parity)*` in its phase:

- **P2** — **RoPE scaling + sliding-window attention** (the silent-wrong-answer gap: a `rope_scaling` or
  `sliding_window` checkpoint currently loads clean and degrades quietly past the parity probes' reach);
  **Llama-family port** (the loader that multiplies checkpoint coverage most per unit of new surface);
  **Qwen3-Embedding** (a modern 32k embedder that IS the already-ported qwen3 arch + last-token pooling).
- **P3** — **architecture auto-detection** (dispatch off the checkpoint's own `config.json` /
  `general.architecture` so any supported-arch checkpoint runs by path alone — the "no config" door
  mistral.rs opens with auto-detect and Mummu currently gates behind the 12-entry catalog). GPTQ/AWQ
  import was already open; the compressed-tensors reader plan stands.
- **P5** — **sampling breadth** (min-p / penalties / logit bias — the request surface consumers will
  forward); **batched forward** (the honest subset of continuous batching: prerequisite for the
  spec-decode verify step and bulk embedding, and unusually valuable on a dispatch-bound profile);
  **in-memory prompt-prefix KV reuse** (their prefix cache — the warm sibling of P9's persistence item);
  plus a production-corroboration note on the grammar item (mistral.rs ships structured output on
  llguidance, the substrate already picked here).
- **P6 / P9** — nothing new to spawn, which is itself a finding: ISQ ≙ **auto-quantize-to-fit** (P9,
  open), auto device mapping ≙ the **placement plan** (P6, open), FP8 KV cache and keep-quantized
  weights were already the P9 backbone, PagedAttention-class block KV management only earns its
  complexity at batch > 1 and so waits on the P5 batching item, and their MTP-only speculative decoding
  independently confirms the P5 route-(a) conclusion (native heads, never a second model on consumer
  VRAM).
- **P10** — fleshed out from one line into **LoRA inference** (merge-on-load + unmerged), **adapter
  hot-swap**, and the fine-tune loop that produces what those consume.

**Already at or past parity:** chat-template handling (their auto-detect ≙ `tok_config` detection, and
the byte-gate vs `transformers.apply_chat_template` is a stronger claim than any engine makes);
tool-call render + parse (Hermes AND LFM Pythonic, auto-detected from the checkpoint); GGUF import
breadth (F32/F16/BF16, legacy quants, Q2_K–Q6_K superblocks); tokenizer import (`tokenizer.json`, GGUF
metadata, both SentencePiece proto types); streaming + cooperative cancellation at the library seam;
resumable sha256-verified hub fetch; warm-up API; true-VRAM device inventory. **Non-goals,
deliberately** (the consumer contract below already draws this line): the entire serving surface
(OpenAI/Anthropic HTTP, web UI, interactive CLI, Prometheus metrics) and the entire server-side agentic
layer (code/shell sandboxes, approval flows, web search, skills, MCP) — that is Nanna's half of the
split, powered by Mummu primitives (grammar-guaranteed tool calls, prefix reuse, adapters); UQFF-style
bespoke artifacts (GGUF + safetensors are the ecosystem; a third format is surface without capability);
X-LoRA / AnyMoE (no consumer demand — revisit if one appears); image generation and speech (FLUX / Dia /
Voxtral — a different product); multi-node distribution (the North-Star machine is one box); and every
config knob whose job the planner can do (`--arch`, topology TOML, per-request quant hints — each exists
as an item above precisely so the answer can be automatic instead).

## Consumer integration contract

Each app depends on Mummu (path/git dep) and keeps its own glue:
- **Nanna** — wire Mummu as `Provider::Local` (top-priority tier in the complexity router); stream tokens
  to channels + Tauri events; use Mummu embeddings for the memory `embed_fn` and dreaming `summarize_fn`.
- **laurelane** — replace `src/llm_burn.rs` / `src/lfm2_burn.rs` / `src/categorize_burn.rs` with Mummu;
  keep statement structuring, the reconciliation-gated extraction oracle, and the learning categorizer app-side.

## References

- Burn — https://burn.dev · https://github.com/tracel-ai/burn ; `burn-store` safetensors; CubeCL kernels.
- wgpu f16 `SHADER_F16` polyfill — https://github.com/gfx-rs/wgpu/pull/7884
- laurelane's proven Burn slices (Qwen2 / LFM2.5 / MiniLM, byte-identical parity vs Candle) — the extraction source.
- Function-calling small models (2026): Qwen 3.5-9B / Qwen3-A3B MoE + expert CPU-offload; local tool-call reliability.
- Burn multi-device (thread-safe; CUDA / ROCm / Metal / Vulkan / WebGPU + SIMD CPU — primitives, not automatic
  tensor-parallel), Burn 0.20 — https://www.phoronix.com/news/Burn-0.20-Released
- Quantization formats 2026 (GGUF K-quants Q2_K–Q8_0, GPTQ, AWQ, int4/int8; GGUF CPU+GPU hybrid) —
  https://www.digitalapplied.com/blog/gguf-vs-awq-vs-gptq-vs-mlx-llm-quantization-formats-2026
- Burn 0.21 (May 2026): 8× lower framework overhead, differentiable collectives, burn-dispatch, faster
  GEMV/top-k, per-tensor + per-block int8/int4/int2 quant on some backends — https://burn.dev/blog/release-0.21.0/
- wgpu f16 IO polyfill on Vulkan (SHADER_F16 without storageInputOutput16), wgpu PR #7884 — confirmed
  live on the dev box: Vulkan advertises SHADER_F16 on the 4070 Ti SUPER, DX12 does not (2026-07-09 probe).
- colibri — pure-C MoE weight-streaming engine (VRAM/RAM/NVMe hierarchy, MTP speculation, GBNF grammars,
  persistent compressed KV); the 2026-07-30 parity scan's source — https://github.com/JustVugg/colibri
- mistral.rs v0.8.2 — Candle-based Rust serving stack (~50 archs, ISQ/GGUF/GPTQ/AWQ/FP8, PagedAttention,
  MTP speculation, llguidance structured output, LoRA hot-swap, OpenAI+Anthropic APIs, agentic loop); the
  2026-08-21 parity scan's source — https://github.com/EricLBuehler/mistral.rs · https://ericlbuehler.github.io/mistral.rs/
