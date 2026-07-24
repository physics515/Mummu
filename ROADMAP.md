# Mummu — Roadmap

> The single source of truth for Mummu — a from-scratch **Burn** model runner shared by
> [laurelane](https://github.com/physics515/laurelane) and [Nanna](https://github.com/physics515/Nanna).
> The only docs are this file + [README.md](README.md): shipped capability becomes a FEATURE in the
> README (perf claims link a benchmark artifact); everything not-done / discovered / next is a `[ ]`
> here; git history + PRs are the record. Edit surgically; never rewrite wholesale.

**Stack:** Rust 2024 · **Burn 0.21** (`wgpu` 29 + `burn-flex` CPU, `fusion` + `autotune`, multi-device) ·
`burn-store` · HF `tokenizers` · runs on **any hardware** — CPU, one GPU, or several (multi-GPU + CPU
offload). Reference dev machine: Ryzen 9 7950X3D · 128 GB · RTX 4070 Ti SUPER 16 GB.
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
`cargo upgrade`.*

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
      budgets are unmoved by its presence. *(2026-07-24)* **Shipped exactly that shape**, exploiting the
      discovery rule read from cubecl-runtime 0.10 source (`RuntimeConfig::from_current_dir` walks UP
      from the process CWD, `cubecl.toml` checked before `burn.toml` at each level, first hit wins; cargo
      runs each crate's tests/benches with CWD = the package dir): a repo-root **`burn.toml`** sets
      `[cubecl.compilation] check_mode = "validate"` (bounds-check every launch AND validate
      explicitly-unchecked kernels for OOB) so the `crates/mummu` parity/real-model suites run armed,
      and **`crates/mummu-bench/cubecl.toml`** opts the budget/bench crate back to the `auto` default so
      recorded numbers never carry validation overhead. Consumers run from their own CWD — untouched.
      Verified live: an A/B with a malformed root burn.toml makes a GPU test fail at config load (proof
      the file is discovered + parsed), the real-model GPU suite passes with validation armed (no OOB
      found — clean bill), and the budget gates hold their numbers from the opted-out bench crate.
- [ ] Evaluate **CubeCL's now-complete flash-attention kernel** for the decode/prefill attention step —
      the releases page reports a full implementation (causal **masking**, partitions, row-wise
      reductions, multi-plane ops). Mummu currently materializes attention explicitly (q·kᵀ → f32 softmax
      island → ·v); a fused flash-attention kernel collapses those into one dispatch, which is squarely
      the **fewer-kernels-per-step** lever the dispatch-bound decode note above is chasing (and it drops
      the O(t²) scores tensor at prefill). Gate strictly: the f32-softmax island is the whole reason the
      f16 parity holds, so any flash path must re-pass the parity harness (both legs) AND hold/beat
      `bench/BASELINE.md` before adoption. — https://github.com/tracel-ai/cubecl/releases *(2026-07-17 research)*

## Phases

### P0 — Workspace scaffold
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
- [ ] **Qwen3.5 small tier** as the next zoo port: released Feb 2026 in 0.8B / 2B / 4B / 9B, with
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
      `pytorch_model.bin` fallback.
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
      133 unit tests total); `GgufFile::dequant_to_safetensors` bridges a GGUF onto the SAME checked-load
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
- [ ] **GPTQ / AWQ** (HF safetensors) — import the calibration-quantized int4/int8 layouts most "quantized on
      the Hub" models ship as (a `.safetensors` payload + a quant config), dequant or keep-quant into Burn.
      *(2026-07-13 research)* Both are quantization *algorithms*, not formats — the artifact is ordinary
      safetensors shards (packed int4 `qweight`/`qzeros`/`scales`, group size 128 is the de-facto
      standard) plus `quantization_config` in `config.json`; vLLM's **compressed-tensors** is the
      emerging unified on-disk convention to target (one reader covers GPTQ/AWQ/INT8/FP8 exports) —
      https://github.com/vllm-project/compressed-tensors ·
      https://www.digitalapplied.com/blog/gguf-vs-awq-vs-gptq-vs-mlx-llm-quantization-formats-2026
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
- [ ] **Tokenizer + chat-template import** — HF `tokenizer.json` (fast), SentencePiece `tokenizer.model`, BPE
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
- [ ] **SentencePiece `tokenizer.model` import** — the `.model` proto tokenizer (Llama/Gemma/T5 family) that
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
- [ ] **Loaders open the tokenizer for `check_ids_against` + config-driven BOS** — the 2026-07-19 gate is
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
      *(2026-07-24)* **Payoff (1) shipped — the byte gate is real and PASSES 9/9.** Two build decisions:
      (a) hf-chat-template 0.2.1 is NOT a dev-dependency — it enables serde_json's `preserve_order`
      feature, and features being additive that would silently flip our own test builds from serde_json's
      default (sorted) map order to insertion order, diverging the very renders the tests verify from what
      plain-feature consumers build; it lives in **`tools/template-probe`** (own workspace, the
      candle-probe pattern), a 60-line bin: argv = template file, stdin = RenderInput JSON, stdout = the
      transformers-identical render. (b) The gate exposed a REAL byte divergence and it is now FIXED:
      our renders embedded tool JSON compact (`{"a":1}`) where every checkpoint template runs Jinja
      `tojson` = Python `json.dumps` separators (`{"a": 1}`) — and the models emit that spacing back
      (the 07-11 real-GPU proof literally logged `{"name": "get_weather", "arguments": {"city":
      "Paris"}}`). `chat::py_json` (a `json.dumps`-separator serde Formatter, public — consumers
      composing history turns need the same spelling) now feeds all template-embedded JSON.
      `tests/template_gate.rs` (ignored, env-keyed `MUMMU_TEMPLATE_PROBE` + the three checkpoint dirs)
      byte-compares our renderers against each checkpoint's OWN imported template (through
      `TokenizerConfig::from_dir`, incl. LFM's standalone `chat_template.jinja` — fetched into the local
      cache this run) rendered by the reference engine: **Qwen2.5 plain/tools/tool-call-history, Qwen3
      plain/tools, LFM2.5 plain±system/tools±system/think-stripping/pythonic-call+tool turns — all
      byte-identical**; the only divergences are pinned to their exact deltas (Qwen2.5's no-system
      branding preamble "You are Qwen, …" vs our neutral one, Qwen3's no-system no-preamble, Qwen3's
      history think-stripping) so any OTHER drift fails loudly. Payoff (2) — a general fallback renderer
      for family-less checkpoints — stays open below.
- [ ] **`ChatMl::qwen3()` with history think-stripping** — the byte gate documented that Qwen3's template
      strips `<think>…</think>` reasoning from assistant turns at/before the last user query while our
      shared `ChatMl::qwen2()` renderer re-renders history verbatim (fine for fresh prompts + tool loops,
      wrong for long multi-turn chats with a thinking Qwen3). A `qwen3()` constructor wants the LFM-style
      strip (the machinery exists — `turn_content` already does it for `Lfm`) but keyed to Qwen3's
      "at/before the last user query" rule rather than LFM's "every but the last assistant turn"; gate it
      on the template byte gate's think case flipping from documented-divergence to byte-equal.
      *(2026-07-24, found by the byte gate.)*
- [ ] **General fallback chat renderer from the imported template** — payoff (2) of the hf-chat-template
      evaluation: for a checkpoint whose family has no hardcoded renderer, render prompts from its own
      imported `chat_template` (the byte gate now proves the engine is transformers-identical on our
      three families). Weigh the minijinja dependency against the from-scratch ethos — as a *runtime*
      feature it puts a Jinja engine on the prompt path, so it wants an explicit opt-in surface (e.g.
      `chat::from_template(&TokenizerConfig)`) and the preserve_order caveat re-examined (the probe
      sidesteps it today by living out-of-workspace). *(2026-07-24, split from the evaluation item.)*
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
      *(2026-07-24) Template-embedded tool JSON now spells `json.dumps` separators (`chat::py_json` —
      `{"a": 1}`, not compact `{"a":1}`) in BOTH conventions: the P3 template byte gate proved the
      checkpoints' own templates (Jinja `tojson`) and the models' own emissions use that spacing, and
      the renders are now byte-identical to `transformers.apply_chat_template` — see the P3 gate item.*
      *(2026-07-10 research)* 2026 community numbers back the plan: Qwen3-8B keeps tool-calling score
      through Q4_K_M (0.919 quantized vs 0.933 full — quant does NOT cost tool reliability, good news for
      P9); BFCL shows a capability cliff below ~7B (Qwen3.5-9B 66.1% vs 4B 50.3%), so the zoo's
      function-calling tier should target the 7–9B class once quant lands; Hermes 4 (Qwen3-14B fine-tune)
      emits `<tool_call>` tags after an explicit reasoning step — easy to parse with the same template
      machinery — https://www.promptquorum.com/power-local-llm/best-local-models-tool-calling-2026 ·
      https://localaimaster.com/blog/best-ollama-models-for-agents

### P5 — Decode engine *(ex-laurelane)*
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
- [ ] **Precision selection** — pick a per-device dtype (f32 / **f16** / int8 / int4) that fits: f16 via
      `Wgpu<half::f16, i32>`; drop to int8/int4 (P9) when f16 still won't fit. *(2026-07-11) The f16
      backend itself is now **fully validated** (all 3 claims — see the islands item below); what remains
      here is the *picking* logic, which rides the placement-plan item + P9.*
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
- [ ] **One process cannot mix `Gpu` (f32) and `GpuF16` on the same device** — instantiating `GpuF16`
      flips Burn's per-device default dtype policy process-wide: a `Gpu` f32 forward that runs afterwards
      panics reading its logits back (`TypeMismatch: expected F16, got F32`). Found 2026-07-24 as a
      deterministic pre-existing failure of the old mixed `real_qwen3` suite (reproduced serially on
      unmodified HEAD, exonerating that run's changes); the test-side fix is the convention that every
      `GpuF16` leg lives in `real_f16.rs` (own process — qwen3's f16 leg moved there). But the LIBRARY
      hazard remains for consumers: the P6 precision-*selection* work (pick f16 vs f32 at runtime) must
      either pin one dtype alias per process, isolate per-device policy, or root-cause the Burn policy
      flip upstream before an app can switch precision on the fly.
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
- [ ] **Auto-quantize-to-fit** — the planner picks the *highest* precision that fits the detected VRAM
      (f16 → int8 → int4), reports the quality/size trade, and never silently ships a worse tier than asked.

### P10 — Training & adapters
- [ ] LoRA adapters; an on-device fine-tune loop (Burn supports training) — learn-by-format /
      personalization on-device.

### P11 — Vision & OCR (retire Candle)
- [ ] Port a vision/OCR model (DeepSeek-OCR — currently Candle in laurelane, `physics515/deepseek-ocr.rs`)
      to Burn on the same runner, so Candle can be dropped from consumers entirely.
- [ ] **Qwen3.5-4B is multimodal** — `unsloth/Qwen3.5-4B-GGUF` ships an `mmproj-BF16.gguf` (a CLIP-style
      vision projector) beside the text GGUF, and its text tower is the **`qwen3` dense arch Mummu now
      runs + parity-verifies**. That makes it a strong second vision candidate whose LLM half is already
      done: a P11 port would be the mmproj vision encoder + the projector, feeding embeddings into the
      existing Qwen3 decoder — far less new surface than a from-scratch OCR model. Weigh against
      DeepSeek-OCR once the P3 GGUF-mmproj parse + a vision block land. *(2026-07-17 research)*

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
