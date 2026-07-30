# Mummu

**A from-scratch, single-binary local model runner in Rust — import any model, quantize it to fit, and run it on all the hardware you have: CPU, one GPU, or several. No Ollama, no cloud, no CUDA toolchain.**

Mummu imports models from HuggingFace (or disk), **(auto-)quantizes** them to fit, and runs them **natively in Rust on [Burn](https://burn.dev)** across every device you have. One binary — a runtime probe inventories your GPUs (Vulkan / DX12 / Metal via `wgpu`) and your CPU (`burn-flex`) and places the model to use them **to the fullest**: several GPUs together, with CPU offload when VRAM is short, no feature-split builds and no CUDA toolchain. Models are **reimplemented from scratch**, generic over the Burn backend, and **parity-tested byte-for-byte** against a reference so the reimplementations can be trusted.

It exists because two local-first apps — **[laurelane](https://github.com/physics515/laurelane)** (a private budgeting cockpit) and **[Nanna](https://github.com/physics515/Nanna)** (an always-on local AI presence) — were building the *same* runner twice. Mummu is that runner, extracted and generalized: each app consumes it as a dependency and keeps only its own domain glue. Laurelane proved the blueprint (Qwen2.5 / LFM2.5 / all-MiniLM ported to Burn, byte-identical parity vs Candle, validated on an RTX 4070 Ti SUPER 16 GB); Mummu is where it lives, hardens, and grows.

## What it is

- **One binary, every device** — compile both `Wgpu` (Vulkan/DX12/Metal, no CUDA toolchain) and `burn-flex` (CPU); a runtime probe enumerates **all** adapters + the CPU and places the model across them — a single GPU, **several GPUs together**, or GPU + CPU hybrid. No feature-split builds, no per-vendor path.
- **Models from scratch, generic over `B: Backend`** — a growing zoo (Qwen2/2.5, Qwen3 dense with per-head q/k norm + decoupled head_dim, LFM2/2.5 hybrid conv+attention, all-MiniLM embedder) built on shared blocks (RmsNorm · GQA · RoPE · SwiGLU · tied lm-head · depthwise causal conv), with a clean trait to add more.
- **Trustworthy reimplementations** — every port must pass a **parity gate**: single-forward top-k logits *and* a short greedy sequence match a reference (Candle, or a local Ollama of the same model) exactly.
- **Fast** — per-layer KV cache (+ conv-state cache for hybrids), on-GPU argmax (sync only the winning index), sampling, **token streaming**, cooperative cancellation; kernel `fusion` + `autotune`; an **f16** path (f32 attention-score island for numeric safety) that halves VRAM at full speed.
- **A full model-import suite** — pull a model from HuggingFace (by repo id) or from disk and load it: **safetensors**, **PyTorch** state dicts, and **GGUF** (llama.cpp, dequantized) weights; `config.json`-driven hyperparameters; tokenizer + chat-template import (HF `tokenizers` / SentencePiece / BPE); per-architecture weight-name remapping with a **checked load** (fail loudly on a key mismatch, never silently zero-init); resumable, shard-aware downloads into a per-user cache; and a declarative **model registry** so adding a model is a manifest entry, not new code.
- **Quantize to fit, fill the hardware** — a planner probes every GPU + the CPU (VRAM / RAM), then imports or **quantizes on the fly** (GGUF K-quants, GPTQ / AWQ, or Burn's own int8/int4) and chooses precision + **layer placement** so the *largest model that fits* runs and every device is used — sharded across GPUs, spilling cold layers to CPU when needed. Plus a model-management API (download progress, disk usage, remove) apps surface in their settings UI.
- **Local embeddings** — a from-scratch MiniLM-class sentence embedder (CPU) for fully-offline semantic search.

## Status — what runs today

- **Workspace + backends** — `crates/mummu` (library) + `crates/mummu-bench` (criterion); one binary
  compiles both `Wgpu` (with `fusion` + `autotune`) and `burn-flex` (CPU), with a cached runtime GPU probe and a
  device inventory that records per-adapter/per-API `SHADER_F16`, max buffer size, and **true VRAM
  capacity** (DXGI on Windows; wgpu exposes no portable query), plus the host CPU's cores and total
  RAM — the planner's (and settings UIs') device set.
- **Shared blocks, generic over `B: Backend`** — cache-aware GQA attention (optional per-head q/k
  RMSNorm), manual RoPE, SwiGLU, and LFM2's double-gated causal short-conv with rolling decode state;
  unit tests prove prefill+decode ≡ full-forward for both cache kinds.
- **Checked safetensors + PyTorch import** — bf16→backend-float cast adapter, per-architecture key
  remaps, and a fail-loud load (never silently zero-init); `config.json`-driven hyperparameters.
  `pytorch_model.bin` state dicts load through the same checked path (safetensors preferred when both
  exist) — proven byte-identical on MiniLM's real Hub checkpoint in both formats.
- **`tokenizer_config.json` import** — `mummu::tok_config::TokenizerConfig` parses the conventions HF keeps
  beside `tokenizer.json`: the BOS/EOS/PAD/UNK special-token slots (id-resolved from `added_tokens_decoder`),
  the whole added-token map, `model_max_length`, and the raw Jinja `chat_template` (from the JSON key, or a
  standalone sibling `chat_template.jinja` when that key is absent — the layout recent `transformers` writes).
  Total and bounded
  (malformed input is a loud `ImportError::Parse`, never a panic); it doesn't render Jinja (prompt wrapping
  stays the byte-verified `chat` renderers) but gives apps a model's declared ids + template, and detects the
  template's tool-call convention (Hermes vs LFM) so the right render style is picked from the checkpoint.
  Two consistency validators catch repackaging bugs: `check_ids_against` (every added-token id must match the
  real tokenizer) and `check_eos_agrees` (`config.json`'s `eos_token_id` must match the resolved EOS).
  Cross-checked on real weights: all 26 of Qwen3-0.6B's added-token ids agree byte-for-byte with
  `tokenizer.json`, and its `config.json` EOS 151645 agrees with the resolved `<|im_end|>`. The safetensors
  loaders **enforce all of this at load**: `load_from_dir` (Qwen2/Qwen3/LFM2.5) runs the gate right after
  `config.json` parses and before any weights are read — a sibling `tokenizer_config.json` whose EOS
  disagrees with `config.json`, whose chat-template speaks a different tool-call convention than the
  family's renderer, or — when a `tokenizer.json` sits beside it — whose declared added-token ids don't
  match that real tokenizer, is a loud `ImportError::Inconsistent` instead of a model that silently
  mis-stops, mis-templates, or mis-tokenizes. Both sibling files are optional (a GGUF-derived dir has
  neither → no behavior change). On a successful safetensors load the parsed `TokenizerConfig` is surfaced
  on the returned `Loaded{Qwen2,Qwen3,Lfm2}` struct (`tokenizer_config`), so a consumer reads config-driven
  EOS/BOS/PAD ids straight off the model; a GGUF load surfaces `None` (self-contained).
- **Import validation** — a two-stage error taxonomy: `ImportError` for the file→module stage (missing
  file, parse, load, and an `Incomplete` per-tensor missing/errored diff) and `SanityError` for the
  runtime liveness a checked load can't see — NaN/Inf logits, a vocab-width mismatch, or a
  degenerate/dead forward. `CausalLm::sanity_check` is the post-`install` gate an app calls to catch a
  silently-broken import before trusting the model.
- **Four architectures ported and running on real weights** — Qwen2/2.5, Qwen3 dense, the LFM2/2.5
  hybrid, and the all-MiniLM sentence embedder; Qwen2.5-1.5B, Qwen3-0.6B, and LFM2.5-1.2B/230M load and
  greedy-decode correctly on the reference GPU (wgpu/Vulkan), and Qwen2.5-0.5B / LFM2.5-230M do the same
  on the CPU backend. Qwen3 reuses the shared blocks whole (its per-head q/k RMSNorm, absent qkv bias,
  and decoupled `head_dim` were all already supported), loads from safetensors **and** a single Q4_K_M
  GGUF, and handles Qwen3's `<think>` reasoning mode.
- **All three models are parity-verified** — the two-leg P7 gate passes for Qwen2.5-1.5B on the
  reference GPU: single-forward top-5 logits match a Candle f32 reference (max |Δlogit| 2.7e-5,
  `tests/parity_qwen2.rs` + the committed `tools/candle-probe` fixture) and a 24-token greedy sequence
  matches `ollama qwen2.5:1.5b-instruct-fp16` byte-for-byte. **LFM2.5-1.2B** passes both legs against a
  same-weights llama.cpp reference (`tests/parity_lfm2.rs` + the `tests/llama_ref` harness: a local
  `llama-server` on LiquidAI's official BF16 GGUF, raw `/completion`, prompts as token-id arrays):
  top-5 first-forward ids match exactly in order and a 24-token greedy sequence is byte-identical.
  **LFM2.5-230M** passes the same two legs through the same tier-parameterized gate (top-5 ids exact
  in order, greedy byte-identical, max |Δlogprob| 3.2e-2) — one config-driven hybrid loader covers
  both tiers. **Qwen3** is parity-verified through the same llama.cpp harness (`tests/parity_gguf.rs`,
  `qwen3` leg): on Qwen3-0.6B Q4_K_M, top-5 first-forward ids match exactly in order and a 24-token
  greedy sequence — `<think>` reasoning tokens included — is byte-identical to `llama-server` on the
  same file. The MiniLM embedder matches its Candle reference at cosine 0.99999994
  (max |Δcomponent| 1.2e-7, `tests/real_minilm.rs`).
- **Sampling, streaming, cancellation** — temperature / top-k / top-p sampling (deterministic per seed),
  per-token streaming through a `ControlFlow` callback, and cooperative between-token cancellation;
  greedy decoding keeps the argmax on-device.
- **Function calling (both zoo conventions)** — advertise `ToolSpec`s through `render_with_tools` in
  the convention the model was trained on: **Hermes** for Qwen2.5/Qwen3 (the exact
  `# Tools`/`<tool_call>` JSON template, results as merged `<tool_response>` turns) and **LFM** for
  LFM2.5 (bare tool JSON on a `List of tools:` system line, Pythonic calls in
  `<|tool_call_start|>` tokens, results as real `tool` turns, past-turn `</think>` stripping); both
  parsers are bounded with a loud error taxonomy. Proven end-to-end on the real GPU: Qwen2.5-1.5B
  emitted a parseable Hermes call, LFM2.5-1.2B emitted exactly
  `<|tool_call_start|>[get_weather(city="Paris")]<|tool_call_end|>`, and Qwen3-0.6B — from a
  `ChatMl::qwen3()` prompt selected by its own imported template's convention — emitted a
  `<think>` block plus a parseable Hermes call (`tests/real_toolcall.rs`,
  `tests/real_toolcall_lfm.rs`, `tests/real_toolcall_qwen3.rs`).
- **Template byte gate** — the hardcoded renderers are proven **byte-identical to
  `transformers.apply_chat_template`** rendering the checkpoint's own imported `chat_template`
  (via the `hf-chat-template` dev-dependency): plain, multi-turn, the full Hermes `# Tools` block,
  and function-call history match byte-for-byte on **Qwen3-0.6B, Qwen2.5-1.5B, and LFM2.5-1.2B**
  (LFM's legs cover both tool conventions, its `tool` role turns, its history think-stripping, and
  the standalone `chat_template.jinja` import path), `tests/template_gate.rs`. **`ChatMl::qwen3()`**
  carries Qwen3's two template deltas exactly — `<think>` reasoning stripped from assistant turns
  at/before the last user query (kept and re-normalized for later turns mid tool loop), and no default
  system preamble with tools — all three byte-equal against the imported template. The one remaining
  family divergence is pinned to its exact delta so any other drift fails loudly: Qwen2.5's no-system
  branding preamble vs `qwen2()`'s neutral one.
  Prompt JSON deliberately serializes with Python `json.dumps` spacing and insertion-order keys
  (serde_json `preserve_order`) — the exact bytes the reference stack renders and models emit back.
- **f16 inference, validated** — Qwen2.5-1.5B runs coherently on `GpuF16` (weights + KV in f16, the
  q·kᵀ attention scores + softmax computed in an f32 island to stop f16 overflow): **~3.6 GiB runner
  VRAM vs ~7.9 GiB f32, at identical speed**; the parity gate re-passes unchanged on f32, where the
  island casts are no-ops ([bench/BASELINE.md](bench/BASELINE.md)). The same island covers the Qwen3
  arch — Qwen3-0.6B decodes coherently in f16 (its qk-norm + decoupled head_dim ride the same f32 scores).
- **In-process mixed precision is defined behavior** — every runtime tensor-creation site pins its
  dtype to the backend type (`backend::{float_dtype, int_dtype}`), so an f32 (`Gpu`) and an f16
  (`GpuF16`) model can share one process and one device regardless of Burn's first-touch-locked
  per-device dtype policy — proven on the real GPU in the historically failing order
  (`tests/real_mixed_dtype.rs`: f16 locks the policy first, the f32 model still forwards f32 logits,
  agrees on the greedy top token, and decodes coherently).
- **SPIR-V kernels on Vulkan** — CubeCL compiles direct SPIR-V (burn's `vulkan` feature) instead of
  WGSL/naga on Vulkan adapters, worth **+30% decode throughput** on the reference GPU with parity
  byte-identical; other APIs (DX12/Metal) transparently keep WGSL in the same binary.
- **Benchmarked** — Qwen2.5-1.5B on the reference GPU: **TTFT 96.7 ms, decode 18.4 tok/s** (f32,
  11.5 GiB whole-card peak ≈ 8.0 GiB runner; f16: 97.2 ms, 18.4 tok/s, ~3.6 GiB runner) — recorded with
  budgets in [bench/BASELINE.md](bench/BASELINE.md), enforced by an opt-in regression gate
  (`mummu-bench/tests/budget.rs`).
- **Model management** — `ModelManager` gives settings UIs the whole lifecycle over a declarative model
  catalog (`registry::ModelSpec`): install with per-chunk download progress, `is_installed`, per-model
  disk usage, and traversal-safe removal; model switching rides `ModelSlot`.
- **GGUF import, end to end** — `mummu::gguf` parses the llama.cpp container (typed, bounded metadata;
  fully validated tensor table) and dequantizes **every storage dtype** (F32/F16/BF16, the legacy
  Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 blocks, and the Q2_K–Q6_K superblocks) to f32; `qwen2::load_from_gguf`
  turns the one file into a running model — hyperparameters from the GGUF metadata, weights bridged
  through the same checked-load pipeline as safetensors (tied *and* untied lm-heads). Proven against
  the model's true weights (`tests/real_gguf.rs`): F32 norms **bit-exact** vs the bf16 safetensors of
  the same checkpoint, Q4_K rows at cosine 0.9975, and the real Qwen2.5-1.5B **Q4_K_M file greedy-decodes
  "2+2 equals 4." on the GPU** with first-token top-1 identical to the bf16 build (logit cosine 0.977).
  The tokenizer comes from the GGUF too (`tokenizer_from_gguf`: per-family pre-regex → ByteLevel → BPE,
  byte-identical ids vs the checkpoint's `tokenizer.json`) — **one .gguf file is the whole model**.
  Works for the **LFM2.5 hybrid** as well (`lfm2::load_from_gguf`: layer kinds from the per-layer
  kv-head array, conv kernels un-squeezed bit-exactly): the official LiquidAI Q4_K_M greedy-decodes
  "2 + 2 equals 4." with top-1 identical to bf16 (logit cosine 0.991). Next: keep-quantized VRAM
  (tracked in P9).
- **SentencePiece `tokenizer.model` import, both proto types** — `tokenizer_from_spm` builds the HF
  pipeline straight from the SPM proto the Llama/Gemma/T5 families ship (a bounded hand-rolled
  protobuf reader, zero new dependencies). **Unigram** protos (T5/ALBERT/Gemma) get the `Precompiled`
  charsmap + whitespace-collapse normalizers, Metaspace, and a Unigram model; **BPE** protos (Llama-2
  family) get their merge list reconstructed from vocab + scores (HF's `SentencePieceExtractor`
  algorithm) plus the `Prepend`/`Replace` normalizers and `ByteFallback`/`Fuse`/`Strip` decode chain.
  The proto's specials are re-added and id-verified in both. Proven **byte-identical** to the same
  checkpoints' shipped `tokenizer.json` — ids and decode round-trips — on flan-t5-small (Unigram) and
  TinyLlama-1.1B (BPE, 61k reconstructed merges, byte-fallback cases included) across a
  unicode/whitespace/CJK/emoji battery (`tests/real_spm.rs`).
- **Hub downloads** — streaming HuggingFace fetches into the model cache: resumable (`.part` + HTTP
  Range, proven byte-identical after an interrupted transfer), length-verified, shard-index aware, with
  a per-chunk progress callback; verified end-to-end by downloading all-MiniLM and embedding with it.
- **Process-lifetime model cache** — `ModelSlot` loads a checkpoint once per process, switches models by
  key, and `clear()`s to free VRAM; Burn's `Param` isn't `Sync`, so access serializes behind its mutex.

## Design principles

- **Local-first, offline, private** — your own hardware is the whole story; the cloud is never a dependency.
- **Use all the hardware** — inventory every GPU and the CPU and run the model across them to the fullest (multi-GPU + CPU offload); quantize to fit the VRAM you actually have. Great on a laptop CPU, better on one GPU, best on several — same binary.
- **Burn at the core** — [Burn](https://burn.dev) is the one inference *and* training engine. Every model, every backend, and the (future) on-device fine-tune loop is built **from scratch on Burn** (via CubeCL) as a single backend-agnostic codebase (CPU / CUDA / Metal / Vulkan / WebGPU). No second runtime, no per-backend forks, no C/CUDA toolchain — Burn is the foundation the whole runner stands on.
- **Parity or it didn't happen** — a reimplementation ships only when it is numerically byte-identical to a reference.
- **Performance is a gate** — a change lands only when the parity + perf budgets (TTFT, decode tok/s, VRAM ceiling) hold; README perf claims link a benchmark artifact.
- **README + ROADMAP are the only docs** — shipped capability is described here; everything planned or next lives as a `[ ]` in [ROADMAP.md](ROADMAP.md); git history + PRs are the record.

## Consumers

- **[Nanna](https://github.com/physics515/Nanna)** — the runner *is* the agent: local inference for the whole agent loop, plus local embeddings for its memory. Wires Mummu in as `Provider::Local`.
- **[laurelane](https://github.com/physics515/laurelane)** — on-device statement structuring + categorization; Mummu replaces its in-app Burn modules.

Reference GPU: **RTX 4070 Ti SUPER 16 GB**. The plan lives in **[ROADMAP.md](ROADMAP.md)**.
