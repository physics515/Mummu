# Mummu — Roadmap

> The single source of truth for Mummu — a from-scratch **Burn** model runner shared by
> [laurelane](https://github.com/physics515/laurelane) and [Nanna](https://github.com/physics515/Nanna).
> The only docs are this file + [README.md](README.md): shipped capability becomes a FEATURE in the
> README (perf claims link a benchmark artifact); everything not-done / discovered / next is a `[ ]`
> here; git history + PRs are the record. Edit surgically; never rewrite wholesale.

**Stack:** Rust 2024 · **Burn 0.21** (`wgpu` 29 + `ndarray`, `fusion` + `autotune`, multi-device) ·
`burn-store` · HF `tokenizers` · runs on **any hardware** — CPU, one GPU, or several (multi-GPU + CPU
offload). Reference dev machine: Ryzen 9 7950X3D · 128 GB · RTX 4070 Ti SUPER 16 GB.

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
- [ ] `mummu-bench` (criterion) — TTFT, decode tok/s, VRAM per model × device-set (1 GPU / N GPUs / CPU / hybrid).
- [ ] `bench/BASELINE.md` budgets (VRAM ceiling, min decode tok/s, max TTFT) + a `cargo test` gate that fails a regression.

## Phases

### P0 — Workspace scaffold
- [ ] Cargo workspace: `crates/mummu` (the library), model code generic over `B: Backend`. `.gitignore`
      (Rust); commit `Cargo.lock` for reproducible builds/benchmarks.
- [ ] `cargo build` / `test` / `clippy --all-targets` green baseline; `mummu-bench` (criterion) crate stub.

### P1 — Backends & device *(ex-laurelane)*
- [ ] Backend abstraction generic over `B: Backend`; one binary compiling BOTH `Wgpu` (Vulkan/DX12/Metal,
      no CUDA toolchain) and `NdArray` (CPU).
- [ ] Runtime `use_gpu()` probe (`wgpu` adapter enumerate; `pollster::block_on` for the async probe),
      cached in a `OnceCell`; pick GPU else CPU. No feature-split builds.
- [ ] `fusion` + `autotune` on (`Wgpu` becomes `Fusion<Wgpu>`; needs `recursion_limit = 512`).

### P2 — Model zoo (from scratch, generic over `B`) *(ex-laurelane)*
- [ ] Shared blocks: RmsNorm, GQA attention, RoPE (manual rotate-half), SwiGLU MLP, tied lm-head,
      depthwise causal conv (for hybrids).
- [ ] **Qwen2 / Qwen2.5** decoder (1.5B / 0.5B tiers).
- [ ] **LFM2.5-1.2B** hybrid (6 GQA-attention w/ per-head q/k RMSNorm + 10 double-gated short-conv "LIV"
      blocks, SwiGLU, tied head, conv-state cache; ChatML, EOS `<|im_end|>`).
- [ ] **all-MiniLM** BERT sentence-embedder (6-layer post-LN bidirectional attention + GeLU FFN,
      masked-mean-pool + L2-normalize).
- [ ] A `Model` trait so new architectures (Hermes-class function-callers, Gemma, Qwen3, …) slot in.

### P3 — Model import suite (any source / any format → a running model)
The subsystem that turns "a model on HuggingFace or on disk" into a loaded, parity-checked Mummu model.
**Data-driven** — adding a model is a manifest entry, not new code. All import is Burn-native (`burn-store` /
`burn-import`).
- [ ] **Sources** — HuggingFace Hub (repo id + revision), local paths, and a bundled resources dir (checked
      first). Streaming download into a per-user cache: resumable (`.part`), integrity-checked, and
      **sharded-checkpoint aware** (read `*.index.json`, fetch + merge shards).
- [ ] **safetensors** *(ex-laurelane)* — `burn-store` `SafetensorsStore` + `PyTorchToBurnAdapter`; the primary path.
- [ ] **PyTorch state dicts** (`.pth` / `pytorch_model*.bin`) — for models not shipped as safetensors.
- [ ] **GGUF** (llama.cpp) — parse the GGUF container (metadata KV + tensor table), map tensors to modules,
      and **dequantize** Q4/Q5/Q8/K-quant blocks into Burn tensors (or hand keep-quantized to P9). GGUF is how
      most small models are distributed — this makes the whole ecosystem importable.
- [ ] **GPTQ / AWQ** (HF safetensors) — import the calibration-quantized int4/int8 layouts most "quantized on
      the Hub" models ship as (a `.safetensors` payload + a quant config), dequant or keep-quant into Burn.
- [ ] **ONNX** (optional) — `burn-import` ONNX→Burn for models distributed as ONNX graphs.
- [ ] **Dtype handling** — a `CastFloatAdapter` (bf16→f32/f16); quantized→dequant on import; keep-quantized
      handed to P9.
- [ ] **Weight-name remapping + checked load** — per-architecture key-remap tables (checkpoint naming →
      Mummu module names); **fail loudly** on missing/unexpected keys with a readable diff, never silently zero-init.
- [ ] **Config import** — parse `config.json` → model hyperparameters (layers, hidden, heads, kv-heads,
      rope-theta, vocab, tie-word-embeddings, …) so a model is **config-driven**, not hardcoded per checkpoint.
- [ ] **Tokenizer + chat-template import** — HF `tokenizer.json` (fast), SentencePiece `tokenizer.model`, BPE
      merges/vocab; special-tokens map + the chat template from `tokenizer_config.json`.
- [ ] **Model registry / manifest** — a declarative `ModelSpec` (repo, architecture, weight format, dtype,
      tokenizer, chat template, size tier) + a small built-in catalog of known-good models (Qwen2.5, LFM2.5,
      MiniLM, …); adding a model = a manifest entry.
- [ ] **Import validation** — checked load + a first-token parity smoke against a reference before a model is
      marked trusted; a clear error taxonomy (missing file, bad shard, key mismatch, unsupported dtype).

### P4 — Tokenizer & chat templates *(ex-laurelane)*
- [ ] HF `tokenizers` (pinned); explicit chat templates (ChatML + per-model), correct special/EOS tokens.

### P5 — Decode engine *(ex-laurelane)*
- [ ] Per-layer KV cache (+ conv-state cache for hybrids); prompt prefilled once, then one token/step.
- [ ] On-GPU argmax (sync only the winning index); single-token decode skips the causal mask.
- [ ] Sampling beyond greedy (temperature / top-p); **token streaming** via a callback/channel;
      cooperative interrupt/cancellation between tokens.
- [ ] Process-lifetime model + tokenizer cache (per backend; behind a `Mutex` since Burn `Param` isn't `Sync`).

### P6 — Hardware planner: precision, placement & full utilization
The "use all the hardware" phase — inventory the machine, then pick the precision and the device placement
that fits the model AND uses every device to the fullest.
- [ ] **Device inventory** — enumerate every GPU (`wgpu` adapters: name, backend, VRAM) and the CPU (cores,
      RAM); a stable device set cached at startup, reported so the apps can show it in settings.
- [ ] **Precision selection** — pick a per-device dtype (f32 / **f16** / int8 / int4) that fits: f16 via
      `Wgpu<half::f16, i32>` (needs wgpu ≥ 27 `SHADER_F16` polyfill — *laurelane compiles it + a startup
      `SHADER_F16` diagnostic; finish on-GPU validation here*: no naga crash, ~halved VRAM, coherent output);
      drop to int8/int4 (P9) when f16 still won't fit.
- [ ] **Placement plan** — given model size + KV-cache + display headroom and the device set, choose a
      **fit-and-fill** plan: single GPU when it fits; **shard layers across multiple GPUs** (pipeline/
      layer-parallel over Burn's multi-device tensors — Burn gives the multi-device *primitives*, not automatic
      tensor-parallel, so we place modules on devices ourselves); **spill cold layers to CPU** (GGUF-style
      hybrid) when total VRAM is short. Largest-model-that-fits, every device busy.
- [ ] **Multi-GPU execution** — run the sharded plan: per-device sub-modules, activations handed across the
      device boundary between stages, KV-cache per shard, and a micro-batch/pipeline schedule so the GPUs
      overlap rather than idle. *(Tensor-parallel within a layer is the stretch goal; layer/pipeline split is
      the tractable first cut.)*
- [ ] **`[hardware]` config + overrides** — auto by default; explicit overrides (device list, per-device layer
      counts, precision, CPU-offload cap) for power users.

### P7 — Parity & performance harness *(ex-laurelane)*
- [ ] Parity gate: single-forward top-k logits + a short greedy sequence must match a reference (Candle,
      or a local Ollama of the same model) — the trust gate every port passes.
- [ ] Wire the perf suite (above) into the parity harness so a correctness *or* budget regression fails CI.

### P8 — Model management API
- [ ] Download progress · disk usage · switch/remove models — an app-agnostic API the consumers' settings
      UIs call. *(laurelane has disk-usage + remove; add progress + active-model switch.)*

### P9 — Quantization (fit any model to the hardware)
The VRAM lever the P6 planner pulls to make the largest useful model fit the user's actual devices.
- [ ] **Burn-native quant** — Burn's `Quantizer` int8 / int4 (block-wise, ~4–8× weight reduction) on the wgpu
      + CPU paths; quantize on import or on the fly, keyed to the fit target from P6.
- [ ] **Import pre-quantized** — run **GGUF K-quants** (Q2_K–Q8_0, per-layer precision) and **GPTQ / AWQ** int4
      layouts directly (dequant to the compute dtype, or a keep-quantized matmul where the kernel exists), so a
      model already quantized on the Hub loads as-is.
- [ ] **Auto-quantize-to-fit** — the planner picks the *highest* precision that fits the detected VRAM
      (f16 → int8 → int4), reports the quality/size trade, and never silently ships a worse tier than asked.

### P10 — Training & adapters
- [ ] LoRA adapters; an on-device fine-tune loop (Burn supports training) — learn-by-format /
      personalization on-device.

### P11 — Vision & OCR (retire Candle)
- [ ] Port a vision/OCR model (DeepSeek-OCR — currently Candle in laurelane, `physics515/deepseek-ocr.rs`)
      to Burn on the same runner, so Candle can be dropped from consumers entirely.

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
