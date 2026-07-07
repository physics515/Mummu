# Mummu

**A from-scratch, single-binary local model runner in Rust — import any model, quantize it to fit, and run it on all the hardware you have: CPU, one GPU, or several. No Ollama, no cloud, no CUDA toolchain.**

Mummu imports models from HuggingFace (or disk), **(auto-)quantizes** them to fit, and runs them **natively in Rust on [Burn](https://burn.dev)** across every device you have. One binary — a runtime probe inventories your GPUs (Vulkan / DX12 / Metal via `wgpu`) and your CPU (`ndarray`) and places the model to use them **to the fullest**: several GPUs together, with CPU offload when VRAM is short, no feature-split builds and no CUDA toolchain. Models are **reimplemented from scratch**, generic over the Burn backend, and **parity-tested byte-for-byte** against a reference so the reimplementations can be trusted.

It exists because two local-first apps — **[laurelane](https://github.com/physics515/laurelane)** (a private budgeting cockpit) and **[Nanna](https://github.com/physics515/Nanna)** (an always-on local AI presence) — were building the *same* runner twice. Mummu is that runner, extracted and generalized: each app consumes it as a dependency and keeps only its own domain glue. Laurelane proved the blueprint (Qwen2.5 / LFM2.5 / all-MiniLM ported to Burn, byte-identical parity vs Candle, validated on an RTX 4070 Ti SUPER 16 GB); Mummu is where it lives, hardens, and grows.

## What it is

- **One binary, every device** — compile both `Wgpu` (Vulkan/DX12/Metal, no CUDA toolchain) and `NdArray` (CPU); a runtime probe enumerates **all** adapters + the CPU and places the model across them — a single GPU, **several GPUs together**, or GPU + CPU hybrid. No feature-split builds, no per-vendor path.
- **Models from scratch, generic over `B: Backend`** — a growing zoo (Qwen2/2.5, LFM2/2.5 hybrid conv+attention, all-MiniLM embedder) built on shared blocks (RmsNorm · GQA · RoPE · SwiGLU · tied lm-head · depthwise causal conv), with a clean trait to add more.
- **Trustworthy reimplementations** — every port must pass a **parity gate**: single-forward top-k logits *and* a short greedy sequence match a reference (Candle, or a local Ollama of the same model) exactly.
- **Fast** — per-layer KV cache (+ conv-state cache for hybrids), on-GPU argmax (sync only the winning index), sampling, **token streaming**, cooperative cancellation; kernel `fusion` + `autotune`; an opt-in **f16** path to roughly halve VRAM.
- **A full model-import suite** — pull a model from HuggingFace (by repo id) or from disk and load it: **safetensors**, **PyTorch** state dicts, and **GGUF** (llama.cpp, dequantized) weights; `config.json`-driven hyperparameters; tokenizer + chat-template import (HF `tokenizers` / SentencePiece / BPE); per-architecture weight-name remapping with a **checked load** (fail loudly on a key mismatch, never silently zero-init); resumable, shard-aware downloads into a per-user cache; and a declarative **model registry** so adding a model is a manifest entry, not new code.
- **Quantize to fit, fill the hardware** — a planner probes every GPU + the CPU (VRAM / RAM), then imports or **quantizes on the fly** (GGUF K-quants, GPTQ / AWQ, or Burn's own int8/int4) and chooses precision + **layer placement** so the *largest model that fits* runs and every device is used — sharded across GPUs, spilling cold layers to CPU when needed. Plus a model-management API (download progress, disk usage, remove) apps surface in their settings UI.
- **Local embeddings** — a from-scratch MiniLM-class sentence embedder (CPU) for fully-offline semantic search.

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
