# Mummu

**A from-scratch, single-binary local model runner in Rust — small open models on one consumer GPU. No Ollama, no cloud, no CUDA toolchain.**

Mummu runs modern small language, embedding, and (later) vision models **natively in Rust on [Burn](https://burn.dev)** — one binary that transparently uses your GPU (Vulkan / DX12 / Metal via `wgpu`) when present and falls back to CPU (`ndarray`), chosen by a runtime device probe with no feature-split builds. Models are **reimplemented from scratch**, generic over the Burn backend, and **parity-tested byte-for-byte** against a reference so the reimplementations can be trusted.

It exists because two local-first apps — **[laurelane](https://github.com/physics515/laurelane)** (a private budgeting cockpit) and **[Nanna](https://github.com/physics515/Nanna)** (an always-on local AI presence) — were building the *same* runner twice. Mummu is that runner, extracted and generalized: each app consumes it as a dependency and keeps only its own domain glue. Laurelane proved the blueprint (Qwen2.5 / LFM2.5 / all-MiniLM ported to Burn, byte-identical parity vs Candle, validated on an RTX 4070 Ti SUPER 16 GB); Mummu is where it lives, hardens, and grows.

## What it is

- **One binary, dual backend, runtime probe** — compile both `Wgpu` (Vulkan/DX12/Metal, no CUDA toolchain) and `NdArray` (CPU); a cheap `wgpu` adapter probe picks the GPU if present, else CPU. No feature-split builds.
- **Models from scratch, generic over `B: Backend`** — a growing zoo (Qwen2/2.5, LFM2/2.5 hybrid conv+attention, all-MiniLM embedder) built on shared blocks (RmsNorm · GQA · RoPE · SwiGLU · tied lm-head · depthwise causal conv), with a clean trait to add more.
- **Trustworthy reimplementations** — every port must pass a **parity gate**: single-forward top-k logits *and* a short greedy sequence match a reference (Candle, or a local Ollama of the same model) exactly.
- **Fast on one GPU** — per-layer KV cache (+ conv-state cache for hybrids), on-GPU argmax (sync only the winning index), sampling, **token streaming**, cooperative cancellation; kernel `fusion` + `autotune`; an opt-in **f16** path to roughly halve VRAM.
- **Weights the modern way** — HF safetensors via `burn-store` (`SafetensorsStore` + `PyTorchToBurnAdapter` + a bf16→f32/f16 cast), checked load, streamed HF download into a per-user cache (resumable, shard-merging).
- **Sized to the device** — a size-tier picker (bigger model on GPU, smaller on CPU), VRAM budgeting that accounts for KV cache + display headroom, and a model-management API (download progress, disk usage, remove) that apps surface in their own settings UI.
- **Local embeddings** — a from-scratch MiniLM-class sentence embedder (CPU) for fully-offline semantic search.

## Design principles

- **Local-first, offline, private** — a single consumer GPU (or CPU) is the whole story; the cloud is never a dependency.
- **From scratch, in Rust, on Burn** — one backend-agnostic codebase (CPU / CUDA / Metal / Vulkan / WebGPU) instead of per-backend feature flags; first-class on-device **training / fine-tuning** later.
- **Parity or it didn't happen** — a reimplementation ships only when it is numerically byte-identical to a reference.
- **Performance is a gate** — a change lands only when the parity + perf budgets (TTFT, decode tok/s, VRAM ceiling) hold; README perf claims link a benchmark artifact.
- **README + ROADMAP are the only docs** — shipped capability is described here; everything planned or next lives as a `[ ]` in [ROADMAP.md](ROADMAP.md); git history + PRs are the record.

## Consumers

- **[Nanna](https://github.com/physics515/Nanna)** — the runner *is* the agent: local inference for the whole agent loop, plus local embeddings for its memory. Wires Mummu in as `Provider::Local`.
- **[laurelane](https://github.com/physics515/laurelane)** — on-device statement structuring + categorization; Mummu replaces its in-app Burn modules.

Reference GPU: **RTX 4070 Ti SUPER 16 GB**. The plan lives in **[ROADMAP.md](ROADMAP.md)**.
