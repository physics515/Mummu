# mummu-serve

A minimal HTTP server + single-page chat UI over [mummu](../mummu). Sync all
the way down (tiny_http worker threads, no async runtime), one resident model
per backend, streamed completions over SSE.

## Endpoints

| Method | Path          | What                                              |
| ------ | ------------- | ------------------------------------------------- |
| GET    | `/`           | the embedded chat UI                              |
| GET    | `/api/health` | device policy + GPU adapter inventory             |
| GET    | `/api/models` | the registry catalog with `installed` flags       |
| POST   | `/api/pull`   | `{"model": name}` — download, SSE progress frames |
| POST   | `/api/chat`   | messages + options — SSE `delta`/`done` frames    |
| POST   | `/api/unload` | drop the resident model (frees VRAM/RAM)          |

## Ollama-compatible shim

A second listener (default `0.0.0.0:11435`, env `MUMMU_OLLAMA_ADDR`, `off`
disables) speaks the Ollama HTTP protocol over the same engine, so Ollama
clients — open-webui, LangChain's Ollama integration, plain curl scripts —
can use mummu without knowing it isn't ollama. Both surfaces share the
resident model. Port 11435 (not 11434) so a real ollama on the same network
namespace can never collide.

Implemented: `GET /` ("Ollama is running"), `/api/version`, `/api/tags`
(installed catalog models), `/api/show`, `/api/ps`, `/api/chat` and
`/api/generate` (NDJSON streaming and non-stream, ollama option names incl.
`num_predict`), `/api/pull` (catalog names only), `DELETE /api/delete`.
Embeddings / create / copy / push answer with an explicit error. Model names
are mummu catalog names; a trailing `:latest` is accepted and stripped.

`POST /api/chat` body:

```json
{
  "model": "qwen3-0.6b",
  "messages": [{"role": "user", "content": "hi"}],
  "options": {"temperature": 0.7, "top_p": 0.9, "top_k": 64, "seed": 1, "max_tokens": 512}
}
```

Every option is optional; `temperature: 0` is exact greedy. Roles are
`system` / `user` / `assistant`; the family's byte-verified chat template
(ChatML for Qwen2/Qwen3/LFM2.5, Tulu for OLMoE) renders the prompt.

## Configuration

| Env                | Default        | What                                        |
| ------------------ | -------------- | ------------------------------------------- |
| `MUMMU_ADDR`       | `0.0.0.0:8095` | listen address                              |
| `MUMMU_MODELS_DIR` | `./models`     | model cache root (registry layout)          |
| `MUMMU_BACKEND`    | unset (auto)   | `cuda` \| `wgpu` \| `cpu`; auto = wgpu probe |
| `MUMMU_FORCE_CPU`  | unset          | `1` forces the CPU backend (wins over all)  |
| `MUMMU_QUANT`      | unset (off)    | starting quant policy `q8` \| `q4` for the fit planner (qwen35 + OLMoE GGUFs); the planner escalates from here to fit |
| `MUMMU_GPU_BUDGET_GB` | unset (15 GiB, or 7/8 of inventoried VRAM) | GPU memory budget the fit planner assumes |
| `MUMMU_PACK`       | unset (on)     | `off` skips the one-time `.mummu` pack import and serves GGUFs through the streaming re-quantizing loader |
| `MUMMU_PACK_PRECISIONS` | unset (`q4,q8,f16,f32`) | stored levels for the pack import |
| `MUMMU_TIERS`      | unset (on)     | MoE packs and partitioned dense packs: `off` keeps every expert / FFN cluster on the main backend at the planned level; `cpu` tiers onto the CPU only |
| `MUMMU_FFN_SKIP_TOLERANCE` | unset (exact) | dense packs: allow per-token FFN cluster skipping up to this measured max \|Δ log-prob\| (needs the pack's skip table from `pack-calibrate`) |
| `MUMMU_FFN_SKIP_TAU` | unset (exact) | dense packs: force a skip threshold directly (logged as unmeasured when the table lacks it) |

Auto policy is mummu's runtime probe: GPU (wgpu — Vulkan/DX12/Metal, no CUDA
toolchain) when a hardware adapter is present, else CPU (burn-flex). The
`cuda` backend exists only in `--features cuda` builds and is never picked
automatically.

**Fit planner.** For GGUF models of the quantizable families (qwen35, OLMoE)
every request is planned: resident bytes are estimated from the GGUF header
under each quant policy, and the first `(backend, policy)` that fits wins —
the preferred backend first (Off→Q8→Q4 from the `MUMMU_QUANT` start), then
the CPU with the same ladder; Q4-on-wgpu is never planned. That is how one
server keeps a 2B on CUDA at f32 and loads a 27B as `Cpu @ Q4`. Every fit
check is logged.

**`.mummu` pack (P9 stage 3).** The first load of a qwen35 / OLMoE GGUF
imports it **once** into `<model dir>/pack`: every tensor — and every MoE
expert separately — at every level (`q4.bin`, `q8.bin`, `f16.bin`,
`f32.bin`, block-32 scales, burn's canonical quantized layout), so later
loads are a copy at whatever level the planner picked, never a
re-quantization. Pre-import with `cargo run --release --example pack-import
-- <model.gguf> <model dir>/pack`. The import is atomic (temp dir + rename).

**Dense models tier too (P9 stage 3c).** A dense qwen35 pack gets its
FFNs **partitioned** into neuron clusters on import (or on first load of an
older pack) — exact, the neurons are only reordered in place, so the dense
loader is unchanged — and the same planner then places clusters: a local
slab on the main backend at the planned level, the rest on the other
devices at theirs (`tiers: N FFN clusters on cpu @ Q8 …`). Every cluster
always runs, so this is the same model. **Skipping** clusters per token
(gate-energy router, training-free) is a separate, opt-in, *measured*
trade: `pack-calibrate` writes a skip table (tau → max |Δ log-prob|,
argmax agreement, clusters kept) into the manifest, and the server picks a
tau only under `MUMMU_FFN_SKIP_TOLERANCE`.

**Tiered experts (P9 stage 3b).** For an MoE pack the experts are not all
loaded onto one backend at one level: `mummu::tier::plan_tiers` places each
expert on a device (CPU, and the wgpu/CUDA GPU when present) at the
precision that device runs best — f32 first on the GPU, int8/int4 on the
CPU — within each device's budget after the trunk, hottest experts first.
Every request's routing hits feed a smoothed hotness; after each request
the plan is recomputed and up to 16 experts are **hot-swapped** between
tiers on a background thread (the pack makes every level available
instantly). Placement and every re-tier are logged; `/api/ps` still reports
the model's main backend. Host memory is one pool: before the experts load,
the server checks `MemAvailable` against the worst-case CPU tier (every
expert at int8) plus slack and **evicts another model from the CPU slot**
if that is what it takes — a resident 27B plus a tiered OLMoE once tripped
the Docker VM's OOM killer. On a GPU shared with a desktop set
`MUMMU_GPU_BUDGET_GB` below the card (the DeepStack compose uses 11 on a
16 GB card); the tiers and the fit planner both honor it.

## Docker

Build from the **repo root** (the workspace is the context):

```
docker build -f crates/mummu-serve/Dockerfile -t mummu-serve:local .
docker run --gpus all -e MUMMU_BACKEND=cuda -p 8095:8095 -v mummu-models:/models mummu-serve:local
```

GPU-in-container facts (established empirically 2026-08-21, full trail in
the Dockerfile header):

- **Native Linux**: the nvidia container toolkit injects the real NVIDIA
  Vulkan ICD; auto mode uses the parity-validated wgpu stack. Leave
  `MUMMU_BACKEND` unset.
- **Docker Desktop / WSL2**: no NVIDIA Vulkan ICD reaches containers, and
  mesa's dzn (Vulkan-on-D3D12) computes garbage on ~GiB weight buffers —
  the only correct GPU path is `MUMMU_BACKEND=cuda` (libcuda is injected;
  kernels NVRTC-compile at runtime). Correct output verified byte-identical
  to the CPU backend under greedy decode; throughput is launch-latency-bound
  through GPU paravirtualization (≈ CPU parity at ≤1.5B f32), so small
  models may prefer `MUMMU_BACKEND=cpu`.
- With no adapter and no override the server falls back to CPU automatically
  — same binary; the startup log and `/api/health` report the live backend.
