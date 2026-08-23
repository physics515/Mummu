# mummu-serve — minimal container for running mummu models with GPU support.
#
# Build from the REPO ROOT (the workspace manifest is the build context):
#   docker build -f crates/mummu-serve/Dockerfile -t mummu-serve:local .
#
# GPU strategy (settled empirically 2026-08-21):
#   - The binary carries BOTH GPU backends. wgpu/Vulkan stays the default
#     (`MUMMU_BACKEND` unset = auto): on native Linux the nvidia container
#     runtime injects the real NVIDIA Vulkan ICD and the parity-validated
#     wgpu stack just works.
#   - Docker Desktop / WSL2 exposes NO NVIDIA Vulkan ICD to containers. The
#     mesa dzn (Vulkan-on-D3D12) route was built and tried: with a patched
#     conformance version, an x11-WSI build for VK_KHR_swapchain, and a
#     raised maxStorageBufferRange, kernels EXECUTE but compute garbage
#     (NaN logit collapse on both the SPIR-V and WGSL compilers) — LLM
#     weight tensors are ~1 GiB single storage buffers, far past what dzn
#     handles today. Dead end for correctness; deliberately NOT shipped.
#   - What WSL2 containers DO get is libcuda (the same injection ollama
#     relies on), so the image builds the `cuda` feature and the compose
#     service sets MUMMU_BACKEND=cuda: burn's CUDA backend, kernels
#     NVRTC-compiled at runtime. Requires `--gpus all` (or the compose
#     nvidia device reservation).
#   - With no adapter and no MUMMU_BACKEND, the server falls back to the
#     CPU backend automatically — same binary; GET /api/health reports
#     which backend is live.

FROM rust:1-bookworm AS build
WORKDIR /src
COPY . .
RUN cargo build --release -p mummu-serve --features cuda

# Runtime: CUDA base (sets up the NVIDIA apt repo + LD_LIBRARY_PATH) plus
# NVRTC for CubeCL's runtime kernel compilation. nvrtc-dev/cudart-dev carry
# the headers (cuda_fp16.h et al.) NVRTC includes while compiling kernels.
# libvulkan1 stays for the native-Linux wgpu path.
FROM nvidia/cuda:12.6.3-base-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libvulkan1 \
        cuda-nvrtc-12-6 \
        cuda-nvrtc-dev-12-6 \
        cuda-cudart-dev-12-6 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/mummu-serve /usr/local/bin/mummu-serve

ENV MUMMU_MODELS_DIR=/models \
    MUMMU_ADDR=0.0.0.0:8095 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all
VOLUME /models
# 8095 = native API + web UI; 11435 = ollama-compatible shim (MUMMU_OLLAMA_ADDR).
EXPOSE 8095 11435
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=5 \
    CMD curl -sf -o /dev/null http://127.0.0.1:8095/api/health || exit 1
ENTRYPOINT ["mummu-serve"]
