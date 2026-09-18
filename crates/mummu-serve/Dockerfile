# mummu-serve — minimal container for running mummu models with GPU support.
#
# Build from the REPO ROOT (the workspace manifest is the build context):
#   docker build -f crates/mummu-serve/Dockerfile -t mummu-serve:local .
#
# GPU strategy (settled empirically 2026-08-21):
#   - The binary carries BOTH GPU backends. wgpu/Vulkan is the default
#     (`MUMMU_BACKEND` unset = auto).
#     CORRECTION 2026-09-14: this used to claim that on native Linux the
#     nvidia container runtime injects the real NVIDIA Vulkan ICD and the
#     parity-validated wgpu stack "just works". Measured on the DeepStack
#     Linux host, it does NOT come up in-container. Two faults: the ICD
#     needs Xlib (fixed below, libx11-6/libxext6), and past that the loader
#     still cannot resolve vkCreateInstance via vk_icdGetInstanceProcAddr
#     despite every library and device node being present and identical to
#     the host's. Until that is solved, a container wanting the GPU must
#     set MUMMU_BACKEND=cuda even on native Linux.
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
# Which commit is this? The image cannot work it out for itself: `.dockerignore`
# excludes `.git/` (deliberately — the repository is history the image has no
# use for), so the tree `COPY . .` lands has no repository in it, `build.rs`
# finds no git to ask, and the binary stamps itself `unknown`. That is what the
# released v0.2.0 image answers on /api/health today: the one answer that
# identifies nothing, in the field whose entire job is to confirm a deploy.
#
# So the sha is passed IN. Build with:
#   docker build --build-arg MUMMU_BUILD_SHA=$(git rev-parse --short=7 HEAD) \
#                -f crates/mummu-serve/Dockerfile -t mummu-serve:local .
# The compose service passes it for you (`build.args` in ai.yaml), so a deploy
# through compose cannot forget. Left unset it degrades to exactly the old
# behaviour — a build that says it does not know, never a build that fails.
ARG MUMMU_BUILD_SHA=unknown
ENV MUMMU_BUILD_SHA=${MUMMU_BUILD_SHA}
RUN cargo build --release -p mummu-serve --features cuda

# Runtime: CUDA base (sets up the NVIDIA apt repo + LD_LIBRARY_PATH) plus
# NVRTC for CubeCL's runtime kernel compilation. nvrtc-dev/cudart-dev carry
# the headers (cuda_fp16.h et al.) NVRTC includes while compiling kernels.
# libvulkan1 stays for the native-Linux wgpu path.
FROM nvidia/cuda:12.6.3-base-ubuntu24.04
# libx11-6 + libxext6 are load-bearing for the wgpu path, not decoration:
# NVIDIA's Vulkan ICD is libGLX_nvidia.so.0, which links against Xlib. Without
# them the loader logs "libX11.so.6: cannot open shared object file", DISCARDS
# the only driver ("vkCreateInstance: Found no drivers!"), and wgpu silently
# falls back to CPU — /api/health reports `"device":"CPU (flex)","gpus":[]` on
# a host whose GPU is fully injected and whose nvidia-smi works in-container.
# Diagnosed 2026-09-14 deploying to the DeepStack Linux host: the GPU-strategy
# note above claiming the wgpu stack "just works" natively was wrong by exactly
# these two packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libvulkan1 \
        libx11-6 \
        libxext6 \
        cuda-nvrtc-12-6 \
        cuda-nvrtc-dev-12-6 \
        cuda-cudart-dev-12-6 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/mummu-serve /usr/local/bin/mummu-serve
# CubeCL discovers its config by walking UP from the process CWD, and this
# image runs with CWD=/ — so the file has to live at the root to be found.
# Without it the autotune cache has nowhere to persist (the default root is
# the Cargo target tree, absent from a runtime image) and every container
# start re-tunes from scratch.
COPY crates/mummu-serve/cubecl.toml /cubecl.toml

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
