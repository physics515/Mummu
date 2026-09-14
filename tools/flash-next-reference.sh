#!/usr/bin/env bash
# The qwen4exp parity reference.
#
# Qwen3.8-Flash-Next is `general.architecture = qwen4exp`, which mainline
# llama.cpp releases did not carry when this was written. The ONLY
# qwen4exp-capable llama.cpp on this host is the one bundled inside the
# `ollama/ollama:latest` image (0.34.0, built 2026-09-09): verified by
# `grep -c qwen4exp /usr/lib/ollama/libllama.so` -> 3 hits. It lives in the
# image rather than on the host, so the reference RUNS FROM THE IMAGE instead
# of being extracted — llama-server needs its bundled libggml/libllama
# siblings, and copying the set out only to recreate its library path is a
# worse failure mode than mounting the model in.
#
# This deliberately does NOT start ollama itself: the compose service is
# retired, and a second daemon on the model store is a corruption risk (see
# the ollama block in compose-linux/ai.yaml). Only llama-server runs, via
# --entrypoint.
#
# Usage:
#   tools/flash-next-reference.sh [port]
#
# Env:
#   MUMMU_QWEN4EXP_DIR  directory holding the four shards
#                       (default: /mnt/deepmem/AI Models/qwen3.8-flash-next)
#
# Point it at shard 1 only: llama.cpp follows the `-of-` naming to the rest,
# the same convention mummu's GgufFile::open_sharded implements.
set -euo pipefail

PORT="${1:-8099}"
DIR="${MUMMU_QWEN4EXP_DIR:-/mnt/deepmem/AI Models/qwen3.8-flash-next}"
SHARD="Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
IMAGE="ollama/ollama:latest"

if [ ! -f "$DIR/$SHARD" ]; then
  echo "reference: no model at $DIR/$SHARD" >&2
  echo "fetch the 4-shard set first (111 GB)" >&2
  exit 1
fi

# Every shard must be present: llama.cpp opens the set, not the one file, and
# a missing shard surfaces as a confusing mid-load failure otherwise.
for i in 1 2 3 4; do
  f=$(printf "Qwen3.8-Flash-Next-UD-Q4_K_XL-%05d-of-00004.gguf" "$i")
  [ -f "$DIR/$f" ] || { echo "reference: missing shard $f" >&2; exit 1; }
done

echo "reference: llama-server (qwen4exp) on :$PORT from $IMAGE"
exec docker run --rm --gpus all \
  -p "$PORT:$PORT" \
  -v "$DIR:/models:ro" \
  --entrypoint /usr/lib/ollama/llama-server \
  "$IMAGE" \
  --model "/models/$SHARD" \
  --host 0.0.0.0 --port "$PORT" \
  --ctx-size 4096 \
  --n-gpu-layers 0
