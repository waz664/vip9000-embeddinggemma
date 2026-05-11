#!/usr/bin/env bash
set -euo pipefail

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
MODEL="${MODEL:-$LLAMA_CPP_DIR/Qwen3-0.6B-F16-from-Q8.gguf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8081}"
CPUSET="${CPUSET:-}"
THREADS="${THREADS:-}"
THREADS_BATCH="${THREADS_BATCH:-}"

cd "$LLAMA_CPP_DIR"

cmd=(
  "$LLAMA_CPP_DIR/build-vulkan/bin/llama-server"
  -m "$MODEL"
  --host "$HOST"
  --port "$PORT"
  -c "${CTX_SIZE:-2048}"
  -b "${BATCH_SIZE:-8}"
  -ub "${UBATCH_SIZE:-8}"
  -ngl "${GPU_LAYERS:-2}"
  --no-kv-offload
  -fa off
  --reasoning off
  -np "${PARALLEL_SLOTS:-1}"
  --no-cache-idle-slots
  --no-warmup
  --alias "${MODEL_ALIAS:-qwen3-0.6b-powervr}"
)

if [[ -n "$THREADS" ]]; then
  cmd+=(--threads "$THREADS")
fi
if [[ -n "$THREADS_BATCH" ]]; then
  cmd+=(--threads-batch "$THREADS_BATCH")
fi
if [[ -n "$CPUSET" ]]; then
  cmd=(taskset -c "$CPUSET" "${cmd[@]}")
fi

env_vars=(
  "GGML_VK_VISIBLE_DEVICES=${GGML_VK_VISIBLE_DEVICES:-0}"
  "LLAMA_VK_NO_OUTPUT_OFFLOAD=${LLAMA_VK_NO_OUTPUT_OFFLOAD:-1}"
)
for name in \
  GGML_VK_POWERVR_ALLOW_RMS_NORM \
  GGML_VK_POWERVR_ALLOW_SWIGLU \
  GGML_VK_POWERVR_ALLOW_ROPE \
  GGML_VK_POWERVR_ALLOW_ELEMENTWISE \
  GGML_VK_POWERVR_FULL_OPS; do
  if [[ -n "${!name:-}" ]]; then
    env_vars+=("$name=${!name}")
  fi
done

exec env \
  "${env_vars[@]}" \
  "${cmd[@]}"
