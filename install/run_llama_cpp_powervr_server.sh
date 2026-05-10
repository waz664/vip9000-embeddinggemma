#!/usr/bin/env bash
set -euo pipefail

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
MODEL="${MODEL:-$LLAMA_CPP_DIR/Qwen3-0.6B-F16-from-Q8.gguf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8081}"

cd "$LLAMA_CPP_DIR"

exec env GGML_VK_VISIBLE_DEVICES="${GGML_VK_VISIBLE_DEVICES:-0}" \
  "$LLAMA_CPP_DIR/build-vulkan/bin/llama-server" \
  -m "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  -c "${CTX_SIZE:-1024}" \
  -b "${BATCH_SIZE:-8}" \
  -ub "${UBATCH_SIZE:-8}" \
  -ngl "${GPU_LAYERS:-2}" \
  --no-kv-offload \
  -fa off \
  --reasoning off \
  --alias "${MODEL_ALIAS:-qwen3-0.6b-powervr}"
