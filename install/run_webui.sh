#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${INSTALL_DIR:-$HOME/embeddinggemma_npu_seq128_bias_hidden_fp32}"
PORT="${PORT:-8080}"

if [[ ! -f "$INSTALL_DIR/rag_demo/index/chunks.json" || ! -f "$INSTALL_DIR/rag_demo/index/embeddings.npy" ]]; then
  echo "RAG index not found. Building a small default index first..." >&2
  (cd "$INSTALL_DIR/rag_demo" && ./build_index.py --max-chunks "${MAX_CHUNKS:-8}")
fi

export VIP9000_RAG_MODEL_DIR="$INSTALL_DIR"
export VIP9000_RAG_PORT="$PORT"
exec "$INSTALL_DIR/webui/app.py"
