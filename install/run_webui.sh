#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${INSTALL_DIR:-$HOME/embeddinggemma_npu_seq128_bias_hidden_fp32}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PORT="${PORT:-8080}"

if [[ ! -f "$INSTALL_DIR/rag_demo/index/chunks.json" || ! -f "$INSTALL_DIR/rag_demo/index/embeddings.npy" ]]; then
  echo "RAG index not found. Building a small default index first..." >&2
  (cd "$INSTALL_DIR/rag_demo" && python3 ./build_index.py --max-chunks "${MAX_CHUNKS:-8}")
fi

export VIP9000_RAG_MODEL_DIR="$INSTALL_DIR"
export VIP9000_RAG_PORT="$PORT"
export VIP9000_RAG_LLM_PROVIDER="${VIP9000_RAG_LLM_PROVIDER:-ollama}"
if [[ -z "${EMBEDDINGGEMMA_VIP_RUNNER:-}" && -x "$REPO_DIR/tools/persistent_viplite_runner" ]]; then
  export EMBEDDINGGEMMA_VIP_RUNNER="$REPO_DIR/tools/persistent_viplite_runner"
fi
exec python3 "$REPO_DIR/webui/app.py"
