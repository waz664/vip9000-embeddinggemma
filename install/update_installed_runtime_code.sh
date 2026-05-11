#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/embeddinggemma_npu_seq128_bias_hidden_fp32}"

if [[ ! -d "$INSTALL_DIR" ]]; then
  echo "install dir not found: $INSTALL_DIR" >&2
  exit 1
fi

cp "$REPO_DIR"/embed_text_bias_hidden_npu.py "$REPO_DIR"/benchmark_bias_vs_cpu.py "$INSTALL_DIR"/
mkdir -p "$INSTALL_DIR/rag_demo"
cp "$REPO_DIR"/rag_demo/*.py "$INSTALL_DIR/rag_demo"/
rm -rf "$INSTALL_DIR/webui"
cp -r "$REPO_DIR/webui" "$INSTALL_DIR/webui"

chmod +x \
  "$INSTALL_DIR"/embed_text_bias_hidden_npu.py \
  "$INSTALL_DIR"/benchmark_bias_vs_cpu.py \
  "$INSTALL_DIR"/rag_demo/*.py \
  "$INSTALL_DIR"/webui/app.py

echo "updated installed runtime code under $INSTALL_DIR"
