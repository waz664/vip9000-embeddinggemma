#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${INSTALL_DIR:-$HOME/embeddinggemma_npu_seq128_bias_hidden_fp32}"
QUERY="${1:-does Cubie A7S support NVMe storage?}"

required=(
  network_binary.nb
  nbg_meta.json
  token_embedding_fp16.dat
  dense_2_weight_f32.npy
  dense_3_weight_f32.npy
  tokenizer.model
  embed_text_bias_hidden_npu.py
)

for file in "${required[@]}"; do
  if [[ ! -f "$INSTALL_DIR/$file" ]]; then
    echo "missing required file: $INSTALL_DIR/$file" >&2
    exit 1
  fi
done

cd "$INSTALL_DIR"
python3 ./embed_text_bias_hidden_npu.py "$QUERY"
