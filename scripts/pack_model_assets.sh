#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-/home/radxa/embeddinggemma_npu_seq128_bias_hidden_fp32}"
OUT_DIR="${2:-/home/radxa/vip9000-embeddinggemma/dist}"
VERSION="${VERSION:-v0.1.0}"
ZSTD_LEVEL="${ZSTD_LEVEL:-3}"
ASSET="vip9000-embeddinggemma-a7s-seq128-fp32-${VERSION}.tar.zst"

required=(
  network_binary.nb
  nbg_meta.json
  token_embedding_fp16.dat
  dense_2_weight_f32.npy
  dense_3_weight_f32.npy
)

mkdir -p "$OUT_DIR"

for file in "${required[@]}"; do
  if [[ ! -f "$MODEL_DIR/$file" ]]; then
    echo "missing required file: $MODEL_DIR/$file" >&2
    exit 1
  fi
done

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/runtime"

for file in "${required[@]}"; do
  cp "$MODEL_DIR/$file" "$tmp/runtime/$file"
done

cat > "$tmp/runtime/MANIFEST.json" <<EOF
{
  "name": "vip9000-embeddinggemma-a7s-seq128-fp32",
  "version": "$VERSION",
  "target": "Radxa Cubie A7S / Allwinner A733 / Vivante VIP9000",
  "inputs": {
    "inputs_embeds": [1, 128, 768],
    "attention_bias": [1, 1, 1, 128]
  },
  "output": [1, 128, 768],
  "dtype": "float32"
}
EOF

(cd "$tmp/runtime" && sha256sum "${required[@]}" MANIFEST.json > SHA256SUMS)

tar -C "$tmp" -I "zstd -${ZSTD_LEVEL} -T0" -cf "$OUT_DIR/$ASSET" runtime
(cd "$OUT_DIR" && sha256sum "$ASSET" > "$ASSET.sha256")

echo "$OUT_DIR/$ASSET"
echo "$OUT_DIR/$ASSET.sha256"
