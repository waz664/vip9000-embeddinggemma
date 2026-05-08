#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-waz664/vip9000-embeddinggemma}"
VERSION="${VERSION:-v0.1.0}"
ASSET="${ASSET:-vip9000-embeddinggemma-a7s-seq128-fp32-${VERSION}.tar.zst}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/embeddinggemma_npu_seq128_bias_hidden_fp32}"
WORK_DIR="${WORK_DIR:-/tmp/vip9000-embeddinggemma-install}"
BASE_URL="${BASE_URL:-https://github.com/${REPO}/releases/download/${VERSION}}"
SKIP_DEPS="${SKIP_DEPS:-0}"

mkdir -p "$WORK_DIR" "$INSTALL_DIR"

if [[ "$SKIP_DEPS" != "1" ]]; then
  python3 -m pip install --user -r "requirements.txt"
fi

download() {
  local url="$1"
  local out="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 -o "$out" "$url"
  else
    wget -O "$out" "$url"
  fi
}

if [[ -n "${LOCAL_ASSET:-}" ]]; then
  cp "$LOCAL_ASSET" "$WORK_DIR/$ASSET"
  if [[ -f "$LOCAL_ASSET.sha256" ]]; then
    cp "$LOCAL_ASSET.sha256" "$WORK_DIR/$ASSET.sha256"
  fi
else
  download "$BASE_URL/$ASSET" "$WORK_DIR/$ASSET"
  download "$BASE_URL/$ASSET.sha256" "$WORK_DIR/$ASSET.sha256"
fi

if [[ -f "$WORK_DIR/$ASSET.sha256" ]]; then
  (cd "$WORK_DIR" && sha256sum -c "$ASSET.sha256")
fi

tar -I zstd -xf "$WORK_DIR/$ASSET" -C "$WORK_DIR"
cp "$WORK_DIR/runtime/"* "$INSTALL_DIR/"

cp embed_text_bias_hidden_npu.py benchmark_bias_vs_cpu.py "$INSTALL_DIR/"
mkdir -p "$INSTALL_DIR/rag_demo"
cp rag_demo/*.py "$INSTALL_DIR/rag_demo/"
cp -r webui "$INSTALL_DIR/"
chmod +x "$INSTALL_DIR"/embed_text_bias_hidden_npu.py "$INSTALL_DIR"/benchmark_bias_vs_cpu.py "$INSTALL_DIR"/rag_demo/*.py "$INSTALL_DIR"/webui/app.py

echo "installed runtime to $INSTALL_DIR"
echo
echo "Smoke test:"
echo "  cd $INSTALL_DIR"
echo "  ./embed_text_bias_hidden_npu.py \"does Cubie A7S support NVMe storage?\""
