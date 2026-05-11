#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
AI_SDK_DIR="${AI_SDK_DIR:-$HOME/ai-sdk}"
VIPLITE_DIR="${VIPLITE_DIR:-$AI_SDK_DIR/viplite-tina/lib/aarch64-none-linux-gnu/v2.0}"
OUT="${OUT:-$REPO_DIR/tools/persistent_viplite_runner}"

"${CC:-gcc}" -O2 -Wall -Wextra \
  -I"$VIPLITE_DIR/inc" \
  "$REPO_DIR/tools/persistent_viplite_runner.c" \
  -L"$VIPLITE_DIR" -Wl,-rpath,"$VIPLITE_DIR" \
  -lNBGlinker -lVIPhal -lm \
  -o "$OUT"

echo "$OUT"
