#!/usr/bin/env bash
set -euo pipefail

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required" >&2
  exit 1
fi

python3 -m pip install --user -r requirements.txt

AI_SDK_DIR="${AI_SDK_DIR:-$HOME/ai-sdk}"
VPM_RUN="${VIP9000_VPM_RUN:-$AI_SDK_DIR/examples/vpm_run/vpm_run}"
VIPLITE_LIB="${VIP9000_VIPLIB:-$AI_SDK_DIR/viplite-tina/lib/aarch64-none-linux-gnu/v2.0}"

if [[ ! -x "$VPM_RUN" ]]; then
  cat >&2 <<'EOF'
WARNING: VIPLite vpm_run was not found.

Install the Radxa/Allwinner VIPLite AI SDK for this board, then verify:

  $AI_SDK_DIR/examples/vpm_run/vpm_run
  $AI_SDK_DIR/viplite-tina/lib/aarch64-none-linux-gnu/v2.0

Override paths with:

  AI_SDK_DIR=/path/to/ai-sdk
  VIP9000_VPM_RUN=/path/to/vpm_run
  VIP9000_VIPLIB=/path/to/viplite/lib

EOF
fi

if [[ ! -d "$VIPLITE_LIB" ]]; then
  echo "WARNING: VIPLite library directory was not found: $VIPLITE_LIB" >&2
fi

if [[ ! -e /dev/vipcore ]]; then
  echo "WARNING: /dev/vipcore was not found. The NPU driver may not be loaded." >&2
fi

echo "dependency setup complete"
