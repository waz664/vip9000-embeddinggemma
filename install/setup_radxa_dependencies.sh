#!/usr/bin/env bash
set -euo pipefail

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required" >&2
  exit 1
fi

python3 -m pip install --user -r requirements.txt

if [[ ! -x /home/radxa/ai-sdk/examples/vpm_run/vpm_run ]]; then
  cat >&2 <<'EOF'
WARNING: /home/radxa/ai-sdk/examples/vpm_run/vpm_run was not found.

Install the Radxa/Allwinner VIPLite AI SDK for this board, then verify:

  /home/radxa/ai-sdk/examples/vpm_run/vpm_run
  /home/radxa/ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v2.0

EOF
fi

if [[ ! -e /dev/vipcore ]]; then
  echo "WARNING: /dev/vipcore was not found. The NPU driver may not be loaded." >&2
fi

echo "dependency setup complete"
