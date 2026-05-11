#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/embeddinggemma_npu_seq128_bias_hidden_fp32}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
SKIP_RUNTIME="${SKIP_RUNTIME:-0}"
SKIP_LLAMA_BUILD="${SKIP_LLAMA_BUILD:-0}"
INSTALL_SERVICES="${INSTALL_SERVICES:-1}"
RUN_BENCHMARK="${RUN_BENCHMARK:-1}"

cd "$REPO_DIR"

echo "==> Installing Python/runtime dependencies"
bash install/setup_radxa_dependencies.sh

if [[ "$SKIP_RUNTIME" != "1" ]]; then
  echo "==> Installing prebuilt NPU runtime"
  bash install/install_runtime.sh
else
  echo "==> Skipping runtime install"
fi

echo "==> Running NPU smoke test"
bash install/smoke_test.sh

echo "==> Building demo vector index"
(cd "$INSTALL_DIR/rag_demo" && python3 ./build_index.py --max-chunks "${MAX_CHUNKS:-8}")

if [[ "$SKIP_LLAMA_BUILD" != "1" ]]; then
  echo "==> Building patched llama.cpp PowerVR backend"
  taskset -c "${BUILD_CPUS:-6,7}" bash install/build_llama_cpp_powervr.sh
else
  echo "==> Skipping llama.cpp build"
fi

if [[ "$INSTALL_SERVICES" == "1" ]]; then
  echo "==> Installing and starting services"
  bash install/install_systemd_services.sh
else
  echo "==> Skipping service install"
fi

if [[ "$RUN_BENCHMARK" == "1" ]]; then
  echo "==> Waiting for services"
  sleep 5
  python3 scripts/benchmark_webui.py --runs 2
fi

cat <<EOF

Full stack install complete.

WebUI:
  http://<radxa-ip>:8080

Useful commands:
  systemctl --user status vip9000-llama.service
  systemctl --user status vip9000-rag-webui.service
  journalctl --user -u vip9000-llama.service -f
  journalctl --user -u vip9000-rag-webui.service -f

EOF
