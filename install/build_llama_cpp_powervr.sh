#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
LLAMA_CPP_REPO="${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
BUILD_DIR="${BUILD_DIR:-$LLAMA_CPP_DIR/build-vulkan}"
JOBS="${JOBS:-1}"

if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
  git clone "$LLAMA_CPP_REPO" "$LLAMA_CPP_DIR"
fi

cd "$LLAMA_CPP_DIR"

for patch in "$REPO_ROOT"/patches/llama.cpp/*.patch; do
  if git apply --check "$patch" >/dev/null 2>&1; then
    git apply "$patch"
  else
    echo "skipping already-applied or incompatible patch: $patch" >&2
  fi
done

cmake_args=(
  -B "$BUILD_DIR"
  -DGGML_VULKAN=ON
)

if [[ -n "${Vulkan_INCLUDE_DIR:-}" ]]; then
  cmake_args+=("-DVulkan_INCLUDE_DIR=$Vulkan_INCLUDE_DIR")
fi
if [[ -n "${Vulkan_LIBRARY:-}" ]]; then
  cmake_args+=("-DVulkan_LIBRARY=$Vulkan_LIBRARY")
fi
if [[ -n "${Vulkan_GLSLC_EXECUTABLE:-}" ]]; then
  cmake_args+=("-DVulkan_GLSLC_EXECUTABLE=$Vulkan_GLSLC_EXECUTABLE")
fi

cmake "${cmake_args[@]}"
cmake --build "$BUILD_DIR" --target ggml-vulkan -- -j"$JOBS"

if ! cmake --build "$BUILD_DIR" --target llama-server llama-completion test-backend-ops -- -j"$JOBS"; then
  echo "build failed; trying PowerVR shader-stub relink workaround" >&2
  "$REPO_ROOT/install/fix_llama_cpp_vulkan_shader_stubs.sh"
  cmake --build "$BUILD_DIR" --target llama-server llama-completion test-backend-ops -- -j"$JOBS"
fi

echo "llama.cpp PowerVR build is ready under $BUILD_DIR"
