#!/usr/bin/env bash
set -euo pipefail

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
BUILD_DIR="${BUILD_DIR:-$LLAMA_CPP_DIR/build-vulkan}"
STUB_SRC="${STUB_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/missing_vulkan_shader_stubs.cpp}"
VK_BUILD_DIR="$BUILD_DIR/ggml/src/ggml-vulkan"

if [[ ! -f "$VK_BUILD_DIR/CMakeFiles/ggml-vulkan.dir/link.txt" ]]; then
  echo "missing Vulkan link.txt under $VK_BUILD_DIR; build ggml-vulkan once first" >&2
  exit 1
fi

cp "$STUB_SRC" "$VK_BUILD_DIR/missing_vulkan_shader_stubs.cpp"
cd "$VK_BUILD_DIR"
c++ -fPIC -O2 -std=gnu++17 -c missing_vulkan_shader_stubs.cpp -o missing_vulkan_shader_stubs.cpp.o
cmd=$(sed 's# -Wl,-rpath# missing_vulkan_shader_stubs.cpp.o -Wl,-rpath#' CMakeFiles/ggml-vulkan.dir/link.txt)
bash -c "$cmd"

cd "$BUILD_DIR/bin"
ln -sf libggml-vulkan.so.0.11.0 libggml-vulkan.so.0
ln -sf libggml-vulkan.so.0 libggml-vulkan.so

echo "relinked $BUILD_DIR/bin/libggml-vulkan.so with shader stubs"
