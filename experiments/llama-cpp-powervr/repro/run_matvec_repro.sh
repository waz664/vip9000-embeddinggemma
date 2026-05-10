#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_ROOT="${LLAMA_ROOT:-/home/radxa/llama.cpp}"
VK_INC="${VK_INC:-$LLAMA_ROOT/build-vulkan-deps/khronos-include}"
VK_LIB="${VK_LIB:-$LLAMA_ROOT/build-vulkan-deps/root/usr/lib/aarch64-linux-gnu/libvulkan.so}"
GLSLC="${GLSLC:-$LLAMA_ROOT/build-vulkan-deps/bin/glslc}"

"$GLSLC" "$ROOT/matvec_scalar.comp" -o "$ROOT/matvec_scalar.spv"
g++ -O2 -std=c++17 "$ROOT/vulkan_matvec_repro.cpp" -I"$VK_INC" "$VK_LIB" -ldl -lpthread -o "$ROOT/vulkan_matvec_repro"

"$ROOT/vulkan_matvec_repro" "${1:-1024}" "${2:-1024}" "$ROOT/matvec_scalar.spv"
