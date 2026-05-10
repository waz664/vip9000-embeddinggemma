#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_ROOT="${LLAMA_ROOT:-/home/radxa/llama.cpp}"
VK_INC="${VK_INC:-$LLAMA_ROOT/build-vulkan-deps/khronos-include}"
VK_LIB="${VK_LIB:-$LLAMA_ROOT/build-vulkan-deps/root/usr/lib/aarch64-linux-gnu/libvulkan.so}"
GLSLC="${GLSLC:-$LLAMA_ROOT/build-vulkan-deps/bin/glslc}"

MODE="${3:-f32}"
if [[ "$MODE" == "f16a" ]]; then
  SHADER="$ROOT/matvec_f16a_scalar.comp"
  SPV="$ROOT/matvec_f16a_scalar.spv"
else
  SHADER="$ROOT/matvec_scalar.comp"
  SPV="$ROOT/matvec_scalar.spv"
fi

"$GLSLC" "$SHADER" -o "$SPV"
g++ -O2 -std=c++17 "$ROOT/vulkan_matvec_repro.cpp" -I"$VK_INC" "$VK_LIB" -ldl -lpthread -o "$ROOT/vulkan_matvec_repro"

"$ROOT/vulkan_matvec_repro" "${1:-1024}" "${2:-1024}" "$SPV" "$MODE"
