# llama.cpp PowerVR Vulkan Experiment

This directory tracks the experimental llama.cpp work for the Radxa Cubie A7S
PowerVR BXM-4-64 GPU. It is separate from the VIP9000 embedding runtime.

## Current Status

Date: 2026-05-10

Branch in local llama.cpp checkout:

```bash
cd /home/radxa/llama.cpp
git switch radxa-powervr-vulkan
```

Patches exported here:

```text
experiments/llama-cpp-powervr/patches/0001-power-vr-vulkan-llama-cpp-experimental.patch
experiments/llama-cpp-powervr/patches/0003-power-vr-vulkan-guard-unstable-matvec.patch
experiments/llama-cpp-powervr/patches/0004-vulkan-extend-powervr-scalar-matvec-coverage.patch
```

The Vulkan backend builds and detects the GPU:

```text
Vulkan0: PowerVR B-Series BXM-4-64 MC1
```

Baseline llama.cpp Vulkan behavior:

- `-ngl 1` can initialize but only offloads the output layer.
- `-ngl 2` and higher fail on upstream llama.cpp matvec pipelines:
  `Compute pipeline creation failed for mul_mat_vec_q4_0_f32_f32`,
  `mul_mat_vec_q8_0_f32_f32`, or `mul_mat_vec_f16_f32_f32`.
- The device reports subgroup size 1 and only 16 KiB compute shared memory,
  which is outside the assumptions of the current llama.cpp Vulkan matvec
  shaders.

Implemented so far:

- Allow llama.cpp Vulkan initialization to continue on devices with too little
  shared memory for the normal matmul shader family.
- Disable subgroup matvec selection when the device reports subgroup size 1.
- Add an experimental scalar `F32 x F32 -> F32` matvec shader that avoids
  subgroup reductions, shared-memory reductions, and 8/16-bit storage feature
  requirements.

Test results:

- Q4, Q8, and F16 model paths still fail with upstream matvec shader pipeline
  creation errors at `-ngl 2`.
- A converted F32 test model with the scalar shader successfully loads,
  initializes, warms up, and generates with `-ngl 2` without pipeline creation
  failure.
- The scalar F32 path now passes exported Qwen-shaped op tests through 256
  output rows. Wider rows are guarded back to CPU because llama.cpp/ggml
  dispatches still corrupt data at 512+ rows.
- The standalone Vulkan repro in `repro/` passes chunked 1024-row matvecs, so
  the remaining wide-row failure appears to be in the ggml integration path,
  not a fundamental PowerVR hardware limit.

## Rebuild Notes

This board currently builds with local Vulkan headers and tools staged inside
the llama.cpp build tree:

```bash
cd /home/radxa/llama.cpp
cmake -B build-vulkan -DGGML_VULKAN=ON \
  -DVulkan_INCLUDE_DIR=/home/radxa/llama.cpp/build-vulkan-deps/khronos-include \
  -DVulkan_LIBRARY=/home/radxa/llama.cpp/build-vulkan-deps/root/usr/lib/aarch64-linux-gnu/libvulkan.so \
  -DVulkan_GLSLC_EXECUTABLE=/home/radxa/llama.cpp/build-vulkan-deps/bin/glslc
cmake --build build-vulkan --target ggml-vulkan -- -j1
```

The local glslang toolchain cannot compile a few optional llama.cpp shader
variants. Until that is cleaned up, the build tree uses
`missing_vulkan_shader_stubs.cpp` to satisfy those unused symbols for this
experiment.

## Repro Commands

Boundary op tests:

```bash
cd /home/radxa/llama.cpp
timeout 60s env GGML_VK_VISIBLE_DEVICES=0 \
  build-vulkan/bin/test-backend-ops test -b Vulkan0 \
  --test-file /tmp/vcur_m256.txt

timeout 60s env GGML_VK_VISIBLE_DEVICES=0 \
  build-vulkan/bin/test-backend-ops test -b Vulkan0 \
  --test-file /tmp/vcur_m512.txt
```

Standalone driver repro:

```bash
cd /home/radxa/vip9000-embeddinggemma
experiments/llama-cpp-powervr/repro/run_matvec_repro.sh 1024 1024
```

Create test models from the local Q8 GGUF:

```bash
cd /home/radxa/llama.cpp
build-vulkan/bin/llama-quantize --allow-requantize \
  Qwen3-0.6B-Q8_0.gguf Qwen3-0.6B-F16-from-Q8.gguf F16 1
build-vulkan/bin/llama-quantize --allow-requantize \
  Qwen3-0.6B-Q8_0.gguf Qwen3-0.6B-F32-from-Q8.gguf F32 1
```

Known-good execution milestone:

```bash
timeout 180s env GGML_VK_VISIBLE_DEVICES=0 \
  build-vulkan/bin/llama-completion \
  -m /home/radxa/llama.cpp/Qwen3-0.6B-F32-from-Q8.gguf \
  -p 'Say OK.' -n 2 -c 64 -ngl 2 \
  --no-op-offload --no-kv-offload -fa off \
  --no-display-prompt --no-conversation --reasoning off \
  --temp 0.2 --top-p 0.9 --no-perf
```

Correctness control that currently fails for `-ngl 2`:

```bash
build-vulkan/bin/llama-completion \
  -m /home/radxa/llama.cpp/Qwen3-0.6B-F32-from-Q8.gguf \
  -p 'Say OK.' -n 1 -c 64 -ngl 0 --device none \
  --no-display-prompt --no-conversation --reasoning off \
  --temp 0 --seed 42 --no-perf

env GGML_VK_VISIBLE_DEVICES=0 build-vulkan/bin/llama-completion \
  -m /home/radxa/llama.cpp/Qwen3-0.6B-F32-from-Q8.gguf \
  -p 'Say OK.' -n 1 -c 64 -ngl 2 \
  --no-op-offload --no-kv-offload -fa off \
  --no-display-prompt --no-conversation --reasoning off \
  --temp 0 --seed 42 --no-perf
```

## Next Work

1. Isolate why the standalone Vulkan repro passes 1024-row chunked matvec while
   ggml's Vulkan dispatch corrupts 512+ row F32 matvec.
2. Fix wide F32 scalar correctness against CPU for a one-layer offload.
3. Once F32 is correct, port the same scalar strategy to Q8_0 or Q4_0 so the
   model size is practical.
4. After correctness, measure whether any scalar GPU path is faster or merely
   useful for CPU offload/background work.
