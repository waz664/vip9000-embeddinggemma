# 2026-05-10: Forced wide matvec diagnostic override

## Summary

Added a developer-only llama.cpp override:

```bash
GGML_VK_POWERVR_FORCE_WIDE_MATVEC=1
```

By default, PowerVR B-Series still guards F32 matvec above 128 output rows back
to CPU. The override bypasses that guard so failing wide rows can be tested
without rebuilding llama.cpp.

## Result

Forced wide rows are not reliable:

- Forced `m=256` can pass once, then fail on the next process.
- Forced `m=512` fails.
- Forced `m=1024` fails.

This confirms the default 128-row ceiling is the correct user-facing behavior
for now.

## Patch

- `patches/0006-vulkan-add-powervr-wide-matvec-test-override.patch`

Local llama.cpp commit:

- `d7ee6a85a vulkan: add PowerVR wide matvec test override`

## Commands

```bash
cd /home/radxa/llama.cpp
timeout 60s env GGML_VK_VISIBLE_DEVICES=0 \
  GGML_VK_POWERVR_FORCE_WIDE_MATVEC=1 \
  build-vulkan/bin/test-backend-ops test -b Vulkan0 \
  --test-file /tmp/vcur_m256.txt
```

The env var is for debugging only. Do not use it for normal inference or
benchmarks.
