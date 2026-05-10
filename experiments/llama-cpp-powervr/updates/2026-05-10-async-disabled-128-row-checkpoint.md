# 2026-05-10: PowerVR async-disabled 128-row checkpoint

## Summary

The previous 256-row checkpoint was too optimistic. Follow-up testing showed
that default ggml Vulkan async execution can produce `inf` even for small F32
matvecs on PowerVR B-Series, and that 256 output rows is not reliable even when
async is disabled.

The current conservative state is:

- PowerVR B-Series disables ggml Vulkan async internally.
- `m=16` and `m=128` Qwen-shaped F32 matvec op tests pass without setting
  `GGML_VK_DISABLE_ASYNC`.
- `m=256` and larger are guarded back to CPU.
- The standalone Vulkan repro still passes chunked `m=1024`, so wide-row
  corruption remains specific to the ggml integration path.

## llama.cpp Patch

Patch:

- `patches/0005-vulkan-disable-async-on-powervr-bxm.patch`

Local llama.cpp commit:

- `5a663705e vulkan: disable async on PowerVR BXM`

## Validation

Passing boundary:

```bash
cd /home/radxa/llama.cpp
timeout 60s env GGML_VK_VISIBLE_DEVICES=0 \
  build-vulkan/bin/test-backend-ops test -b Vulkan0 \
  --test-file /tmp/vcur_m128.txt
```

Observed result:

```text
ggml_vulkan: WARNING: Async execution disabled on this Vulkan device.
MUL_MAT(name=Vcur_m128,type=f32,ne=[128,1,1,1],sources=f32[1024,128,1,1],f32[1024,1,1,1]): OK
```

Guarded boundary:

```bash
timeout 60s env GGML_VK_VISIBLE_DEVICES=0 \
  build-vulkan/bin/test-backend-ops test -b Vulkan0 \
  --test-file /tmp/vcur_m256.txt
```

Observed result:

```text
ggml_vulkan: WARNING: Async execution disabled on this Vulkan device.
MUL_MAT(name=Vcur_m256,type=f32,ne=[256,1,1,1],sources=f32[1024,256,1,1],f32[1024,1,1,1]): not supported [Vulkan0]
```

## Repro Narrowing

The standalone repro now matches more of the ggml shape:

- five storage-buffer bindings,
- 13-word push constant block,
- specialization constants,
- chunked row dispatch,
- one descriptor set per chunk,
- the same `batch_a_index()` helper shape.
- nonzero descriptor offsets into one arena buffer,
- device-local arena execution with staging upload and readback copies.

It still passes `512x1024` and `1024x1024`, so the next likely suspects are
ggml graph/backend scheduling, tensor initialization, or queue submission
sequencing outside the simple repro.
