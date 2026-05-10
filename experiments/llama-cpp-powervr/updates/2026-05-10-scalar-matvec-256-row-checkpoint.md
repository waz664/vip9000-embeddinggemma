# 2026-05-10: PowerVR scalar F32 matvec 256-row checkpoint

## Summary

The llama.cpp Vulkan branch now has a conservative PowerVR BXM checkpoint for
F32 matvec:

- `m=16`, `m=128`, and `m=256` Qwen-shaped F32 matvec op tests pass on
  `Vulkan0`.
- `m=512` and larger are guarded back to CPU instead of returning corrupt
  values.
- A standalone Vulkan repro in `repro/` still passes chunked 1024-row matvecs,
  so the remaining wide-row failure appears to be in the ggml/llama.cpp Vulkan
  integration path rather than a fundamental hardware limit.

## llama.cpp Patch

Patch:

- `patches/0004-vulkan-extend-powervr-scalar-matvec-coverage.patch`

Local llama.cpp commit:

- `b8dee287b vulkan: extend PowerVR scalar matvec coverage`

## Validation

Passing boundary:

```bash
cd /home/radxa/llama.cpp
timeout 60s env GGML_VK_VISIBLE_DEVICES=0 \
  build-vulkan/bin/test-backend-ops test -b Vulkan0 \
  --test-file /tmp/vcur_m256.txt
```

Observed result:

```text
MUL_MAT(name=Vcur_m256,type=f32,ne=[256,1,1,1],sources=f32[1024,256,1,1],f32[1024,1,1,1]): OK
```

Guarded boundary:

```bash
timeout 60s env GGML_VK_VISIBLE_DEVICES=0 \
  build-vulkan/bin/test-backend-ops test -b Vulkan0 \
  --test-file /tmp/vcur_m512.txt
```

Observed result:

```text
MUL_MAT(name=Vcur_m512,type=f32,ne=[512,1,1,1],sources=f32[1024,512,1,1],f32[1024,1,1,1]): not supported [Vulkan0]
```

Standalone repro:

```bash
cd /home/radxa/vip9000-embeddinggemma
experiments/llama-cpp-powervr/repro/run_matvec_repro.sh 1024 1024
```

Observed result:

```text
rows=1024 cols=1024 max_err=2.49071e-06 max_row=53
```

## Notes

The shader now emits one output row per workgroup and uses the previously unused
`stride_a` push-constant field as a base row for chunked dispatch. This keeps the
push constant layout size unchanged while avoiding descriptor rebasing, which was
one of the earlier failure modes.

Wide ggml dispatches still fail if forced past the 256-row guard. Next useful
work is to isolate the difference between the passing standalone repro and the
failing ggml dispatch path: descriptor set allocation/reuse, tensor buffer
suballocation, or command-buffer sequencing.
