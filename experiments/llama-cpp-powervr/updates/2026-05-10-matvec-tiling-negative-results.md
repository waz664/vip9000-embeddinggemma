# PowerVR Vulkan matvec tiling negative results

Date: 2026-05-10 UTC

This note records the next iteration after the guarded baseline. These changes
were tested locally in llama.cpp and intentionally not kept as code because they
regressed correctness.

## What was tested

- Re-enabled F32 transformer projection `MUL_MAT` shapes up to 4096 output rows.
- Tried single-dispatch execution with X workgroups capped below 256 and Z used
  for larger row counts.
- Tried tiled F32 matvec dispatches with row chunks of 240, 128, 16, and 1.
- Tried shifted source/destination descriptors, broad descriptor ranges, narrowed
  descriptor ranges, and an explicit destination row-base expression in the
  scalar shader.
- Tried explicit `ggml_vk_sync_buffers()` before and after PowerVR scalar
  dispatches.
- Tried copying row slices into `prealloc_x` scratch before dispatch.

## Observed behavior

- Original scalar F32 shader is correct for small single-dispatch shapes:
  - `f32[1024,16] x f32[1024,1]`: OK
  - `f32[1024,128] x f32[1024,1]`: sometimes OK, sometimes fails without extra
    guardrails, suggesting a visibility or driver issue.
  - `f32[1024,255] x f32[1024,1]`: observed OK in single-dispatch probing.
  - `f32[1024,256] x f32[1024,1]`: fails.
- Multi-dispatch tiling into the same destination tensor remains corrupt even
  when descriptor offsets are aligned and explicit syncs are inserted.
- Narrowing descriptor ranges can make previously passing small cases fail.
- Adding a shader destination row-base expression also regressed previously
  passing cases, so it is not a safe workaround on this driver.
- Scratch-copying source rows into a small buffer changed the error mode but did
  not produce correct large projection results.

## Current conclusion

The next path should avoid multi-dispatch writes into a shifted destination and
avoid modifying the scalar shader's destination index expression. The better
direction is likely a new PowerVR-specific shader that maps output rows within a
single dispatch without exceeding the driver's failing workgroup pattern, or a
separate standalone Vulkan reproducer to isolate whether the issue is ggml
descriptor management or the PowerVR compiler/driver.

The llama.cpp tree was restored to the guarded safe baseline after these tests.
