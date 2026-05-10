# PowerVR scalar matvec wide-local-size negative result

Date: 2026-05-10 UTC

After the tiling experiments, I tested a different shader strategy: keep the
large projection in a single dispatch, but use a 16-lane local workgroup where
each invocation computes one output row. The intent was to reduce `groups_x`
below the PowerVR failure boundary without splitting the destination tensor.

## Result

This strategy failed immediately:

```text
MUL_MAT f32[1024,16] x f32[1024,1] -> f32[16,1]: FAIL
MUL_MAT f32[1024,128] x f32[1024,1] -> f32[128,1]: FAIL
MUL_MAT f32[1024,256] x f32[1024,1] -> f32[256,1]: FAIL
MUL_MAT f32[1024,1024] x f32[1024,1] -> f32[1024,1]: FAIL
```

The earlier single-thread scalar shader remains the only known correct shader
variant, and only for small row counts. The llama.cpp tree was restored to the
guarded baseline after this test.

## Next direction

The remaining likely options are:

- build a standalone Vulkan reproducer outside ggml to isolate whether this is
  descriptor setup, command-buffer ordering, or the PowerVR compiler;
- generate a family of very small fixed-shape shaders and route specific Qwen
  projections through them;
- abandon the generic Vulkan matvec path and write a PowerVR-specific tiled
  kernel with explicit row/block indexing validated in isolation first.
