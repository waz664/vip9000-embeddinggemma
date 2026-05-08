# PowerVR Vulkan LLM checkpoint: guarded matvec baseline

Date: 2026-05-08 23:50 UTC

Local llama.cpp branch: `radxa-powervr-vulkan`
Local commit: `302545b93bea0f728a6fb18f151e98f5b8de5ecb`

## What changed

This checkpoint makes the experimental PowerVR Vulkan backend more honest about which Qwen graph ops it can compute correctly.

Changes in the local llama.cpp patch:

- Keeps the earlier shared-memory and subgroup-size-1 workarounds.
- Fixes the custom scalar F32 matvec shader so B and D use the generated `B_TYPE` / `D_TYPE` instead of being hard-coded to `float`.
- Marks `B_IS_F16` for generated f16-B scalar shader variants.
- Declines known-bad PowerVR BXM shapes instead of advertising support and returning wrong tensors:
  - broadcast `MUL` for repeated/rank-3 Qwen prompt tensors
  - f16-left-hand `MUL_MAT` paths used by KV-cache attention
  - standard F32 prompt-batch matmul paths that fail pipeline creation
  - F32 scalar matvecs larger than the currently proven 16 output rows

## Verification run

Targeted backend probes on the Radxa Cubie A7S / Allwinner A733 / PowerVR BXM:

```text
MUL_MAT f32[1024,16] x f32[1024,1] -> f32[16,1]: OK
MUL_MAT f32[1024,128] x f32[1024,1] -> f32[128,1]: not supported [Vulkan0]
MUL_MAT f32[1024,1024] x f32[1024,1] -> f32[1024,1]: not supported [Vulkan0]
```

This is not yet a GPU-running Qwen layer. It is a safety baseline: llama.cpp should no longer schedule the known-corrupt PowerVR paths as if they were valid.

## Current hypothesis

The real Qwen F32 projection matvecs fail when the shader reads larger weight matrices from the original storage buffer. A descriptor offset/range slicing attempt did not fix this. The next implementation direction is to copy each weight-row slice into a small scratch buffer, then run the scalar matvec shader against that scratch buffer so the shader never reads the large original weight allocation directly.

## Next step

Implement and test a PowerVR-only scratch-buffer matvec path:

1. Allocate/reuse `prealloc_x` sized for one safe row slice.
2. Copy `src0` rows for that slice into scratch with `vkCmdCopyBuffer`.
3. Dispatch the scalar matvec against scratch and write into the real destination offset.
4. Validate exact Qwen graph ops again before attempting `llama-completion -ngl`.
