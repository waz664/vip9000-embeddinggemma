# PowerVR Vulkan LLM checkpoint: guarded matvec baseline

Date: 2026-05-08 23:50 UTC

Local llama.cpp branch: `radxa-powervr-vulkan`
Local commit: `302545b93bea0f728a6fb18f151e98f5b8de5ecb`

## What changed

This checkpoint makes the experimental PowerVR Vulkan backend more honest about which Qwen graph ops it can compute correctly.

- Fixes the custom scalar F32 matvec shader so B and D use generated `B_TYPE` / `D_TYPE`.
- Marks `B_IS_F16` for generated f16-B scalar shader variants.
- Declines known-bad PowerVR BXM shapes instead of advertising support and returning wrong tensors.
- Keeps large Qwen F32 projections on CPU until the scratch-buffer matvec path is validated.

## Verification

```text
MUL_MAT f32[1024,16] x f32[1024,1] -> f32[16,1]: OK
MUL_MAT f32[1024,128] x f32[1024,1] -> f32[128,1]: not supported [Vulkan0]
MUL_MAT f32[1024,1024] x f32[1024,1] -> f32[1024,1]: not supported [Vulkan0]
```

This is not yet a GPU-running Qwen layer. It is a safety baseline before trying a scratch-buffer matvec implementation.
