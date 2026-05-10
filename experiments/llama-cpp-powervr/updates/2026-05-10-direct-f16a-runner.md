# 2026-05-10: Direct PowerVR F16-weight matvec runner

## Summary

The adjacent direct Vulkan runner now supports:

- F32 weights x F32 vector -> F32 output.
- F16 weights x F32 vector -> F32 output.
- Device-local arena with explicit staging upload/readback.
- Nonzero descriptor offsets.
- One descriptor set per row chunk.
- Explicit transfer/compute/transfer barriers.

This is the shape we want for a custom PowerVR matvec backend. It continues to
pass wider rows that ggml's integrated Vulkan path cannot run reliably.

## Commands

```bash
cd /home/radxa/vip9000-embeddinggemma
experiments/llama-cpp-powervr/repro/run_matvec_repro.sh 512 1024 f32
experiments/llama-cpp-powervr/repro/run_matvec_repro.sh 512 1024 f16a
experiments/llama-cpp-powervr/repro/run_matvec_repro.sh 1024 1024 f16a
```

Observed:

```text
mode=f32 rows=512 cols=1024 max_err=7.06204e-05 max_row=274
mode=f16a rows=512 cols=1024 max_err=7.43212e-05 max_row=384
mode=f16a rows=1024 cols=1024 max_err=8.83967e-05 max_row=623
```

## Interpretation

The PowerVR BXM hardware and driver can execute the target matvec shape for both
F32 and F16-weight inputs when the command stream is controlled directly. The
remaining work is to either:

- call this direct path from llama.cpp for selected Qwen projections, or
- make ggml's Vulkan backend emit an equivalent command stream for PowerVR.
