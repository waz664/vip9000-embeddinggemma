# 2026-05-10: Real Qwen projections through direct PowerVR matvec

## Summary

The direct PowerVR runner now executes real F16 projection matrices extracted
from `Qwen3-0.6B-F16-from-Q8.gguf`, using an F32 input vector and F32 output.
This is the first selected-Qwen-op validation for the custom direct path.

## Extractor

Use:

```bash
cd experiments/llama-cpp-powervr/repro
./export_gguf_tensor_for_matvec.py \
  /path/to/Qwen3-0.6B-F16-from-Q8.gguf \
  blk.0.attn_k.weight \
  /tmp/powervr_qwen \
  --gguf-py /path/to/llama.cpp/gguf-py
```

Then run the printed `run_matvec_repro.sh` command.

## Results

All of these are real first-block Qwen tensors:

```text
blk.0.attn_k.weight       mode=f16a rows=1024 cols=1024 max_err=2.23198e-06
blk.0.attn_q.weight       mode=f16a rows=2048 cols=1024 max_err=3.19418e-06
blk.0.attn_output.weight  mode=f16a rows=1024 cols=2048 max_err=6.55169e-06
blk.0.ffn_gate.weight     mode=f16a rows=3072 cols=1024 max_err=2.47334e-06
blk.0.ffn_up.weight       mode=f16a rows=3072 cols=1024 max_err=1.53787e-06
blk.0.ffn_down.weight     mode=f16a rows=1024 cols=3072 max_err=5.47042e-06
```

## Interpretation

The direct path is now validated on the actual Qwen projection shapes that
matter for layer execution. This is stronger than the earlier synthetic matvec
tests and supports continuing toward a direct PowerVR projection runner or a
llama.cpp integration that bypasses the unstable ggml Vulkan matvec dispatch
path on this GPU.
