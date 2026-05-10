# PowerVR llama.cpp Status

This repo includes an experimental llama.cpp patch for Radxa Cubie A7S / Allwinner A733 PowerVR BXM Vulkan.

Patch:

```bash
patches/llama.cpp/0001-vulkan-enable-powervr-scalar-f16-matvec-offload.patch
```

Tested local result:

```bash
cd /home/radxa/llama.cpp
env GGML_VK_VISIBLE_DEVICES=0 ./build-vulkan/bin/llama-completion \
  -m /home/radxa/llama.cpp/Qwen3-0.6B-F16-from-Q8.gguf \
  -p 'Say OK.' \
  -n 4 -c 64 -b 8 -ub 8 -ngl 2 \
  --no-kv-offload -fa off \
  --no-display-prompt --no-conversation --reasoning off \
  --temp 0.2 --top-p 0.9 --no-perf
```

The important signal is that llama.cpp now:

- loads about 30 MiB of real Qwen F16 layer weights into the PowerVR Vulkan model buffer
- creates the F16 matvec pipeline successfully
- completes generation through the Vulkan path

Current limits:

- only small token-batch F16 matvec is enabled on PowerVR
- prompt batches should stay small, for example `-b 8 -ub 8`
- KV cache and flash attention are kept off for this path
- generated text still needs correctness work before this is a recommended chat backend

The patch works by adding a scalar F16-weight/F32-vector matvec shader matching the standalone Vulkan repro and avoiding the subgroup/RTE SPIR-V mutations that the PowerVR driver rejects for this pipeline.
