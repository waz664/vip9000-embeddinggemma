# Guarded llama.cpp Vulkan sanity result

Date: 2026-05-09 UTC

Local llama.cpp branch: `radxa-powervr-vulkan`
Local commit tested: `302545b93bea0f728a6fb18f151e98f5b8de5ecb`

## Command

```bash
env GGML_VK_VISIBLE_DEVICES=0 build-vulkan/bin/llama-completion \
  -m Qwen3-0.6B-F32-from-Q8.gguf \
  -p 'OK' -n 1 -c 64 -ngl 2 \
  --no-op-offload --no-kv-offload -fa off \
  --no-display-prompt --no-conversation --reasoning off \
  --temp 0 --seed 7 --no-perf
```

## Result

The run completed without the earlier Vulkan pipeline creation crash and produced the same fixed-greedy token text observed on CPU:

```text
ED
```

Important caveat: this is not yet meaningful GPU acceleration. With the guarded support rules, llama.cpp reported only about `0.01 MiB` in the Vulkan model buffer because the real Qwen projection matvecs are currently rejected instead of being allowed to return corrupt values.

## Current status

- Stable/safe baseline: yes.
- Large Qwen projection matvec on PowerVR GPU: not solved yet.
- Next direction: scratch-buffer or custom tiled shader path for F32 projection matrices, then re-enable Qwen layer projection offload only after exact exported graph ops pass.
