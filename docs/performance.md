# Performance Notes

These measurements are from the Radxa Cubie A7S using the WebUI `/api/chat` endpoint and the query:

```text
Does the Cubie A7S support NVMe?
```

The WebUI path includes:

- NPU EmbeddingGemma query embedding
- directory-backed vector search over the Radxa Cubie A7S docs
- llama.cpp Qwen3 0.6B answer generation
- source citation formatting

## 2026-05-11 WebUI CPU vs PowerVR Check

Both llama.cpp servers used:

```text
context: 512
batch: 8
ubatch: 8
KV cache: CPU
flash attention: off
warmup: disabled
```

CPU-only used `-ngl 0`.

PowerVR hybrid used:

```text
-ngl 2
LLAMA_VK_NO_OUTPUT_OFFLOAD=1
quality-first patch 0004
```

Results:

| Path | Server State | Embedding | llama.cpp | Total | Answer |
| --- | --- | ---: | ---: | ---: | --- |
| CPU-only | cold after restart | 18.97 s | 39.39 s | 58.37 s | correct, cited |
| CPU-only | repeated warm query | 18.98 s | 6.70 s | 25.68 s | correct, cited |
| PowerVR hybrid | cold after restart | 18.97 s | 61.23 s | 80.20 s | correct, cited |
| PowerVR hybrid | repeated warm query | 18.97 s | 9.71 s | 28.68 s | correct, cited |

Earlier warm user-observed hybrid timing was:

```text
embedding 19.0s · llama.cpp 12.2s · total 31.2s
```

## Interpretation

The PowerVR path is stable and quality-correct for this RAG workload, but it is not yet a universal latency win. For this specific repeated WebUI query, CPU-only Qwen generation was faster than the conservative PowerVR hybrid path.

The hybrid path is still useful because it proves Qwen projection matvecs can run correctly on the PowerVR Vulkan driver, and it spreads work across NPU, GPU, and CPU. The current implementation pays overhead from graph splits, CPU/GPU synchronization, and a deliberately narrow quality-first offload policy.

Optimization should focus on:

1. reducing the NPU embedding fixed cost, currently about `19 s`
2. reducing llama.cpp graph split/synchronization overhead
3. re-enabling one additional safe Vulkan op family at a time
4. preserving full generation quality as the main gate

Do not use isolated `test-backend-ops` success alone as the quality gate. Earlier broad PowerVR enablement passed isolated op checks and still corrupted generation.
