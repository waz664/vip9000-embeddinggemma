# Performance Notes

These measurements are from the Radxa Cubie A7S using the WebUI `/api/chat` endpoint and the query:

```text
Does the Cubie A7S support NVMe?
```

Run the local benchmark helper with:

```bash
python3 scripts/benchmark_webui.py --runs 2
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

## Query Embedding Cache

The WebUI now has a persistent exact-query embedding cache under:

```text
<runtime>/webui_work/query_cache/
```

It is enabled by default and can be disabled with:

```bash
VIP9000_RAG_QUERY_CACHE=0
```

This does not change retrieval quality for repeated exact queries. It reuses the NPU-generated query vector and avoids launching the NPU runner again.

Validation on the same NVMe query:

| Path | Embedding Cache | Embedding | llama.cpp | Total | Answer |
| --- | --- | ---: | ---: | ---: | --- |
| PowerVR hybrid WebUI | miss | 19.05 s | 62.20 s | 81.26 s | correct, cited |
| PowerVR hybrid WebUI | hit | 0.0007 s | 9.65 s | 9.66 s | correct, cited |

This is the largest current usability improvement for repeated or common questions. New questions still pay the NPU embedding cost.

## Prompt Context Trim

The WebUI now defaults to:

```text
VIP9000_RAG_TOP_K=1
VIP9000_RAG_CONTEXT_CHARS=450
```

The vector search still computes ranked candidates, but only the best chunk is sent to Qwen by default. Users can raise `VIP9000_RAG_TOP_K` for harder questions that need more context.

Validation on the NVMe query with the query embedding already cached:

| WebUI Context | llama.cpp State | Embedding | llama.cpp | Total | Answer |
| --- | --- | ---: | ---: | ---: | --- |
| top-2, 450 chars each | cold prompt cache | 0.0007 s | 51.81 s | 51.82 s | correct, cited |
| top-2, 450 chars each | warm prompt cache | 0.0007 s | 13.16 s | 13.16 s | correct, cited |
| top-1, 450 chars | cold prompt cache | 0.0007 s | 34.86 s | 34.86 s | correct, cited |
| top-1, 450 chars | warm prompt cache | 0.0007 s | 11.18 s | 11.19 s | correct, cited |

For this small Qwen model, shorter retrieved context is a clear latency win. The default is therefore tuned for quick, source-backed answers rather than maximal recall.
