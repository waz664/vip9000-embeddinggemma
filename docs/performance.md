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

## Systemd-Managed Stack

After installing the user services with `install/install_systemd_services.sh`, the same benchmark helper measured:

```text
run=1 wall=33.20s embedding=19.0452s llm=14.13s total=33.18s embedding_cache_hit=False
run=2 wall=9.17s  embedding=0.0007s  llm=9.16s  total=9.17s  embedding_cache_hit=True
```

This is the current best end-to-end user experience: the first new question still pays the NPU embedding cost, while repeated exact questions return in about 9 seconds on this board.

## PowerVR Auxiliary Op Trials

Patch `0005` added opt-in gates for testing one extra Vulkan op family at a time:

```text
GGML_VK_POWERVR_ALLOW_RMS_NORM=1
GGML_VK_POWERVR_ALLOW_SWIGLU=1
GGML_VK_POWERVR_ALLOW_ROPE=1
GGML_VK_POWERVR_ALLOW_ELEMENTWISE=1
```

Results:

| Family | Op Check | Generation | Decision |
| --- | --- | --- | --- |
| RMS_NORM | pass | coherent | keep opt-in |
| SWIGLU | pass | coherent | keep opt-in |
| ROPE | fails Q-cur shapes with `ERR = inf` | not promoted | unsafe |
| same-shape elementwise | pass | corrupt output | unsafe |

Combined `RMS_NORM + SWIGLU` WebUI benchmark:

```text
run=1 wall=55.07s embedding=19.2015s llm=35.85s total=55.06s embedding_cache_hit=False
run=2 wall=12.47s embedding=0.0007s llm=12.46s total=12.46s embedding_cache_hit=True
```

That is slower than the default projection-matvec-only service benchmark:

```text
run=1 total=33.18s
run=2 total=9.17s
```

Conclusion: do not enable RMS_NORM or SWIGLU by default yet. The extra GPU ops are quality-correct, but the additional graph splits and synchronization cost outweigh their compute savings.

## A76 Pinning Trial

Pinning `llama-server` to CPUs `6,7` and using two llama.cpp threads was tested as an OS-level tuning option:

```text
run=1 wall=57.31s embedding=19.0720s llm=38.22s total=57.29s embedding_cache_hit=False
run=2 wall=8.19s  embedding=0.0007s  llm=8.18s  total=8.19s  embedding_cache_hit=True
```

This is a tradeoff, not a default improvement. Cached repeated queries improved slightly, but first/cold queries regressed badly compared with the default service-managed result.

## Embedding Runtime Object Cache

The NPU embedding Python runner now caches process-local objects:

- SentencePiece processor
- token embedding memmap
- dense projection tail weights

This does not change model output. It avoids reloading those objects inside long-running processes such as the WebUI.

Direct embedding benchmark after the change:

```text
call 1: 18.825s
call 2: 18.193s
```

The NPU execution is still the dominant cost, but repeated uncached queries in the same WebUI process avoid some Python/model-file overhead.

## Embedding Persistent Cache And Batch Tool

The embedding runner now also has an optional persistent cache controlled by:

```bash
EMBEDDINGGEMMA_EMBED_CACHE_DIR=/path/to/cache
```

The cache key includes the input text, model metadata, tokenizer, NBG, token embedding table, and dense-tail files. It is safe to reuse across processes and invalidates when model assets change.

Validated with `scripts/embed_batch.py`:

```text
first pass:
  text 1: 18.844s
  text 2: 18.188s
cached pass:
  text 1: 0.002s
  text 2: 0.001s
max_abs_diff=0.0
```

`rag_demo/build_index.py` now defaults to an index-local embedding cache:

```text
rag_demo/index/embedding_cache/
```

That makes repeated index builds and source refreshes much less wasteful.

## Embedding Timing Breakdown

Set:

```bash
EMBEDDINGGEMMA_TIMING=1
```

Example timing:

```text
token_s=0.595266 inputs_s=0.008721 npu_s=18.175598 tail_s=0.058137 total_s=18.843525
token_s=0.000308 inputs_s=0.005333 npu_s=18.171834 tail_s=0.006416 total_s=18.188001
```

The NPU runner call dominates. Python-side caching helps, but the main remaining embedding target is reducing or avoiding the `vpm_run` execution cost.

## GPU Layer Count Trial

The stable service default is `-ngl 2`, which keeps one repeating layer's projection matvecs on PowerVR because the output layer is forced to CPU with `LLAMA_VK_NO_OUTPUT_OFFLOAD=1`.

Higher offload counts were retested with the current quality-first path:

| GPU Layers | Vulkan Model Buffer | Graph Splits | First Run Total | Repeated Run Total | Result |
| ---: | ---: | ---: | ---: | ---: | --- |
| `-ngl 2` | 30 MiB | 11 | 33.18 s | 9.17 s | default |
| `-ngl 3` | 60 MiB | 21 | 81.84 s | 16.84 s | slower |
| `-ngl 4` | 90 MiB | 31 | 107.87 s | 20.81 s | slower |

Generation quality stayed coherent, but the additional graph splits dominate. Keep `GPU_LAYERS=2` for the WebUI service until the Vulkan scheduler/synchronization overhead is reduced.

## Context Length Floor

Reducing the top-1 snippet from 450 characters to 300 characters was tested:

```text
run=1 wall=44.26s embedding=18.8417s llm=25.40s total=44.24s
run=2 wall=5.66s  embedding=0.0008s  llm=5.65s  total=5.65s
```

It was faster, but it produced a wrong answer for the NVMe query:

```text
The Cubie A7S does not support NVMe. [1]
```

So `VIP9000_RAG_CONTEXT_CHARS=450` remains the default quality floor for this demo index.
