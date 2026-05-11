# Release Notes

## v0.2.5

- Added opt-in WebUI Think mode for Qwen3 requests.
- Normal WebUI requests explicitly disable thinking so fast chat remains the default.
- Think requests enable Qwen thinking, use a larger answer budget, and cap hidden reasoning so the model still has room for the final answer.
- Updated the llama.cpp service defaults to `CTX_SIZE=4096` and `--reasoning auto`.
- Added a reasoning-budget message so capped Think requests are pushed toward a final response.
- Made the service-managed WebUI execute the repo copy of `webui/app.py`, avoiding stale installed code.
- Refreshed README and docs so new users land on the current service-managed stack.

## v0.2.4

- Made the WebUI viewport fixed with a sticky top status bar and chat auto-scroll after prompts/responses.
- Added cumulative runtime stats to `/api/status` and the sidebar, including request count, web searches, added chunks, and LLM token totals.
- Added sidebar knowledge ingestion for file uploads and URLs, with same-site one-level link following for URLs.
- Added optional web-search context for Qwen, exposed by the Web checkbox in the chat form.
- Web search results are passed to the model with `[W1]` citation labels and returned in the API response.

## v0.2.3

- Added fixed WebUI RAG eval suite, `scripts/evaluate_webui_rag.py`, with 10 Radxa Cubie A7S hardware questions.
- Added exact-response cache keyed by query, index fingerprint, model/provider, retrieval settings, and ranked chunk fingerprints.
- Added hardware-term query expansion and lexical boosts for SoC, CPU, LPDDR5, NVMe, PCIe, USB-C, and DisplayPort.
- Added persistent VIPLite runner prototype in `tools/` and optional `EMBEDDINGGEMMA_VIP_RUNNER` integration.
- Validated persistent VIPLite output against `vpm_run` with `max_abs_diff=0.0`.
- Documented Qwen Q4/Q8 trials; Q4 Vulkan is unsafe on the current PowerVR driver, Q8 CPU beat Q4 CPU in the short local test.
- Latest WebUI eval results: `10/10` pass, cached median total `0.0038 s`.

## v0.2.2

- Added optional persistent embedding cache keyed by text and model asset fingerprints.
- Added batch embedding utility, `scripts/embed_batch.py`.
- Added embedding timing breakdown via `EMBEDDINGGEMMA_TIMING=1`.
- Made `rag_demo/build_index.py` use an index-local embedding cache by default.
- Added direct llama.cpp server benchmark utility, `scripts/benchmark_llama_server.py`.
- Added `CPUSET`, `THREADS`, and `THREADS_BATCH` runtime knobs to the llama.cpp launcher.
- Added safe passthrough for PowerVR op-family debug gates in the llama.cpp launcher.
- Documented the eight embedding/Qwen refinement changes and validation results.

## v0.2.1

- Added PowerVR patch `0005` with opt-in auxiliary op family gates.
- Tested RMS_NORM, SWIGLU, ROPE, same-shape elementwise, and combined RMS_NORM+SWIGLU.
- Confirmed the default projection-matvec-only PowerVR path remains fastest for the WebUI workload.
- Added runtime update helper for refreshing installed WebUI/runner code after `git pull`.
- Added optional OS tuning script and documented CPU governor/swappiness checks.
- Cached tokenizer, token embedding memmap, and dense-tail weights inside the embedding runner.
- Tested A76 pinning, higher GPU layer counts, and shorter RAG context; documented why they are not defaults.

## v0.2.0

- Added full-stack hobbyist quickstart for:
  - VIP9000 EmbeddingGemma NPU runtime
  - Radxa Cubie A7S vector index demo
  - patched llama.cpp PowerVR Vulkan server
  - RAG WebUI
- Added experimental llama.cpp PowerVR patches through `0004`, using a quality-first Qwen F16 projection matvec path.
- Added patch `0005` with opt-in PowerVR op-family debug gates for RMS_NORM, SWIGLU, ROPE, and same-shape elementwise ops.
- Added user/system systemd service installer:
  - `vip9000-llama.service`
  - `vip9000-rag-webui.service`
- Added one-command full-stack installer:
  - `install/full_stack_install.sh`
- Added WebUI exact-query embedding cache.
- Trimmed default RAG context to top-1 retrieved chunk for lower Qwen latency.
- Added WebUI benchmark helper:
  - `scripts/benchmark_webui.py`
- Verified service-managed WebUI benchmark on Cubie A7S:
  - first run: `33.18 s`, `embedding_cache_hit=False`
  - repeated run: `9.17 s`, `embedding_cache_hit=True`
- Documented CPU-only vs PowerVR hybrid benchmark results and current limitations.
- Documented auxiliary PowerVR op trials; RMS_NORM and SWIGLU are coherent but slower in the current WebUI workload, while ROPE and elementwise remain unsafe.

## v0.1.1

- Prebuilt runtime bundle now includes `tokenizer.model`.
- Runtime scripts no longer require `/home/radxa`.
- RAG chunking now filters more page chrome and uses overlapping chunks.
- WebUI now behaves as a chat interface and only injects KB context when retrieval clears a relevance threshold.
- Larger 16-chunk / 20-query quality benchmark passes current targets:
  - query cosine mean: `0.9460`
  - query cosine min: `0.9302`
  - overlap@5: `0.9300`
  - reference-top MRR: `0.8667`
- Added environment overrides:
  - `VIP9000_VPM_RUN`
  - `VIP9000_VIPLIB`
  - `EMBEDDINGGEMMA_TOKENIZER`
  - `EMBEDDINGGEMMA_TFLITE`
  - `VIP9000_RAG_MODEL_DIR`
  - `VIP9000_RAG_WORK_DIR`
  - `VIP9000_RAG_PORT`
- Added install, smoke-test, and WebUI launch scripts.
- Added troubleshooting documentation.

## v0.1.0

- Initial public release.
- FP32 `seq128` masked/bias hidden-state VIPLite NBG metadata.
- NPU embedding runner.
- RAG demo and CPU-vs-NPU benchmark scripts.
- Export and ONNX rewrite scripts.
