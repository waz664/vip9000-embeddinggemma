# RAG WebUI

The optional demo app is in:

```text
webui/app.py
webui/static/index.html
```

The recommended live copy is this repo checkout:

```text
~/vip9000-embeddinggemma
```

It uses:

- NPU EmbeddingGemma retrieval from `~/embeddinggemma_npu_seq128_bias_hidden_fp32`
- the directory-backed vector index in `rag_demo/index`
- Ollama model `qwen3:0.6b` by default, or a local llama.cpp server
- Python standard-library HTTP server, no Flask/FastAPI dependency
- a relevance threshold, `VIP9000_RAG_MIN_COSINE`, default `0.35`
- a persistent exact-query embedding cache, `VIP9000_RAG_QUERY_CACHE`, default enabled
- a persistent exact-response cache, `VIP9000_RAG_RESPONSE_CACHE`, default enabled
- a small hybrid retrieval boost for hardware spec terms such as SoC, LPDDR5, NVMe, PCIe, USB-C, and DisplayPort
- top-3 retrieved context by default, configurable with `VIP9000_RAG_TOP_K`
- a `VIP9000_RAG_MAX_TOKENS` answer budget, default `256`, so multi-part spec answers do not truncate early
- an optional Think checkbox that enables Qwen thinking with a larger final-answer budget
- cumulative runtime stats from `/api/status`, including request count and LLM token counts
- URL/file knowledge ingestion from the sidebar
- optional web search results for Qwen when the chat prompt enables web access

Start the recommended service-managed stack:

```bash
cd ~/vip9000-embeddinggemma
SCOPE=user bash install/install_systemd_services.sh
```

Open:

```text
http://<radxa-ip>:8080
```

The app sends the top three retrieved snippets to Qwen3 by default, and only uses KB context when the best retrieved chunk clears the relevance threshold. It also highlights obvious labeled fields, such as `Engine`, before the raw chunks so small models are less likely to answer from adjacent infobox fields. If no chunk looks relevant, it answers normally without KB citations. Raise `VIP9000_RAG_TOP_K` for harder questions that need more context. Larger prompts worked mechanically but were slower on this board. A representative earlier Ollama NVMe query completed in about:

```text
embedding_s=19.2
llm_s=93.3
total_s=112.4
```

The answer produced:

```text
Yes, the Cubie A7S supports NVMe storage. [1]
```

This is a complete local RAG path, but Qwen3 generation latency is still high on this board. For the highest-quality DIY demo, keep `qwen3:0.6b`, use the llama.cpp PowerVR provider, and accept the wait.

## Sidebar Tools

The sidebar can add knowledge without leaving the WebUI:

- `Index URL` fetches the supplied URL, extracts text, follows same-site links one level down, chunks the pages, embeds the chunks, and appends them to `rag_demo/index`.
- `Upload` accepts local text, Markdown, or HTML files and appends the extracted chunks to the same index.
- Adding knowledge clears the exact-response cache so future answers use the updated index.

Limits are controlled by:

```bash
VIP9000_RAG_INGEST_MAX_PAGES=8
VIP9000_RAG_INGEST_MAX_CHUNKS=96
```

The Web checkbox enables a lightweight web-search tool for the prompt. When enabled, the backend bypasses local KB retrieval, skips the NPU embedding step, searches the web, and passes web sources to Qwen with `[W1]` style citation labels. Leave Web unchecked for local/private KB answers.

Disable the web tool with:

```bash
VIP9000_RAG_WEB_SEARCH=0
```

The Think checkbox is for slower, higher-effort answers. Normal requests explicitly send
`enable_thinking=false` to llama.cpp and keep `VIP9000_RAG_MAX_TOKENS=256`. Think requests
send `enable_thinking=true`, use `VIP9000_RAG_THINK_MAX_TOKENS=768`, and cap hidden
reasoning with `VIP9000_RAG_THINK_REASONING_BUDGET=384` so the model has room to produce
the final answer. Think requests use `VIP9000_RAG_THINK_TIMEOUT_S=900` by default because
the small board can take several minutes on harder prompts. The WebUI strips any raw
`<think>...</think>` block before display.

## WebUI Eval

Run the fixed Radxa Cubie A7S eval suite:

```bash
python3 scripts/evaluate_webui_rag.py
```

Latest local result after the hybrid retrieval and cache changes:

```text
cases=10
passed=10
pass_rate=100.00%
cached median_total_s=0.0038
```

The uncached run is intentionally slower because new query wordings still execute the NPU embedding and Qwen generation paths. The cached eval confirms that exact repeated questions can bypass both the NPU embedding and LLM generation without changing the index or answer.

## llama.cpp PowerVR Provider

After applying the experimental llama.cpp patch in `patches/llama.cpp/`, start the local GPU-enabled server:

```bash
LLAMA_CPP_DIR="$HOME/llama.cpp" \
MODEL="$HOME/llama.cpp/Qwen3-0.6B-F16-from-Q8.gguf" \
bash install/run_llama_cpp_powervr_server.sh
```

Then start the WebUI against that server:

```bash
VIP9000_RAG_LLM_PROVIDER=llama_cpp \
VIP9000_RAG_LLAMA_CPP_URL=http://127.0.0.1:8081/v1/chat/completions \
bash install/run_webui.sh
```

This path uses `-ngl 2`, `-c 4096`, `-b 8`, `-ub 8`, CPU KV cache, flash attention disabled, `LLAMA_VK_NO_OUTPUT_OFFLOAD=1`, and the quality-first PowerVR matvec-only policy from patch `0004`.
The installed service uses `--reasoning auto` so the WebUI can toggle thinking per
request while normal prompts still run with thinking disabled. It also sets
`--reasoning-budget-message "Now provide the final answer."` so capped thinking requests
are pushed toward a final response.

Latest local smoke result:

```text
query="Does the Cubie A7S support NVMe?"
answer="Yes, the Cubie A7S supports NVMe storage via PCIe 3.0 x1 expansion. [2]"
embedding_s=18.99
llm_s=63.03
total_s=82.03
provider=llama_cpp
model=qwen3-0.6b-powervr
```

Repeated exact queries use the WebUI query embedding cache. A validated repeat query skipped the NPU embedding step and completed in `9.66 s` total:

```text
embedding_cache_hit=true
response_cache_hit=false
embedding_s=0.0007
llm_s=9.65
total_s=9.66
```

With the response cache enabled, exact repeated questions can return without calling Qwen:

```text
embedding_cache_hit=true
response_cache_hit=true
embedding_s=0.0006
llm_s=0.00
total_s=0.0038 median over the 10-case eval
```

An earlier top-1 latency trial through the llama.cpp PowerVR provider measured:

```text
embedding_cache_hit=true
embedding_s=0.0007
llm_s=11.18
total_s=11.19
```
