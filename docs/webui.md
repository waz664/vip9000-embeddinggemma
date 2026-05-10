# RAG WebUI

The optional demo app is in:

```text
webui/app.py
webui/static/index.html
```

On the Radxa board, the live copy is:

```text
~/rag_webui
```

It uses:

- NPU EmbeddingGemma retrieval from `~/embeddinggemma_npu_seq128_bias_hidden_fp32`
- the directory-backed vector index in `rag_demo/index`
- Ollama model `qwen3:0.6b` by default, or a local llama.cpp server
- Python standard-library HTTP server, no Flask/FastAPI dependency
- a relevance threshold, `VIP9000_RAG_MIN_COSINE`, default `0.35`

Start it:

```bash
cd ~/rag_webui
./app.py
```

Open:

```text
http://<radxa-ip>:8080
```

The app intentionally sends only the top 2 retrieved snippets to Qwen3, and only when the best retrieved chunk clears the relevance threshold. If no chunk looks relevant, it answers normally without KB citations. Larger prompts worked mechanically but were too slow on this board through Ollama. A representative NVMe query completed in about:

```text
embedding_s=19.2
llm_s=93.3
total_s=112.4
```

The answer produced:

```text
Yes, the Cubie A7S supports NVMe storage. [1]
```

This is a complete local RAG path, but the Qwen3 generation latency is high. For a more usable interactive demo, try `gemma3:270m` or reduce retrieved context further. For a quality-oriented demo, keep `qwen3:0.6b` and accept the wait.

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

This path uses `-ngl 2`, `-b 8`, `-ub 8`, CPU KV cache, and flash attention disabled. It is a GPU-offload milestone, not yet the recommended quality path.
