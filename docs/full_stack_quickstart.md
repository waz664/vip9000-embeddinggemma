# Full Stack Quickstart

This guide is for a hobbyist starting from a Radxa Cubie A7S / Allwinner A733 board and wanting the complete local demo:

- EmbeddingGemma-300M embeddings on the Vivante VIP9000-series NPU
- a tiny directory-backed vector index for Radxa Cubie A7S docs
- Qwen3 0.6B served by llama.cpp with the experimental PowerVR Vulkan matvec path
- the local RAG WebUI on port `8080`

The current stack is a hybrid. The NPU runs the embedding transformer body. PowerVR runs selected Qwen F16 projection matvecs. CPU handles tokenization, retrieval glue, attention/output/other fragile Qwen ops, and WebUI logic.

## 1. Board Assumptions

Known working board:

```text
Radxa Cubie A7S
Allwinner A733
2x Cortex-A76 + 6x Cortex-A55
Vivante VIP9000-series NPU
PowerVR B-Series BXM-4-64 MC1 GPU
```

Useful sanity checks:

```bash
lscpu
ls -l /dev/vipcore
vulkaninfo --summary
```

On this board, CPUs `6,7` are the faster Cortex-A76 cores. For long local compiles, use:

```bash
taskset -c 6,7 <build command>
```

## 2. Clone The Repo

```bash
git clone https://github.com/waz664/vip9000-embeddinggemma.git
cd vip9000-embeddinggemma
```

## 3. Install Python Dependencies

```bash
bash install/setup_radxa_dependencies.sh
```

This installs the Python packages from `requirements.txt` and checks for the VIPLite paths used by the NPU runner.

The scripts expect the Radxa/Allwinner AI SDK at:

```text
~/ai-sdk/examples/vpm_run/vpm_run
~/ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v2.0
```

If your SDK is elsewhere:

```bash
export AI_SDK_DIR=/path/to/ai-sdk
export VIP9000_VPM_RUN=/path/to/vpm_run
export VIP9000_VIPLIB=/path/to/viplite-tina/lib/aarch64-none-linux-gnu/v2.0
```

## 4. Install The Prebuilt NPU Runtime

The model/runtime bundle is too large for normal git. Install it from the GitHub release asset:

```bash
bash install/install_runtime.sh
bash install/smoke_test.sh
```

Default install location:

```text
~/embeddinggemma_npu_seq128_bias_hidden_fp32
```

Expected files there include:

```text
network_binary.nb
nbg_meta.json
token_embedding_fp16.dat
dense_2_weight_f32.npy
dense_3_weight_f32.npy
tokenizer.model
embed_text_bias_hidden_npu.py
rag_demo/
webui/
```

If you downloaded the runtime bundle manually:

```bash
LOCAL_ASSET=/path/to/vip9000-embeddinggemma-a7s-seq128-fp32-v0.1.1.tar.zst \
  bash install/install_runtime.sh
```

## 5. Test One NPU Embedding

```bash
cd ~/embeddinggemma_npu_seq128_bias_hidden_fp32
python3 ./embed_text_bias_hidden_npu.py "does Cubie A7S support NVMe storage?"
```

Success looks like:

```text
embedding_norm=1
embedding_first16=...
```

## 6. Build The Demo Vector Index

```bash
cd ~/embeddinggemma_npu_seq128_bias_hidden_fp32/rag_demo
python3 ./build_index.py --max-chunks 8
python3 ./search_index.py "does Cubie A7S support NVMe?"
```

This creates:

```text
rag_demo/index/chunks.json
rag_demo/index/embeddings.npy
rag_demo/index/sources.json
```

This is intentionally a tiny directory-backed vector store, not a database service.

## 7. Build Patched llama.cpp For PowerVR

Install normal build dependencies first. Package names vary by image, but this is the usual set:

```bash
sudo apt update
sudo apt install -y git cmake build-essential python3 python3-pip vulkan-tools glslc
```

Then build llama.cpp with this repo's patches:

```bash
cd ~/vip9000-embeddinggemma
taskset -c 6,7 bash install/build_llama_cpp_powervr.sh
```

By default this clones or uses:

```text
~/llama.cpp
~/llama.cpp/build-vulkan
```

If your Vulkan headers, loader, or `glslc` are not in standard locations, pass them explicitly:

```bash
Vulkan_INCLUDE_DIR=/path/to/include \
Vulkan_LIBRARY=/path/to/libvulkan.so \
Vulkan_GLSLC_EXECUTABLE=/path/to/glslc \
taskset -c 6,7 bash install/build_llama_cpp_powervr.sh
```

The build helper also applies the local shader-stub relink workaround if the PowerVR Vulkan build hits missing generated shader symbols.

## 8. Put Qwen3 0.6B GGUF In llama.cpp

Place your Qwen3 0.6B F16 GGUF at:

```text
~/llama.cpp/Qwen3-0.6B-F16-from-Q8.gguf
```

Or set `MODEL=/path/to/model.gguf` when starting the server.

## 9. Start The llama.cpp PowerVR Server

```bash
cd ~/vip9000-embeddinggemma
setsid -f bash -c 'exec bash install/run_llama_cpp_powervr_server.sh > /tmp/llama-server-powervr.log 2>&1'
```

Stable current defaults:

```text
host: 0.0.0.0
port: 8081
context: 512
batch: 8
ubatch: 8
gpu layers: 2
KV cache: CPU
flash attention: off
output layer: CPU via LLAMA_VK_NO_OUTPUT_OFFLOAD=1
warmup: disabled
```

Smoke test:

```bash
curl -s http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-0.6b-powervr","messages":[{"role":"user","content":"What is the capital of France? Answer briefly."}],"max_tokens":24,"temperature":0.2}'
```

Expected answer:

```text
The capital of France is Paris.
```

## 10. Start The RAG WebUI

```bash
cd ~/vip9000-embeddinggemma
VIP9000_RAG_LLM_PROVIDER=llama_cpp \
VIP9000_RAG_LLAMA_CPP_URL=http://127.0.0.1:8081/v1/chat/completions \
bash install/run_webui.sh
```

Open:

```text
http://<radxa-ip>:8080
```

To install both the llama.cpp server and WebUI as systemd user services:

```bash
bash install/install_systemd_services.sh
```

For a single command that installs dependencies, runtime files, the index, the patched llama.cpp build, services, and a benchmark:

```bash
bash install/full_stack_install.sh
```

See `docs/services.md` for service management and configuration.

API smoke test:

```bash
curl -s http://127.0.0.1:8080/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"Does the Cubie A7S support NVMe?"}'
```

Benchmark two runs, showing the cache effect:

```bash
python3 scripts/benchmark_webui.py --runs 2
```

Verified service-managed result on this board:

```text
run=1 total=33.18s embedding_cache_hit=False
run=2 total=9.17s  embedding_cache_hit=True
```

Known local cold-server PowerVR result:

```text
answer="Yes, the Cubie A7S supports NVMe storage via PCIe 3.0 x1 expansion. [2]"
embedding_s=18.99
llm_s=63.03
total_s=82.03
```

The user's later repeated run showed:

```text
embedding_s=19.0
llm_s=12.2
total_s=31.2
```

That faster number is plausible when the model server is warm, the answer is shorter, prompt cache helps, or less retrieved context is used.

For the latest CPU-only vs PowerVR comparison, see:

```text
docs/performance.md
```

The WebUI caches exact query embeddings by default. Repeating the same query can skip the roughly 19 second NPU embedding step:

```text
first query:  embedding_cache_hit=false
repeat query: embedding_cache_hit=true
```

Disable this with `VIP9000_RAG_QUERY_CACHE=0` when benchmarking uncached retrieval.

The WebUI also sends only the top retrieved chunk to Qwen by default. Increase context for harder questions with:

```bash
VIP9000_RAG_TOP_K=2 VIP9000_RAG_CONTEXT_CHARS=700 bash install/run_webui.sh
```

## 11. What The Current PowerVR Path Does

The llama.cpp patches are intentionally conservative:

```text
PowerVR Vulkan: selected Qwen F16 projection matvecs
CPU: attention matvecs, softmax, output projection, embedding lookup, KV writes, norms, ROPE, SWIGLU, elementwise ops
```

This is not "the whole LLM on GPU." It is a stable hybrid that keeps quality intact and moves known-good matvec work to PowerVR.

Use this to inspect the exact state:

```bash
cat docs/powervr-llama.md
```

## 12. Common Problems

If `/dev/vipcore` is missing, the NPU driver is not loaded or the image does not include the VIPLite driver.

If `llama-server` exits immediately when detached with `nohup`, use the `setsid -f bash -c 'exec ...'` launch pattern shown above.

If WebUI returns a llama.cpp 400 error about context size, increase `CTX_SIZE` for the server:

```bash
CTX_SIZE=768 bash install/run_llama_cpp_powervr_server.sh
```

If Qwen output becomes corrupt, return to the stable defaults:

```text
GPU_LAYERS=2
CTX_SIZE=512
BATCH_SIZE=8
UBATCH_SIZE=8
LLAMA_VK_NO_OUTPUT_OFFLOAD=1
```

If the Vulkan build fails with missing IQ4/MXFP/NVFP shader symbols:

```bash
bash install/fix_llama_cpp_vulkan_shader_stubs.sh
cmake --build ~/llama.cpp/build-vulkan --target llama-server llama-completion test-backend-ops -- -j1
```

More troubleshooting notes are in `docs/troubleshooting.md`.
