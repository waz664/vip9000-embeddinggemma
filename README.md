# EmbeddingGemma on Vivante VIP9000 NPU

This repository documents and packages a working EmbeddingGemma-300M embedding path for the Radxa Cubie A7S / Allwinner A733 Vivante VIP9000-series NPU.

The working deployment uses:

- FP32 `seq128` transformer graph compiled to VIPLite NBG
- two NPU inputs: token embeddings and additive attention bias
- NPU transformer body
- CPU tokenization, embedding lookup, masked pooling, dense projection tail, and L2 normalization
- a small directory-backed RAG demo for Radxa Cubie A7S product/docs pages

## Start Here

For a complete hobbyist-level walkthrough, including the prebuilt NPU runtime, patched llama.cpp PowerVR build, Qwen server, and RAG WebUI, follow:

```text
docs/full_stack_quickstart.md
```

For systemd service setup and the one-command installer, see:

```text
docs/services.md
```

For optional dedicated-board CPU governor and swappiness tuning, see:

```text
docs/os_tuning.md
```

## Known Working Setup

- Board: Radxa Cubie A7S
- SoC: Allwinner A733
- NPU: Vivante VIP9000-series, `VIP9000NANODI_PLUS_PID0X1000003B`
- VIPLite driver seen during testing: `2.0.3.2-AW-2024-08-30`
- Runtime path expected by default: `~/ai-sdk`

The important result is that adding the attention-bias input restored most of the original model quality. The previous unmasked NPU graph was only about `0.747` cosine against official CPU TFLite output for a representative query. The corrected masked/bias graph measured `0.944`.

## Current Results

Single-query benchmark on Radxa Cubie A7S:

| Path | Wall Time | CPU Time | Cosine vs Official TFLite |
| --- | ---: | ---: | ---: |
| CPU TFLite, 1 thread | 28.142 s | 25.975 s | 1.000 |
| CPU TFLite, 4 threads | 13.354 s | 40.977 s | 1.000 |
| old NPU, unmasked hidden FP32 seq128 | 23.665 s | 4.236 s | 0.747 |
| corrected NPU, masked/bias hidden FP32 seq128 | 19.331 s | 3.169 s | 0.944 |

Larger retrieval A/B eval, official CPU TFLite reference vs corrected NPU:

| Metric | Value |
| --- | ---: |
| chunks | 16 |
| queries | 20 |
| document cosine mean | 0.9364 |
| document cosine min | 0.8869 |
| query cosine mean | 0.9460 |
| query cosine min | 0.9302 |
| top-1 match rate | 0.7500 |
| overlap@5 mean | 0.9300 |
| reference top-1 MRR in NPU ranking | 0.8667 |

The retrieval result meets the current quality target: query cosine mean >= `0.93`, query cosine min >= `0.90`, overlap@5 >= `0.85`, and reference-top MRR >= `0.70`.

The WebUI includes a persistent exact-query embedding cache, exact-response cache, and a small hybrid retrieval boost for hardware spec terms. The fixed WebUI eval suite now passes `10/10` Radxa Cubie A7S questions. With both caches hot, the median eval response time was about `0.004 s`; new questions still pay the NPU embedding and Qwen generation costs.

The embedding runner also caches tokenizer and dense-tail objects inside long-running processes to avoid repeated Python-side model-file loads.

An experimental persistent VIPLite runner is available in `tools/`. It loads `network_binary.nb` once and can serve repeated embedding requests over stdin/stdout. The WebUI launcher automatically uses it when `tools/persistent_viplite_runner` has been built.

Additional utilities:

```text
scripts/embed_batch.py
scripts/benchmark_llama_server.py
scripts/evaluate_webui_rag.py
tools/build_persistent_viplite_runner.sh
```

The batch embedding helper supports a persistent embedding cache with `EMBEDDINGGEMMA_EMBED_CACHE_DIR`.

## What Is Not Committed

The following files are intentionally not included:

- `network_binary.nb` - generated VIPLite NBG, about 404 MB
- `token_embedding_fp16.dat` - EmbeddingGemma token embedding table, about 384 MB
- `dense_2_weight_f32.npy` and `dense_3_weight_f32.npy` - dense tail weights, about 9 MB each
- Google model weights and TFLite model files
- Vivante/Radxa SDK files and Docker images

Those files are large and/or governed by upstream licenses. This repo contains the code, export scripts, metadata, and reproducibility notes.

## Install With Prebuilt Runtime

Once the release asset is uploaded, users do not need to convert the model themselves:

```bash
git clone https://github.com/waz664/vip9000-embeddinggemma.git
cd vip9000-embeddinggemma
bash install/setup_radxa_dependencies.sh
bash install/install_runtime.sh
bash install/smoke_test.sh
```

The installer downloads:

```text
vip9000-embeddinggemma-a7s-seq128-fp32-v0.1.1.tar.zst
```

from the GitHub release and installs the runtime files under:

```text
~/embeddinggemma_npu_seq128_bias_hidden_fp32
```

See `docs/prebuilt_runtime.md` for bundle creation and release upload instructions.
See `docs/troubleshooting.md` if the NPU device, VIPLite runtime, or tokenizer is not found.

## Runtime Layout

On the Radxa board, place the runtime files in one directory:

```text
embeddinggemma_npu_seq128_bias_hidden_fp32/
  network_binary.nb
  nbg_meta.json
  token_embedding_fp16.dat
  dense_2_weight_f32.npy
  dense_3_weight_f32.npy
  embed_text_bias_hidden_npu.py
  benchmark_bias_vs_cpu.py
  rag_demo/
```

`nbg_meta.json` in `artifacts/` shows the compiled NBG interface:

```text
inputs_embeds:  [1, 128, 768], float32
attention_bias: [1, 1, 1, 128], float32
output hidden:  [1, 128, 768], float32
```

The attention bias is `0.0` for real tokens and `-10000.0` for padding tokens.

## Run One Embedding

```bash
cd ~/embeddinggemma_npu_seq128_bias_hidden_fp32
./embed_text_bias_hidden_npu.py "does Cubie A7S support NVMe storage?"
```

The script uses:

```bash
LD_LIBRARY_PATH=~/ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v2.0
~/ai-sdk/examples/vpm_run/vpm_run
```

Optional persistent VIPLite runner:

```bash
bash tools/build_persistent_viplite_runner.sh
EMBEDDINGGEMMA_VIP_RUNNER="$PWD/tools/persistent_viplite_runner" \
  ./embed_text_bias_hidden_npu.py "does Cubie A7S support NVMe storage?"
```

## Run The RAG Demo

```bash
cd ~/embeddinggemma_npu_seq128_bias_hidden_fp32/rag_demo
./build_index.py --max-chunks 12
./search_index.py "does Cubie A7S have NVMe support?"
```

The demo stores:

```text
index/chunks.json
index/embeddings.npy
index/sources.json
```

This is a tiny directory-backed vector index, not a full vector database.

## Quality Eval

```bash
cd ~/embeddinggemma_npu_seq128_bias_hidden_fp32/rag_demo
./evaluate_retrieval_quality.py --max-chunks 8 --top-k 5 --threads 4
```

The latest summary from this board is included at:

```text
artifacts/quality_eval_summary_v0.1.1.json
```

## Ollama RAG WebUI

An optional end-to-end demo is included in `webui/`. It uses the corrected NPU embedding path for retrieval and `qwen3:0.6b` through Ollama for answer generation.

```bash
cd ~/rag_webui
./app.py
```

Open `http://<radxa-ip>:8080`.

See `docs/webui.md` for notes and measured timing.

## Experimental PowerVR llama.cpp Patch

`patches/llama.cpp/` contains experimental llama.cpp Vulkan patches for the Cubie A7S PowerVR BXM GPU. The current milestone runs Qwen3 0.6B F16 projection matvecs on Vulkan0 while keeping the fragile attention/output/elementwise paths on CPU. With `LLAMA_VK_NO_OUTPUT_OFFLOAD=1`, `-b 8 -ub 8`, `--no-kv-offload`, and `-fa off`, the mixed CPU/PowerVR path now produces coherent text with several repeating layers offloaded.

See `docs/powervr-llama.md` for the exact command and current limitations.
See `docs/performance.md` for the latest CPU-only vs PowerVR WebUI timing.

## Export Flow

The successful path was:

1. Export a PyTorch/HF EmbeddingGemma text model wrapper with `inputs_embeds` and additive float `attention_bias`.
2. Use an attention-mask dict:

   ```python
   {"full_attention": attention_bias, "sliding_attention": attention_bias}
   ```

   This avoids the ONNX bool/GatherND path that failed during Vivante NBG generation.

3. Rewrite unsupported ONNX ops:

   - replace `Expand` with multiply-by-ones where shape is constant
   - replace `Gelu` with tanh approximation

4. Import and export with Acuity/Vivante tools targeting `VIP9000NANODI_PLUS_PID0X1000003B`.

See:

```text
scripts/export_seq128_bias_hidden.py
scripts/make_seq128_bias_npu_onnx.py
docs/export_commands.md
```

## Assessment

The corrected NPU path is useful when CPU availability matters, for example indexing new knowledge in the background while CPU cores run an LLM. It does not currently beat a 4-thread CPU TFLite run on latency, but it preserves CPU headroom and now produces embeddings close enough to the original model to justify further quality and end-to-end RAG testing.

## License

Repository code is MIT licensed. See `docs/licensing.md` for model/runtime asset notes.
