# EmbeddingGemma on Vivante VIP9000 NPU

This repository documents and packages a working EmbeddingGemma-300M embedding path for the Radxa Cubie A7S / Allwinner A733 Vivante VIP9000-series NPU.

The working deployment uses:

- FP32 `seq128` transformer graph compiled to VIPLite NBG
- two NPU inputs: token embeddings and additive attention bias
- NPU transformer body
- CPU tokenization, embedding lookup, masked pooling, dense projection tail, and L2 normalization
- a small directory-backed RAG demo for Radxa Cubie A7S product/docs pages

The important result is that adding the attention-bias input restored most of the original model quality. The previous unmasked NPU graph was only about `0.747` cosine against official CPU TFLite output for a representative query. The corrected masked/bias graph measured `0.944`.

## Current Results

Single-query benchmark on Radxa Cubie A7S:

| Path | Wall Time | CPU Time | Cosine vs Official TFLite |
| --- | ---: | ---: | ---: |
| CPU TFLite, 1 thread | 28.142 s | 25.975 s | 1.000 |
| CPU TFLite, 4 threads | 13.354 s | 40.977 s | 1.000 |
| old NPU, unmasked hidden FP32 seq128 | 23.665 s | 4.236 s | 0.747 |
| corrected NPU, masked/bias hidden FP32 seq128 | 19.331 s | 3.169 s | 0.944 |

Small retrieval A/B eval, official CPU TFLite reference vs corrected NPU:

| Metric | Value |
| --- | ---: |
| document cosine mean | 0.9285 |
| document cosine min | 0.9029 |
| query cosine mean | 0.9434 |
| query cosine min | 0.9346 |
| top-1 match rate | 0.6250 |
| overlap@5 mean | 0.9250 |
| reference top-1 MRR in NPU ranking | 0.7500 |

The retrieval result is promising: top-5 overlap is high, and several top-1 disagreements appear to be between generic overview chunks and more specific product-page chunks rather than obvious failures.

## What Is Not Committed

The following files are intentionally not included:

- `network_binary.nb` - generated VIPLite NBG, about 404 MB
- `token_embedding_fp16.dat` - EmbeddingGemma token embedding table, about 384 MB
- `dense_2_weight_f32.npy` and `dense_3_weight_f32.npy` - dense tail weights, about 9 MB each
- Google model weights and TFLite model files
- Vivante/Radxa SDK files and Docker images

Those files are large and/or governed by upstream licenses. This repo contains the code, export scripts, metadata, and reproducibility notes.

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
cd /home/radxa/embeddinggemma_npu_seq128_bias_hidden_fp32
./embed_text_bias_hidden_npu.py "does Cubie A7S support NVMe storage?"
```

The script uses:

```bash
LD_LIBRARY_PATH=/home/radxa/ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v2.0
/home/radxa/ai-sdk/examples/vpm_run/vpm_run
```

## Run The RAG Demo

```bash
cd /home/radxa/embeddinggemma_npu_seq128_bias_hidden_fp32/rag_demo
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
cd /home/radxa/embeddinggemma_npu_seq128_bias_hidden_fp32/rag_demo
./evaluate_retrieval_quality.py --max-chunks 8 --top-k 5 --threads 4
```

The latest summary from this board is included at:

```text
artifacts/quality_eval_summary.json
```

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
