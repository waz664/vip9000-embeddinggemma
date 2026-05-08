# Local Cleanup Notes

The working runtime directory on the Radxa board is:

```text
/home/radxa/embeddinggemma_npu_seq128_bias_hidden_fp32
```

Useful files inside it:

```text
network_binary.nb
nbg_meta.json
token_embedding_fp16.dat
dense_2_weight_f32.npy
dense_3_weight_f32.npy
embed_text_bias_hidden_npu.py
benchmark_bias_vs_cpu.py
rag_demo/
```

Older experimental directories can be removed after this repo is created and the corrected runtime is preserved:

```text
/home/radxa/embeddinggemma_npu_seq128
/home/radxa/embeddinggemma_npu_seq128_prenorm
/home/radxa/embeddinggemma_npu_seq128_prenorm_fp32
/home/radxa/embeddinggemma_npu_seq128_hidden_fp32
```

Do not remove:

```text
/home/radxa/llama.cpp
/home/radxa/ai-sdk
/home/radxa/embeddinggemma
/home/radxa/embeddinggemma_npu_seq128_bias_hidden_fp32
```

`ai-sdk` is required for VIPLite runtime and `vpm_run`.
`embeddinggemma` contains the tokenizer and official TFLite model used for benchmarking.
