# Release Notes

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
