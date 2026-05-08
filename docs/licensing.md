# Licensing Notes

Repository code is MIT licensed.

The prebuilt runtime release asset contains generated/runtime files derived from upstream model and vendor tooling:

- EmbeddingGemma tokenizer and model-derived weights are subject to the upstream Google model license and terms.
- `network_binary.nb` is a generated VIPLite network binary produced with Vivante/Radxa tooling.
- The Radxa/Allwinner/Vivante SDK and runtime libraries are not redistributed in this repository.

This project is intended to make the technical integration reproducible for compatible hardware. Users are responsible for complying with upstream model, SDK, and board-vendor licenses in their own deployment context.
