# Prebuilt Runtime Bundle

The prebuilt runtime bundle contains the files needed to run the corrected FP32 `seq128` EmbeddingGemma NPU model without converting ONNX or running the Vivante compiler.

Bundle name:

```text
vip9000-embeddinggemma-a7s-seq128-fp32-v0.1.0.tar.zst
```

Contents:

```text
runtime/
  MANIFEST.json
  SHA256SUMS
  network_binary.nb
  nbg_meta.json
  token_embedding_fp16.dat
  dense_2_weight_f32.npy
  dense_3_weight_f32.npy
```

Expected size is roughly 800 MB.

## Install From Release Asset

After the bundle is uploaded to a GitHub release:

```bash
git clone https://github.com/waz664/vip9000-embeddinggemma.git
cd vip9000-embeddinggemma
bash install/setup_radxa_dependencies.sh
bash install/install_runtime.sh
bash install/smoke_test.sh
```

By default, `install_runtime.sh` downloads from:

```text
https://github.com/waz664/vip9000-embeddinggemma/releases/download/v0.1.0/vip9000-embeddinggemma-a7s-seq128-fp32-v0.1.0.tar.zst
```

To install from a local asset file:

```bash
LOCAL_ASSET=/path/to/vip9000-embeddinggemma-a7s-seq128-fp32-v0.1.0.tar.zst \
  bash install/install_runtime.sh
```

## Create The Bundle

On a board that already has the working runtime directory:

```bash
cd /home/radxa/vip9000-embeddinggemma
bash scripts/pack_model_assets.sh
```

This writes:

```text
dist/vip9000-embeddinggemma-a7s-seq128-fp32-v0.1.0.tar.zst
dist/vip9000-embeddinggemma-a7s-seq128-fp32-v0.1.0.tar.zst.sha256
```

## Upload To GitHub Release

Normal git cannot store this bundle because the binary files exceed GitHub's regular file-size limit. Upload it as a release asset:

```bash
cd /home/radxa/vip9000-embeddinggemma
gh release create v0.1.0 \
  dist/vip9000-embeddinggemma-a7s-seq128-fp32-v0.1.0.tar.zst \
  dist/vip9000-embeddinggemma-a7s-seq128-fp32-v0.1.0.tar.zst.sha256 \
  --repo waz664/vip9000-embeddinggemma \
  --title "VIP9000 EmbeddingGemma A7S seq128 FP32 runtime" \
  --notes "Prebuilt VIPLite NBG and CPU-side embedding tables for Radxa Cubie A7S / Allwinner A733."
```

If using GitHub CLI on a machine with browser auth:

```bash
gh auth login
```

Then rerun the release command.

## Run RAG WebUI

```bash
cd ~/embeddinggemma_npu_seq128_bias_hidden_fp32
python3 ./rag_demo/build_index.py --max-chunks 8
bash ~/vip9000-embeddinggemma/install/run_webui.sh
```

Open:

```text
http://<radxa-ip>:8080
```
