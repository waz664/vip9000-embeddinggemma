# Troubleshooting

## `/dev/vipcore` is missing

The VIPLite runtime needs the NPU kernel driver. Check:

```bash
ls -l /dev/vipcore
dmesg | grep -i vip
```

If the device node is missing, install or boot the Radxa/Allwinner image that includes the A733 NPU driver.

## `vpm_run` is missing

The examples use Radxa's VIPLite sample runner. By default scripts look under:

```text
~/ai-sdk/examples/vpm_run/vpm_run
~/ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v2.0
```

Override paths when your SDK is elsewhere:

```bash
export VIP9000_VPM_RUN=/path/to/vpm_run
export VIP9000_VIPLIB=/path/to/viplite-tina/lib/aarch64-none-linux-gnu/v2.0
```

## `libvip_lite.so` cannot be found

Set:

```bash
export VIP9000_VIPLIB=/path/to/viplite-tina/lib/aarch64-none-linux-gnu/v2.0
```

The Python runner prepends this path to `LD_LIBRARY_PATH`.

## `tokenizer.model` is missing

Runtime releases from `v0.1.1` onward include `tokenizer.model`. If installing manually, place it next to:

```text
embed_text_bias_hidden_npu.py
network_binary.nb
token_embedding_fp16.dat
```

or set:

```bash
export EMBEDDINGGEMMA_TOKENIZER=/path/to/tokenizer.model
```

## Checksum failure during install

Remove the partial download and rerun:

```bash
rm -rf /tmp/vip9000-embeddinggemma-install
bash install/install_runtime.sh
```

## WebUI starts but answers are slow

The demo uses the NPU for retrieval and Ollama for Qwen3 generation. On Cubie A7S, Qwen3 0.6B generation can take more than a minute. For quicker UI testing:

```bash
ollama pull gemma3:270m
```

Then edit `webui/app.py` or set a local patch to use `gemma3:270m`.

## Good smoke test

```bash
cd ~/embeddinggemma_npu_seq128_bias_hidden_fp32
python3 ./embed_text_bias_hidden_npu.py "does Cubie A7S support NVMe storage?"
```

Expected shape of success:

```text
embedding_norm=1
embedding_first16=...
```
