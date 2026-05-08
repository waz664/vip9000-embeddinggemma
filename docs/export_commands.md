# Export Commands

These are the key commands used on the x86 Linux NUC with the Radxa/Vivante Docker image and SDK mounted.

Paths used during the experiment:

```text
NUC workspace: /home/brianw/docker_images_v2.0.x/docker_data
Docker image:   ubuntu-npu:v2.0.10.1
Acuity path:    /root/acuity-toolkit-whl-6.30.22/bin
Vivante IDE:    /root/Vivante_IDE/VivanteIDE5.11.0/cmdtools
```

## 1. Export PyTorch To ONNX

The wrapper exports hidden states from the transformer body:

```bash
cd /home/brianw/docker_images_v2.0.x/docker_data
python3 export_seq128_bias_hidden.py
```

Output:

```text
embeddinggemma_seq128_bias_hidden.onnx
```

## 2. Rewrite ONNX For Vivante Import

```bash
cd /home/brianw/docker_images_v2.0.x/docker_data
python3 make_seq128_bias_npu_onnx.py
```

Output:

```text
embeddinggemma_seq128_bias_hidden_npu.onnx
```

The transformed graph avoids the unsupported bool/GatherND mask path and replaces `Gelu`.

## 3. Import ONNX Into Acuity

```bash
cd /home/brianw/docker_images_v2.0.x/docker_data
docker run --rm \
  -v /home/brianw/docker_images_v2.0.x/docker_data:/workspace \
  -v /home/brianw/ai-sdk:/ai-sdk \
  ubuntu-npu:v2.0.10.1 \
  bash -lc '
    set -e
    export ACUITY_PATH=/root/acuity-toolkit-whl-6.30.22/bin
    cd /workspace
    rm -rf eg128_bias_hidden_npu
    mkdir eg128_bias_hidden_npu
    cp embeddinggemma_seq128_bias_hidden_npu.onnx eg128_bias_hidden_npu/embeddinggemma.onnx
    cd eg128_bias_hidden_npu
    python3 $ACUITY_PATH/pegasus.py import onnx \
      --model embeddinggemma.onnx \
      --inputs "inputs_embeds attention_bias" \
      --input-size-list "1,128,768#1,1,1,128" \
      --size-with-batch "true#true" \
      --outputs type_as_144 \
      --input-dtype-list "float#float" \
      --output-model embeddinggemma.json \
      --output-data embeddinggemma.data
  '
```

Observed imported input layer names:

```text
inputs_embeds_2904
attention_bias_2905
```

## 4. Input Meta

Create `embeddinggemma_inputmeta.yml`:

```yaml
inputs:
  inputs_embeds_2904:
    dtype: float32
    shape: [1, 128, 768]
    layout: nchw
  attention_bias_2905:
    dtype: float32
    shape: [1, 1, 1, 128]
    layout: nchw
```

## 5. Export FP32 VIPLite NBG

```bash
cd /home/brianw/docker_images_v2.0.x/docker_data
docker run --rm \
  -v /home/brianw/docker_images_v2.0.x/docker_data:/workspace \
  -v /home/brianw/ai-sdk:/ai-sdk \
  ubuntu-npu:v2.0.10.1 \
  bash -lc '
    set -e
    export ACUITY_PATH=/root/acuity-toolkit-whl-6.30.22/bin
    cd /workspace/eg128_bias_hidden_npu
    rm -rf wksp_export_float32 wksp_export_float32_nbg_unify
    mkdir -p wksp_export_float32
    python3 $ACUITY_PATH/pegasus.py export ovxlib \
      --model embeddinggemma.json \
      --model-data embeddinggemma.data \
      --dtype float32 \
      --target-ide-project linux64 \
      --with-input-meta embeddinggemma_inputmeta.yml \
      --pack-nbg-unify \
      --optimize VIP9000NANODI_PLUS_PID0X1000003B \
      --viv-sdk /root/Vivante_IDE/VivanteIDE5.11.0/cmdtools \
      --output-path wksp_export_float32/embeddinggemma_seq128_bias_hidden_fp32
  '
```

Output:

```text
eg128_bias_hidden_npu/wksp_export_float32_nbg_unify/network_binary.nb
eg128_bias_hidden_npu/wksp_export_float32_nbg_unify/nbg_meta.json
```

The export/codegen step was largely single-threaded on the NUC. RAM was not the bottleneck.
