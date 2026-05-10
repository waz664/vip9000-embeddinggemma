# PowerVR llama.cpp Status

This repo includes an experimental llama.cpp patch for Radxa Cubie A7S / Allwinner A733 PowerVR BXM Vulkan.

Patches:

```bash
patches/llama.cpp/0001-vulkan-enable-powervr-scalar-f16-matvec-offload.patch
patches/llama.cpp/0002-vulkan-fix-powervr-scalar-f16-matvec-dispatch.patch
patches/llama.cpp/0003-vulkan-submit-powervr-scalar-f16-matvec-dispatches.patch
```

Tested local result:

```bash
cd /home/radxa/llama.cpp
env GGML_VK_VISIBLE_DEVICES=0 ./build-vulkan/bin/llama-completion \
  -m /home/radxa/llama.cpp/Qwen3-0.6B-F16-from-Q8.gguf \
  -p 'Say OK.' \
  -n 4 -c 64 -b 8 -ub 8 -ngl 2 \
  --no-kv-offload -fa off \
  --no-display-prompt --no-conversation --reasoning off \
  --temp 0.2 --top-p 0.9 --no-perf
```

The important signal is that llama.cpp now:

- loads about 30 MiB of real Qwen F16 layer weights into the PowerVR Vulkan model buffer
- creates the F16 matvec pipeline successfully
- completes generation through the Vulkan path

Current limits:

- only small token-batch F16 matvec is enabled on PowerVR
- prompt batches should stay small, for example `-b 8 -ub 8`
- KV cache and flash attention are kept off for this path
- generated text should be retested after `0003`; prior runs before this correctness fix were corrupted

Live server command used for the WebUI:

```bash
env GGML_VK_VISIBLE_DEVICES=0 ~/llama.cpp/build-vulkan/bin/llama-server \
  -m ~/llama.cpp/Qwen3-0.6B-F16-from-Q8.gguf \
  --host 0.0.0.0 --port 8081 \
  -c 512 -b 8 -ub 8 -ngl 2 \
  --no-kv-offload -fa off --reasoning off \
  --alias qwen3-0.6b-powervr \
  -np 1 --no-cache-idle-slots
```

Observed WebUI request timing for a Radxa NVMe question:

```text
embedding_s=19.0
llm_s=67.0
total_s=86.1
provider=llama_cpp
model=qwen3-0.6b-powervr
```

The patch works by adding a scalar F16-weight/F32-vector matvec shader matching the standalone Vulkan repro and avoiding the subgroup/RTE SPIR-V mutations that the PowerVR driver rejects for this pipeline.

## Correctness Worklog

The first quality fix targets the isolated Qwen token matvec exported from llama.cpp:

```text
MUL_MAT(name=Vcur-0,type=f32,ne=[1024,1,1,1],sources=f16[1024,1024,1,1],f32[1024,1,1,1])
```

Before `0002`, ggml launched the scalar F16 shader with the old subgroup workgroup denominator. On PowerVR BXM the shader uses one invocation per output row, so most rows were not written and `test-backend-ops` reported `ERR = inf`.

After `0002`, the same op produced finite, close values, but still failed the strict backend tolerance:

```text
[MUL_MAT] ERR = 0.952921962 > 0.000500000
sample diffs: 0.026741, -0.020079, -0.006459, 0.003721
```

The standalone direct Vulkan repro continues to validate the same F16 matvec shape with low absolute error, so the next work item is to close the remaining ggml integration gap rather than changing the model or prompt.

After `0003`, the PowerVR scalar F16 path uses separate descriptor writes and submits/restarts the compute command buffer around each scalar F16 dispatch. This is deliberately conservative, but it removes the moving unwritten-row failures seen in ggml's longer command stream.

Validated on the Cubie A7S with `test-backend-ops`:

```bash
taskset -c 6,7 env GGML_VK_VISIBLE_DEVICES=0 \
  ./build-vulkan/bin/test-backend-ops test \
  -b Vulkan0 \
  --test-file /tmp/qwen_f16_ops/mulmat_1024x1024_tok1.txt
```

Result:

```text
MUL_MAT Vcur-0 f16[1024,1024] x f32[1024,1]: OK
```

Additional Qwen token projection shapes also pass:

```text
node_32    f16[2048,1024] x f32[2048,1]: OK
ffn_out-0  f16[3072,1024] x f32[3072,1]: OK
Qcur-0     f16[1024,2048] x f32[1024,1]: OK
ffn_gate-0 f16[1024,3072] x f32[1024,1]: OK
```

Next steps are to retest full llama generation quality, then expand coverage to batch-8 prompt matvecs and the output projection.
