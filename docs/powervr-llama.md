# PowerVR llama.cpp Status

This repo includes an experimental llama.cpp patch for Radxa Cubie A7S / Allwinner A733 PowerVR BXM Vulkan.

Patches:

```bash
patches/llama.cpp/0001-vulkan-enable-powervr-scalar-f16-matvec-offload.patch
patches/llama.cpp/0002-vulkan-fix-powervr-scalar-f16-matvec-dispatch.patch
patches/llama.cpp/0003-vulkan-submit-powervr-scalar-f16-matvec-dispatches.patch
patches/llama.cpp/0004-vulkan-keep-powervr-qwen-offload-quality-first.patch
patches/llama.cpp/0005-vulkan-add-powervr-op-family-debug-gates.patch
```

Tested local result:

```bash
cd ~/llama.cpp
env GGML_VK_VISIBLE_DEVICES=0 LLAMA_VK_NO_OUTPUT_OFFLOAD=1 \
  ./build-vulkan/bin/llama-completion \
  -m ~/llama.cpp/Qwen3-0.6B-F16-from-Q8.gguf \
  -p 'The capital of France is' \
  -n 12 -c 64 -b 8 -ub 8 -ngl 4 \
  --no-kv-offload -fa off \
  --no-display-prompt --no-conversation --reasoning off \
  --temp 0.2 --top-p 0.9 --no-perf
```

Observed output:

```text
 Paris. The capital of the United States is Washington, D
```

The important signal is that llama.cpp now:

- loads real Qwen F16 repeating-layer weights into the PowerVR Vulkan model buffer
- runs the Qwen projection F16 matvec ops on Vulkan0
- creates the F16 matvec pipeline successfully
- completes coherent generation through the mixed CPU/PowerVR path

Current limits:

- `0004` is quality-first: on PowerVR subgroup-size-1 devices, only Qwen F16 projection matvecs plus metadata views are enabled by default
- the output layer must stay on CPU with `LLAMA_VK_NO_OUTPUT_OFFLOAD=1`
- prompt batches should stay small, for example `-b 8 -ub 8`
- KV cache and flash attention are kept off for this path
- `GGML_VK_POWERVR_ALLOW_RMS_NORM=1`, `GGML_VK_POWERVR_ALLOW_SWIGLU=1`, `GGML_VK_POWERVR_ALLOW_ROPE=1`, and `GGML_VK_POWERVR_ALLOW_ELEMENTWISE=1` can be used to test one auxiliary op family at a time
- `GGML_VK_POWERVR_FULL_OPS=1` can be used for debugging the broader generic Vulkan op set, but that path still corrupts Qwen generation on this driver

Stable server command used for the WebUI:

```bash
env GGML_VK_VISIBLE_DEVICES=0 LLAMA_VK_NO_OUTPUT_OFFLOAD=1 \
  ~/llama.cpp/build-vulkan/bin/llama-server \
  -m ~/llama.cpp/Qwen3-0.6B-F16-from-Q8.gguf \
  --host 0.0.0.0 --port 8081 \
  -c 512 -b 8 -ub 8 -ngl 2 \
  --no-kv-offload -fa off --reasoning off \
  --alias qwen3-0.6b-powervr \
  -np 1 --no-cache-idle-slots --no-warmup
```

`llama-completion` has produced coherent output with `-ngl 4`. The long-running WebUI server is currently kept at `-ngl 2` because that setting survived model load and OpenAI-compatible chat requests repeatedly with a 512-token context.

Observed WebUI request timing for a Radxa NVMe question with NPU embedding retrieval plus the llama.cpp provider:

```text
answer="Yes, the Cubie A7S supports NVMe storage via PCIe 3.0 x1 expansion. [2]"
embedding_s=18.99
llm_s=63.03
total_s=82.03
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

Batch-8 projection shapes also pass under `test-backend-ops`.

After `0004`, the exported Qwen op scan accepts only the known-good projection matvecs on Vulkan and leaves attention matvecs, softmax, output projection, embedding lookup, KV writes, norms, ROPE, and elementwise ops on CPU. This is slower than the broader offload attempt, but it restores coherent generation:

```text
Accepted on Vulkan0: 10/10 Qwen F16 projection matvec tests passed
Rejected to CPU: attention matvecs, SOFT_MAX, output projection, GET_ROWS, SET_ROWS, ROPE, RMS_NORM, SWIGLU, elementwise ops
```

The next work item is to re-enable one auxiliary op family at a time with full generation tests after each addition. Do not treat isolated `test-backend-ops` success as sufficient on this PowerVR driver; the earlier broader path passed isolated tests and still corrupted generation.

After `0005`, auxiliary op families can be tested individually without enabling all generic Vulkan ops. Current results:

| Family | `test-backend-ops` | Generation Quality | WebUI Timing Impact | Default |
| --- | --- | --- | --- | --- |
| `RMS_NORM` | pass | coherent | not faster in full WebUI test when combined with SWIGLU | off |
| `SWIGLU` | pass | coherent | not faster in full WebUI test when combined with RMS_NORM | off |
| `ROPE` | fails Q-cur shapes with `ERR = inf` | not tested further | unsafe | off |
| same-shape `ADD/SUB/MUL/DIV` | pass for exported same-shape ops | corrupt generation | unsafe | off |
| `RMS_NORM + SWIGLU` | pass | coherent | slower than default: `55.07 s` cold, `12.47 s` warm vs default service result `33.20 s` cold, `9.17 s` warm | off |

The current conclusion is that the quality-first default should remain projection-matvec only. RMS_NORM and SWIGLU are useful as debug gates and may become useful if graph split/synchronization overhead is reduced, but they do not improve the present WebUI workload.
