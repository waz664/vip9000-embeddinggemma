# Community Outreach

Use this page when sharing the project with Radxa, llama.cpp, local-LLM, SBC, and edge-AI communities.

## Short Description

Experimental local AI stack for Radxa Cubie A7S / Allwinner A733: EmbeddingGemma-300M retrieval on the Vivante VIP9000 NPU, plus Qwen3 0.6B served through llama.cpp with selected PowerVR BXM Vulkan matvec offload. Includes prebuilt runtime packaging, install scripts, systemd services, a RAG chat WebUI, URL/file ingestion, optional web search, Think mode, benchmarks, and reproducibility notes.

## Suggested GitHub Topics

```text
radxa
cubie-a7s
allwinner-a733
vivante
vip9000
npu
powervr
vulkan
llama-cpp
qwen3
embeddinggemma
rag
edge-ai
sbc
local-llm
```

## Show HN Draft

Title:

```text
Show HN: Running embeddings on a VIP9000 NPU and Qwen on PowerVR on a $SBC
```

Body:

```text
I have been experimenting with the Radxa Cubie A7S / Allwinner A733 and put together a public repo for a local RAG stack on the board.

What works:
- EmbeddingGemma-300M retrieval runs its transformer body on the Vivante VIP9000-series NPU.
- Qwen3 0.6B runs through a patched llama.cpp build with selected F16 projection matvecs offloaded to the PowerVR BXM Vulkan GPU.
- A small WebUI supports local KB chat, URL/file ingestion, optional web search, runtime stats, and an opt-in Qwen Think mode.
- The repo includes install scripts, systemd services, benchmarks, docs, and the experimental llama.cpp patches.

It is not production-perfect. The NPU path is a hybrid, the Qwen GPU path is conservative, and CPU still handles plenty of work. But it is a practical DIY starting point for people interested in low-power edge AI on cheap ARM boards.

Repo: https://github.com/waz664/vip9000-embeddinggemma
```

## Reddit / LocalLLaMA Draft

Title:

```text
Radxa Cubie A7S: EmbeddingGemma on VIP9000 NPU + Qwen3 0.6B on PowerVR via llama.cpp
```

Body:

```text
I published the repo for a Radxa Cubie A7S / Allwinner A733 local RAG experiment:

https://github.com/waz664/vip9000-embeddinggemma

Highlights:
- EmbeddingGemma-300M retrieval path using the Vivante VIP9000-series NPU.
- Corrected masked/bias FP32 seq128 NPU graph measured about 0.944 cosine vs the official CPU TFLite reference for a representative query.
- Experimental llama.cpp Vulkan patches for the board's PowerVR BXM GPU.
- Qwen3 0.6B served through llama.cpp with selected projection matvecs on PowerVR and fragile ops kept on CPU.
- WebUI with KB chat, file/URL ingestion, optional web search, citations, runtime stats, response/query caching, and Think mode.
- Install scripts, systemd units, docs, benchmarks, and current limitations are included.

This is a DIY/community starting point, not a claim that the whole LLM runs on GPU or that the NPU path is fully optimized. The interesting part is that the board can do useful background embedding work on the NPU while Qwen runs through the CPU/PowerVR hybrid path.
```

## Radxa Forum Draft

Title:

```text
Cubie A7S AI demo: VIP9000 NPU embeddings + PowerVR llama.cpp Qwen WebUI
```

Body:

```text
I put together a public repo for an end-to-end local RAG demo on the Radxa Cubie A7S:

https://github.com/waz664/vip9000-embeddinggemma

It includes:
- EmbeddingGemma-300M on the A733 VIP9000-series NPU.
- Prebuilt runtime packaging and install scripts.
- A directory-backed vector index demo.
- Patched llama.cpp Vulkan build for selected Qwen3 0.6B PowerVR matvec offload.
- systemd services for the llama.cpp server and WebUI.
- A WebUI with KB chat, URL/file ingestion, optional web search, citations, runtime stats, and Think mode.

Current caveats are documented in the repo. The NPU and GPU paths are both hybrid/conservative, but the stack is useful enough for experimentation and should give other Cubie A7S owners a reproducible starting point.
```

## llama.cpp Discussion Draft

Title:

```text
Experimental PowerVR BXM Vulkan matvec offload on Allwinner A733 / Radxa Cubie A7S
```

Body:

```text
I published an experimental llama.cpp patch set for the PowerVR BXM GPU in the Allwinner A733 / Radxa Cubie A7S:

https://github.com/waz664/vip9000-embeddinggemma/tree/main/patches/llama.cpp

The current stable path is conservative:
- Qwen3 0.6B F16 model.
- Selected projection matvec ops on Vulkan/PowerVR.
- Attention, output projection, KV, ROPE, norms, SWIGLU, and fragile elementwise paths kept on CPU by default.
- `LLAMA_VK_NO_OUTPUT_OFFLOAD=1`, `-ngl 2`, `-b 8`, `-ub 8`, `-c 4096`.

The repo also includes benchmark notes and the local RAG WebUI that uses the server:

https://github.com/waz664/vip9000-embeddinggemma/blob/main/docs/powervr-llama.md

This is not ready as a broad upstream backend, but it may be useful to anyone interested in subgroup-size-1 PowerVR Vulkan behavior, conservative op gating, and small-model edge inference on cheap ARM SBCs.
```

## Good Places To Share

- Radxa forum and Radxa Discord
- llama.cpp GitHub Discussions
- r/LocalLLaMA
- r/SBCs
- r/embedded
- Hacker News Show HN
- Armbian forum
- ServeTheHome forum

## Suggested First Comment

```text
A few caveats up front: this is a hybrid stack. The NPU path runs the embedding transformer body, while CPU still handles tokenization/pooling/projection tail. The PowerVR path is also intentionally conservative and only offloads selected Qwen matvecs. The point of the repo is reproducibility and a useful DIY baseline, not claiming full accelerator coverage.
```
