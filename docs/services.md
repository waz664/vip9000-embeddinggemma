# Services

The repo includes systemd helpers for running the full stack at boot:

- `vip9000-llama.service` starts the patched llama.cpp Qwen3 PowerVR server on port `8081`
- `vip9000-rag-webui.service` starts the RAG WebUI on port `8080`

## User Services

This is the recommended install mode for hobbyist boards because it does not require root for the app processes:

```bash
cd ~/vip9000-embeddinggemma
bash install/install_systemd_services.sh
```

Check status:

```bash
systemctl --user status vip9000-llama.service
systemctl --user status vip9000-rag-webui.service
```

Follow logs:

```bash
journalctl --user -u vip9000-llama.service -f
journalctl --user -u vip9000-rag-webui.service -f
```

Restart after editing config:

```bash
systemctl --user restart vip9000-llama.service vip9000-rag-webui.service
```

After pulling repo updates, refresh the installed runtime copy used by the service:

```bash
bash install/update_installed_runtime_code.sh
systemctl --user restart vip9000-rag-webui.service
```

If you want user services to start before login, enable linger once:

```bash
sudo loginctl enable-linger "$USER"
```

## System Services

If you prefer system-wide units:

```bash
SCOPE=system bash install/install_systemd_services.sh
```

By default the service user is the current user. Override with:

```bash
SCOPE=system SERVICE_USER=radxa bash install/install_systemd_services.sh
```

Check status:

```bash
sudo systemctl status vip9000-llama.service
sudo systemctl status vip9000-rag-webui.service
```

## Configuration

Important environment overrides:

```text
INSTALL_DIR=~/embeddinggemma_npu_seq128_bias_hidden_fp32
LLAMA_CPP_DIR=~/llama.cpp
MODEL=~/llama.cpp/Qwen3-0.6B-F16-from-Q8.gguf
WEBUI_PORT=8080
LLAMA_PORT=8081
CTX_SIZE=512
GPU_LAYERS=2
VIP9000_RAG_TOP_K=1
VIP9000_RAG_CONTEXT_CHARS=450
CPUSET=
THREADS=
THREADS_BATCH=
GGML_VK_POWERVR_ALLOW_RMS_NORM=
GGML_VK_POWERVR_ALLOW_SWIGLU=
```

The defaults match the tested stable setup.

## One-Command Full Stack Install

For a fresh board with the SDK and runtime asset available:

```bash
bash install/full_stack_install.sh
```

Useful flags:

```bash
SKIP_RUNTIME=1 bash install/full_stack_install.sh
SKIP_LLAMA_BUILD=1 bash install/full_stack_install.sh
INSTALL_SERVICES=0 bash install/full_stack_install.sh
RUN_BENCHMARK=0 bash install/full_stack_install.sh
```

## Verified Service-Managed Benchmark

After installing user services on the development Cubie A7S:

```bash
python3 scripts/benchmark_webui.py --runs 2
```

Measured:

```text
run=1 wall=33.20s embedding=19.0452s llm=14.13s total=33.18s embedding_cache_hit=False
run=2 wall=9.17s  embedding=0.0007s  llm=9.16s  total=9.17s  embedding_cache_hit=True
```

Both answers were correct and cited the Radxa docs. The second run demonstrates the WebUI exact-query embedding cache under systemd-managed services.
