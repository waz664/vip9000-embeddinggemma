#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/embeddinggemma_npu_seq128_bias_hidden_fp32}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
MODEL="${MODEL:-$LLAMA_CPP_DIR/Qwen3-0.6B-F16-from-Q8.gguf}"
SCOPE="${SCOPE:-user}"
WEBUI_PORT="${WEBUI_PORT:-8080}"
LLAMA_PORT="${LLAMA_PORT:-8081}"
CTX_SIZE="${CTX_SIZE:-1024}"
GPU_LAYERS="${GPU_LAYERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
UBATCH_SIZE="${UBATCH_SIZE:-8}"
TOP_K="${VIP9000_RAG_TOP_K:-3}"
CONTEXT_CHARS="${VIP9000_RAG_CONTEXT_CHARS:-1000}"
MAX_TOKENS="${VIP9000_RAG_MAX_TOKENS:-256}"

write_user_units() {
  local unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
  mkdir -p "$unit_dir"

  cat > "$unit_dir/vip9000-llama.service" <<EOF
[Unit]
Description=Qwen3 0.6B llama.cpp PowerVR server
After=default.target

[Service]
Type=simple
WorkingDirectory=$LLAMA_CPP_DIR
Environment=LLAMA_CPP_DIR=$LLAMA_CPP_DIR
Environment=MODEL=$MODEL
Environment=HOST=0.0.0.0
Environment=PORT=$LLAMA_PORT
Environment=CTX_SIZE=$CTX_SIZE
Environment=GPU_LAYERS=$GPU_LAYERS
Environment=BATCH_SIZE=$BATCH_SIZE
Environment=UBATCH_SIZE=$UBATCH_SIZE
Environment=GGML_VK_VISIBLE_DEVICES=0
Environment=LLAMA_VK_NO_OUTPUT_OFFLOAD=1
ExecStart=$REPO_DIR/install/run_llama_cpp_powervr_server.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

  cat > "$unit_dir/vip9000-rag-webui.service" <<EOF
[Unit]
Description=VIP9000 EmbeddingGemma RAG WebUI
After=vip9000-llama.service
Wants=vip9000-llama.service

[Service]
Type=simple
WorkingDirectory=$REPO_DIR
Environment=INSTALL_DIR=$INSTALL_DIR
Environment=PORT=$WEBUI_PORT
Environment=VIP9000_RAG_LLM_PROVIDER=llama_cpp
Environment=VIP9000_RAG_LLAMA_CPP_URL=http://127.0.0.1:$LLAMA_PORT/v1/chat/completions
Environment=VIP9000_RAG_TOP_K=$TOP_K
Environment=VIP9000_RAG_CONTEXT_CHARS=$CONTEXT_CHARS
Environment=VIP9000_RAG_MAX_TOKENS=$MAX_TOKENS
ExecStart=$REPO_DIR/install/run_webui.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

  systemctl --user daemon-reload
  systemctl --user enable vip9000-llama.service vip9000-rag-webui.service
  systemctl --user restart vip9000-llama.service vip9000-rag-webui.service
  echo "installed user services:"
  echo "  systemctl --user status vip9000-llama.service"
  echo "  systemctl --user status vip9000-rag-webui.service"
}

write_system_units() {
  local unit_dir="/etc/systemd/system"
  local user_name="${SERVICE_USER:-$(id -un)}"

  sudo tee "$unit_dir/vip9000-llama.service" >/dev/null <<EOF
[Unit]
Description=Qwen3 0.6B llama.cpp PowerVR server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$user_name
WorkingDirectory=$LLAMA_CPP_DIR
Environment=HOME=$HOME
Environment=LLAMA_CPP_DIR=$LLAMA_CPP_DIR
Environment=MODEL=$MODEL
Environment=HOST=0.0.0.0
Environment=PORT=$LLAMA_PORT
Environment=CTX_SIZE=$CTX_SIZE
Environment=GPU_LAYERS=$GPU_LAYERS
Environment=BATCH_SIZE=$BATCH_SIZE
Environment=UBATCH_SIZE=$UBATCH_SIZE
Environment=GGML_VK_VISIBLE_DEVICES=0
Environment=LLAMA_VK_NO_OUTPUT_OFFLOAD=1
ExecStart=$REPO_DIR/install/run_llama_cpp_powervr_server.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

  sudo tee "$unit_dir/vip9000-rag-webui.service" >/dev/null <<EOF
[Unit]
Description=VIP9000 EmbeddingGemma RAG WebUI
After=vip9000-llama.service
Wants=vip9000-llama.service

[Service]
Type=simple
User=$user_name
WorkingDirectory=$REPO_DIR
Environment=HOME=$HOME
Environment=INSTALL_DIR=$INSTALL_DIR
Environment=PORT=$WEBUI_PORT
Environment=VIP9000_RAG_LLM_PROVIDER=llama_cpp
Environment=VIP9000_RAG_LLAMA_CPP_URL=http://127.0.0.1:$LLAMA_PORT/v1/chat/completions
Environment=VIP9000_RAG_TOP_K=$TOP_K
Environment=VIP9000_RAG_CONTEXT_CHARS=$CONTEXT_CHARS
Environment=VIP9000_RAG_MAX_TOKENS=$MAX_TOKENS
ExecStart=$REPO_DIR/install/run_webui.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

  sudo systemctl daemon-reload
  sudo systemctl enable vip9000-llama.service vip9000-rag-webui.service
  sudo systemctl restart vip9000-llama.service vip9000-rag-webui.service
  echo "installed system services:"
  echo "  sudo systemctl status vip9000-llama.service"
  echo "  sudo systemctl status vip9000-rag-webui.service"
}

case "$SCOPE" in
  user)
    write_user_units
    ;;
  system)
    write_system_units
    ;;
  *)
    echo "SCOPE must be 'user' or 'system'" >&2
    exit 1
    ;;
esac
