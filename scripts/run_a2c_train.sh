#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs

# Headless/服务器环境：避免 SDL/ALSA 噪声与 XDG 报错
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-runtime-$(id -u)}"
mkdir -p "$XDG_RUNTIME_DIR" || true
chmod 700 "$XDG_RUNTIME_DIR" 2>/dev/null || true
export SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-dummy}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
export PYGAME_HIDE_SUPPORT_PROMPT=1

# PyTorch 显存分配器：缓解碎片化（不是 OOM 根因，但更稳）
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/a2c_train_${TS}.log"
PID_FILE="logs/a2c_train_${TS}.pid"

nohup python -u a2c_train.py >"$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" >"$PID_FILE"

echo "started a2c_train.py pid=$PID"
echo "log: $LOG_FILE"
echo "pid: $PID_FILE"
