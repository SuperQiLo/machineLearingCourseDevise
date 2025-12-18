#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/a2c_curriculum_train_${TS}.log"
PID_FILE="logs/a2c_curriculum_train_${TS}.pid"

nohup python -u a2c_curriculum_train.py >"$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" >"$PID_FILE"

echo "started a2c_curriculum_train.py pid=$PID"
echo "log: $LOG_FILE"
echo "pid: $PID_FILE"
