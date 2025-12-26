#!/bin/bash
# run_ppo_curriculum.sh
# Usage: ./scripts/run_ppo_curriculum.sh

# 1. Setup directories
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# 2. Config names
LOG_FILE="$LOG_DIR/ppo_curriculum_$(date +'%Y%m%d_%H%M%S').log"
PID_FILE="$LOG_DIR/ppo_curriculum.pid"

echo ">>> Starting PPO V7.0 Champion Curriculum Training..."
echo ">>> Featuring: Survival Balancing, Multi-Stage LR precision (Final 20% frames)"
echo ">>> Log file: $LOG_FILE"

# 3. Run in background with nohup
PY_CMD="python"
if ! command -v $PY_CMD &> /dev/null; then PY_CMD="python3"; fi

nohup $PY_CMD -u "$PROJECT_ROOT/train_ppo_curriculum.py" > "$LOG_FILE" 2>&1 &

# 4. Save PID
NEW_PID=$!
echo $NEW_PID > "$PID_FILE"
echo ">>> Training is running in background. PID: $NEW_PID"
echo ">>> To view logs: tail -f $LOG_FILE"
echo ">>> To stop ALL processes: bash scripts/stop_training.sh"
