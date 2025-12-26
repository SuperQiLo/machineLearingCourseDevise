#!/bin/bash
# run_dqn_curriculum.sh
# Usage: ./scripts/run_dqn_curriculum.sh [dqn|ddqn|per|dueling]

VARIANT=${1:-"dueling"} # Improved: Default to Dueling-DQN in V7.0

# 1. Setup directories
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# 2. Config names
LOG_FILE="$LOG_DIR/curriculum_${VARIANT}_$(date +'%Y%m%d_%H%M%S').log"
PID_FILE="$LOG_DIR/dqn_curriculum_${VARIANT}.pid"

echo ">>> Starting DQN V7.0 Champion Curriculum Training [Variant: ${VARIANT}]..."
echo ">>> Featuring: Omni-Batch Inference, Chaos Sampling, Multi-Stage LR Decay"
echo ">>> Log file: $LOG_FILE"

# 3. Run in background with nohup
# Note: Using 'python' or 'python3' based on environment
PY_CMD="python"
if ! command -v $PY_CMD &> /dev/null; then PY_CMD="python3"; fi

nohup $PY_CMD -u "$PROJECT_ROOT/train_dqn_curriculum.py" --variant "${VARIANT}" > "$LOG_FILE" 2>&1 &

# 4. Save PID
NEW_PID=$!
echo $NEW_PID > "$PID_FILE"
echo ">>> Training is running in background. PID: $NEW_PID"
echo ">>> To view logs: tail -f $LOG_FILE"
echo ">>> To stop ALL processes: bash scripts/stop_training.sh"
