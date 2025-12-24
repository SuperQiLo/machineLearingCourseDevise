#!/bin/bash
# run_dqn_curriculum.sh
# Usage: ./scripts/run_dqn_curriculum.sh

# 1. Setup directories
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# 2. Config names
LOG_FILE="$LOG_DIR/dqn_curriculum_$(date +'%Y%m%d_%H%M%S').log"
PID_FILE="$LOG_DIR/dqn_curriculum.pid"

echo ">>> Starting DQN Curriculum Training..."
echo ">>> Log file: $LOG_FILE"

# 3. Run in background with nohup
# -u for unbuffered output to see logs in real-time
nohup python3 -u "$PROJECT_ROOT/train_curriculum.py" > "$LOG_FILE" 2>&1 &

# 4. Save PID
NEW_PID=$!
echo $NEW_PID > "$PID_FILE"
echo ">>> Training is running in background. PID: $NEW_PID"
echo ">>> To view logs: tail -f $LOG_FILE"
echo ">>> To stop: kill \$(cat $PID_FILE)"
