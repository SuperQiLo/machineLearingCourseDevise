#!/bin/bash
# scripts/stop_training.sh
# Safely stop all Snake AI training processes (main and sub-processes).

echo ">>> Searching for Snake AI training processes..."

# List of patterns to kill
# - curriculum scripts
# - variant/base scripts
PATTERNS=("train_dqn_curriculum.py" "train_dqn_variants.py" "train_ppo_curriculum.py" "train_ppo.py")

for pattern in "${PATTERNS[@]}"; do
    PIDS=$(pgrep -f "$pattern")
    if [ -n "$PIDS" ]; then
        echo ">>> Killing processes matching: $pattern ($PIDS)"
        pkill -9 -f "$pattern"
    fi
done

# Clean up PID files
find logs -name "*.pid" -type f -delete 2>/dev/null

echo ">>> All training processes stopped."
