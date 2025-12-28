#!/bin/bash
# scripts/stop_training.sh
# Safely stop Snake AI training processes with optional filtering.
# Usage: ./stop_training.sh [dqn|ppo|curriculum|all]

TARGET=${1:-"all"}
echo ">>> Targeted stop: $TARGET"

case $TARGET in
    dqn)
        PATTERNS=("train_dqn_variants.py" "train_dqn_curriculum.py")
        ;;
    ppo)
        PATTERNS=("train_ppo.py" "train_ppo_curriculum.py")
        ;;
    curriculum)
        PATTERNS=("train_dqn_curriculum.py" "train_ppo_curriculum.py")
        ;;
    all)
        PATTERNS=("train_dqn_variants.py" "train_dqn_curriculum.py" "train_ppo.py" "train_ppo_curriculum.py")
        ;;
    *)
        echo "Usage: $0 [dqn|ppo|curriculum|all]"
        exit 1
        ;;
esac

for pattern in "${PATTERNS[@]}"; do
    # -f matches against full argument lists
    PIDS=$(pgrep -f "$pattern")
    if [ -n "$PIDS" ]; then
        echo ">>> Killing $TARGET processes: $pattern (PIDs: $PIDS)"
        pkill -9 -f "$pattern"
    else
        echo ">>> No processes found for pattern: $pattern"
    fi
done

# Clean up PID files if "all" or specific to algorithms
if [ "$TARGET" == "all" ]; then
    find logs -name "*.pid" -type f -delete 2>/dev/null
    echo ">>> All PID files cleaned."
fi

echo ">>> Stop operation for [$TARGET] finished."
