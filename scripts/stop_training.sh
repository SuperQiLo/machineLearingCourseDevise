#!/bin/bash
# scripts/stop_training.sh
# Safely stop Snake AI training processes with precision filtering.
# Usage: ./stop_training.sh [dqn|ddqn|per|dueling|ppo|curriculum|all]

TARGET=${1:-"all"}
echo ">>> Targeted stop: $TARGET"

case $TARGET in
    dqn)
        # Kill any DQN variant training or curriculum
        PATTERNS=("train_dqn_variants.py" "train_dqn_curriculum.py")
        ;;
    ddqn|per|dueling)
        # Kill specific variant training
        # pkill -f matches full command line including "--variant variant_name"
        PATTERNS=("train_dqn_variants.py .*--variant $TARGET")
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
        echo "Usage: $0 [dqn|ddqn|per|dueling|ppo|curriculum|all]"
        exit 1
        ;;
esac

for pattern in "${PATTERNS[@]}"; do
    # Use pgrep -f to show what we are killing
    PIDS=$(pgrep -f "$pattern")
    if [ -n "$PIDS" ]; then
        echo ">>> Killing matching processes: $pattern (PIDs: $PIDS)"
        pkill -9 -f "$pattern"
    else
        echo ">>> No processes found for: $pattern"
    fi
done

# Clean up PID files only for global stop
if [ "$TARGET" == "all" ]; then
    find logs -name "*.pid" -type f -delete 2>/dev/null
    echo ">>> All PID files cleaned."
fi

echo ">>> Stop operation for [$TARGET] finished."
