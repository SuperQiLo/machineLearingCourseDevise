"""
PPO Curriculum Training Script.
Automates Phase 1 (Single Snake) -> Phase 2 (Battle Snake).
"""

import subprocess
import time
import sys
from pathlib import Path

PYTHON_EXE = sys.executable

def run_step(cmd, desc):
    print(f"\n>>> [PPO-Curriculum] Starting Phase: {desc}")
    print(f">>> Command: {cmd}")
    try:
        ret = subprocess.call(f"{PYTHON_EXE} {cmd}", shell=True)
        if ret != 0:
            print(f">>> [PPO-Curriculum] Phase failed with code {ret}")
            sys.exit(ret)
        print(f">>> [PPO-Curriculum] Phase completed successfully.")
    except KeyboardInterrupt:
        print("\n>>> [PPO-Curriculum] Interrupted.")
        sys.exit(1)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    # V47.0: Default to 5,000,000 frames per phase for better convergence.
    parser.add_argument("--steps1", type=int, default=5_000_000, help="Phase 1 frames")
    parser.add_argument("--steps2", type=int, default=5_000_000, help="Phase 2 frames")
    args = parser.parse_args()

    print("=== Snake PPO Curriculum Training ===")

    # Phase 1: Pre-train on Single Snake
    run_step(f"train_ppo.py --single --steps {args.steps1}", "1. Pre-training (Single Snake) -> agent/checkpoints/ppo_best.pth")

    # Phase 2: Fine-tune on Multi Snake (Battle)
    run_step(
        f"train_ppo.py --load agent/checkpoints/ppo_best.pth --steps {args.steps2}",
        "2. Fine-tuning (Battle + Self-Play) -> agent/checkpoints/ppo_battle_best.pth",
    )

    print("\n=== PPO Curriculum Completed ===")
    print("Best Model: agent/checkpoints/ppo_battle_best.pth")
    print("Final Snapshot: agent/checkpoints/ppo_battle_best.final.pth")

if __name__ == "__main__":
    main()
