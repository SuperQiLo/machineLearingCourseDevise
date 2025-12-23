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
    print("=== Snake PPO Curriculum Training ===")
    
    # 1. Phase 1: Pre-train on Single Snake
    # PPO base learning for basics
    run_step("train_ppo.py --single --steps 500000", "1. Pre-training (Single Snake)")
    
    # Target: agent/checkpoints/ppo_best.pth
    
    # 2. Phase 2: Fine-tune on Multi Snake (Battle)
    # Loading base weights.
    # Note: Using half steps for fine-tuning as it's harder/slower or just enough.
    run_step("train_ppo.py --load agent/checkpoints/ppo_best.pth --steps 1000000", "2. Fine-tuning (Battle Mode)")
    
    print("\n=== PPO Curriculum Completed ===")
    print("Final Model: agent/checkpoints/ppo_battle_best.pth")

if __name__ == "__main__":
    main()
