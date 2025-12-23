"""
Curriculum Training Script.
Automates the 'Single Snake -> Battle Snake' curriculum.
"""

import subprocess
import time
import sys
from pathlib import Path

PYTHON_EXE = sys.executable

def run_step(cmd, desc):
    print(f"\n>>> [Curriculum] Starting Phase: {desc}")
    print(f">>> Command: {cmd}")
    try:
        # Use simple os.system or subprocess.run
        # We want to see output live?
        ret = subprocess.call(f"{PYTHON_EXE} {cmd}", shell=True)
        if ret != 0:
            print(f">>> [Curriculum] Phase failed with code {ret}")
            sys.exit(ret)
        print(f">>> [Curriculum] Phase completed successfully.")
    except KeyboardInterrupt:
        print("\n>>> [Curriculum] Interrupted.")
        sys.exit(1)

def main():
    print("=== Snake AI Curriculum Training ===")
    print("Goal: Train a high-performance multi-agent model.")
    
    # Check if we should skip Phase 1?
    # No, assuming full run.
    
    # Phase 1: Pre-train on Single Snake
    # This teaches basic movement and food finding without enemy complexity
    run_step("train_dqn.py --single", "1. Pre-training (Single Snake)")
    
    # The output of Phase 1 is agent/checkpoints/dqn_best.pth
    
    # Phase 2: Fine-tune on Multi Snake (Battle)
    # We load the Phase 1 model and use it to initialize the Battle model
    # The script will save to agent/checkpoints/dqn_battle_best.pth
    run_step("train_dqn.py --load agent/checkpoints/dqn_best.pth", "2. Fine-tuning (Battle Mode)")
    
    print("\n=== Curriculum Completed ===")
    print("Result: agent/checkpoints/dqn_battle_best.pth")
    print("Test it: python gui_game.py --mode battle --algo dqn --model agent/checkpoints/dqn_battle_best.pth")

if __name__ == "__main__":
    main()
