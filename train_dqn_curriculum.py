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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="dqn", choices=["dqn", "ddqn", "per", "dueling"])
    parser.add_argument("--steps1", type=int, default=500000, help="Phase 1 steps")
    parser.add_argument("--steps2", type=int, default=500000, help="Phase 2 steps")
    args = parser.parse_args()

    v = args.variant.lower()
    log_dir = Path("agent/checkpoints")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    p1_model = log_dir / f"{v}_pretrain.pth"
    p2_model = log_dir / f"{v}_battle.pth"

    print(f"=== Snake AI Curriculum Training [{v.upper()}] ===")
    
    # Phase 1: Pre-train on Single Snake
    cmd1 = f"train_dqn_variants.py --variant {v} --single --steps {args.steps1} --save {p1_model}"
    run_step(cmd1, f"1. Pre-training (Single Snake) -> {p1_model}")
    
    # Phase 2: Fine-tune on Multi Snake (Battle)
    cmd2 = f"train_dqn_variants.py --variant {v} --load {p1_model} --steps {args.steps2} --save {p2_model}"
    run_step(cmd2, f"2. Fine-tuning (Battle Mode) -> {p2_model}")
    
    print("\n=== Curriculum Completed ===")
    print(f"Final Model: {p2_model}")
    print(f"Test it: python gui_game.py --mode battle --algo {v} --model {p2_model}")

if __name__ == "__main__":
    main()
