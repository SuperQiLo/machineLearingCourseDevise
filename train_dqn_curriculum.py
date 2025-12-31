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
    # User-Defined Training Steps (V16.0: 5M/3M to saturate 1024-width net)
    parser.add_argument("--steps1", type=int, default=5_000_000, help="Phase 1 frames")
    parser.add_argument("--steps2", type=int, default=3_000_000, help="Phase 2 frames")
    parser.add_argument("--force", action="store_true", help="Force restart from Phase 1")
    args = parser.parse_args()

    v = args.variant.lower()
    log_dir = Path("agent/checkpoints")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    p1_model = log_dir / f"{v}_pretrain.pth"
    p2_model = log_dir / f"{v}_battle.pth"

    print(f"=== Snake AI Curriculum Training [{v.upper()}] ===")
    
    # Phase 1: Pre-train on Single Snake
    p1_final = p1_model.with_suffix(".final.pth")
    if p1_final.exists() and not args.force:
        print(f"\n>>> [Curriculum] Phase 1 already completed ({p1_final} found). Skipping...")
    else:
        cmd1 = f"train_dqn_variants.py --variant {v} --single --steps {args.steps1} --save {p1_model}"
        run_step(cmd1, f"1. Pre-training (Single Snake) -> {p1_model}")
    
    # Phase 2: Fine-tune on Multi Snake (Battle)
    p1_final = p1_model.with_suffix(".final.pth")
    cmd2 = f"train_dqn_variants.py --variant {v} --load {p1_final} --steps {args.steps2} --save {p2_model}"
    run_step(cmd2, f"2. Fine-tuning (Battle Mode) -> {p2_model}")
    
    print("\n=== Curriculum Completed ===")
    print(f"Best Model: {p2_model}")
    print(f"Final Snapshot: {p2_model.with_suffix('.final.pth')}")
    print(f"Test it: python gui_game.py --mode battle --algo {v} --model {p2_model}")

if __name__ == "__main__":
    main()
