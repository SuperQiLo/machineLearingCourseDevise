"""
Curriculum Training Script.
Automates the 'Single Snake -> Battle Snake' curriculum.
"""

import subprocess
import time
import sys
from pathlib import Path
import shlex

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

    # Optional tuning knobs forwarded to train_dqn_variants.py
    # Phase 1 (single)
    parser.add_argument("--eps-start1", type=float, default=None, help="Override epsilon start for Phase 1 (single)")
    parser.add_argument("--eps-min1", type=float, default=None, help="Override epsilon min for Phase 1 (single)")
    # Phase 2 (battle)
    parser.add_argument("--eps-start2", type=float, default=None, help="Override epsilon start for Phase 2 (battle)")
    parser.add_argument("--eps-min2", type=float, default=None, help="Override epsilon min for Phase 2 (battle)")
    parser.add_argument("--sp-prob-start", type=float, default=None, help="Self-play prob early in Phase 2")
    parser.add_argument("--sp-prob-end", type=float, default=None, help="Self-play prob late in Phase 2")
    parser.add_argument("--sp-prob-frac", type=float, default=None, help="Switch point fraction for self-play prob in Phase 2")
    args = parser.parse_args()

    v = args.variant.lower()
    log_dir = Path("agent/checkpoints")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    p1_model = log_dir / f"{v}_pretrain.pth"
    p2_model = log_dir / f"{v}_battle.pth"

    print(f"=== Snake AI Curriculum Training [{v.upper()}] ===")

    def _build_variant_args(*, eps_start=None, eps_min=None, sp_prob_start=None, sp_prob_end=None, sp_prob_frac=None) -> str:
        extra = []
        if eps_start is not None:
            extra += ["--eps-start", str(eps_start)]
        if eps_min is not None:
            extra += ["--eps-min", str(eps_min)]
        if sp_prob_start is not None:
            extra += ["--sp-prob-start", str(sp_prob_start)]
        if sp_prob_end is not None:
            extra += ["--sp-prob-end", str(sp_prob_end)]
        if sp_prob_frac is not None:
            extra += ["--sp-prob-frac", str(sp_prob_frac)]
        return " ".join(shlex.quote(x) for x in extra)
    
    # Phase 1: Pre-train on Single Snake
    p1_final = p1_model.with_suffix(".final.pth")
    if p1_final.exists() and not args.force:
        print(f"\n>>> [Curriculum] Phase 1 already completed ({p1_final} found). Skipping...")
    else:
        extra1 = _build_variant_args(eps_start=args.eps_start1, eps_min=args.eps_min1)
        cmd1 = f"train_dqn_variants.py --variant {v} --single --steps {args.steps1} --save {p1_model} {extra1}".strip()
        run_step(cmd1, f"1. Pre-training (Single Snake) -> {p1_model}")
    
    # Phase 2: Fine-tune on Multi Snake (Battle)
    p1_final = p1_model.with_suffix(".final.pth")
    extra2 = _build_variant_args(
        eps_start=args.eps_start2,
        eps_min=args.eps_min2,
        sp_prob_start=args.sp_prob_start,
        sp_prob_end=args.sp_prob_end,
        sp_prob_frac=args.sp_prob_frac,
    )
    cmd2 = f"train_dqn_variants.py --variant {v} --load {p1_final} --steps {args.steps2} --save {p2_model} {extra2}".strip()
    run_step(cmd2, f"2. Fine-tuning (Battle Mode) -> {p2_model}")
    
    print("\n=== Curriculum Completed ===")
    print(f"Best Model: {p2_model}")
    print(f"Final Snapshot: {p2_model.with_suffix('.final.pth')}")
    print(f"Test it: python gui_game.py --mode battle --algo {v} --model {p2_model}")

if __name__ == "__main__":
    main()
