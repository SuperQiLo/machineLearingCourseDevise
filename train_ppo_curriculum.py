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
    # Recommended A6000 defaults (throughput + stability)
    parser.add_argument("--steps1", type=int, default=10_000_000, help="Phase 1 frames")
    parser.add_argument("--steps2", type=int, default=10_000_000, help="Phase 2 frames")
    parser.add_argument("--envs1", type=int, default=64, help="Phase 1 num envs (A6000 rec: 64)")
    parser.add_argument("--envs2", type=int, default=64, help="Phase 2 num envs (A6000 rec: 64)")
    parser.add_argument("--rollout-steps1", type=int, default=256, help="Phase 1 rollout steps per env (A6000 rec: 256)")
    parser.add_argument("--rollout-steps2", type=int, default=128, help="Phase 2 rollout steps per env (A6000 rec: 128)")
    # Performance-oriented defaults (higher FPS; still stable for this project)
    parser.add_argument("--update-epochs1", type=int, default=2, help="Phase 1 PPO update epochs (default: 2)")
    parser.add_argument("--update-epochs2", type=int, default=2, help="Phase 2 PPO update epochs (default: 2)")
    parser.add_argument("--minibatch-size1", type=int, default=4096, help="Phase 1 PPO minibatch size (default: 4096)")
    parser.add_argument("--minibatch-size2", type=int, default=4096, help="Phase 2 PPO minibatch size (default: 4096)")
    parser.add_argument("--lr1", type=float, default=2.0e-4, help="Phase 1 base LR")
    parser.add_argument("--lr2", type=float, default=1.5e-4, help="Phase 2 base LR (used with --load)")
    parser.add_argument("--target-kl1", type=float, default=0.015, help="Phase 1 target KL")
    parser.add_argument("--target-kl2", type=float, default=0.015, help="Phase 2 target KL")
    # NOTE: train_ppo.py applies finetune_lr_mult when --load is used.
    # To make the effective fine-tune LR explicit, set finetune_lr2 by default.
    parser.add_argument("--finetune-lr2", type=float, default=5.0e-5, help="Phase 2 fine-tune LR (effective LR when --load)")
    parser.add_argument("--finetune-lr-mult2", type=float, default=None, help="Phase 2 fine-tune LR multiplier (ignored if --finetune-lr2 is set)")
    parser.add_argument("--finetune-target-kl2", type=float, default=0.030, help="Phase 2 fine-tune target KL (only used with --load)")
    parser.add_argument("--self-play-prob2", type=float, default=0.30, help="Phase 2 battle self-play probability (0 disables self-play for speed)")
    parser.add_argument("--force", action="store_true", help="Force restart from Phase 1")
    args = parser.parse_args()

    print("=== Snake PPO Curriculum Training ===")

    # Phase 1: Pre-train on Single Snake
    p1_final = Path("agent/checkpoints/ppo_best.final.pth")
    if p1_final.exists() and not args.force:
        print(f"\n>>> [PPO-Curriculum] Phase 1 already completed ({p1_final} found). Skipping...")
    else:
        p1_cmd = f"train_ppo.py --single --steps {args.steps1}"
        if args.envs1 is not None:
            p1_cmd += f" --envs {args.envs1}"
        if args.rollout_steps1 is not None:
            p1_cmd += f" --rollout-steps {args.rollout_steps1}"
        if args.update_epochs1 is not None:
            p1_cmd += f" --update-epochs {args.update_epochs1}"
        if args.minibatch_size1 is not None:
            p1_cmd += f" --minibatch-size {args.minibatch_size1}"
        if args.lr1 is not None:
            p1_cmd += f" --lr {args.lr1}"
        if args.target_kl1 is not None:
            p1_cmd += f" --target-kl {args.target_kl1}"
        run_step(p1_cmd, "1. Pre-training (Single Snake) -> agent/checkpoints/ppo_best.pth")

    # Phase 2: Fine-tune on Multi Snake (Battle)
    p2_cmd = f"train_ppo.py --load agent/checkpoints/ppo_best.final.pth --steps {args.steps2}"
    if args.envs2 is not None:
        p2_cmd += f" --envs {args.envs2}"
    if args.rollout_steps2 is not None:
        p2_cmd += f" --rollout-steps {args.rollout_steps2}"
    if args.update_epochs2 is not None:
        p2_cmd += f" --update-epochs {args.update_epochs2}"
    if args.minibatch_size2 is not None:
        p2_cmd += f" --minibatch-size {args.minibatch_size2}"
    if args.lr2 is not None:
        p2_cmd += f" --lr {args.lr2}"
    if args.target_kl2 is not None:
        p2_cmd += f" --target-kl {args.target_kl2}"
    if args.finetune_lr2 is not None:
        p2_cmd += f" --finetune-lr {args.finetune_lr2}"
    if args.finetune_lr_mult2 is not None:
        p2_cmd += f" --finetune-lr-mult {args.finetune_lr_mult2}"
    if args.finetune_target_kl2 is not None:
        p2_cmd += f" --finetune-target-kl {args.finetune_target_kl2}"
    if args.self_play_prob2 is not None:
        p2_cmd += f" --self-play-prob {args.self_play_prob2}"
    run_step(
        p2_cmd,
        "2. Fine-tuning (Battle + Self-Play) -> agent/checkpoints/ppo_battle_best.final.pth",
    )

    print("\n=== PPO Curriculum Completed ===")
    print("Best Model: agent/checkpoints/ppo_battle_best.pth")
    print("Final Snapshot: agent/checkpoints/ppo_battle_best.final.pth")

if __name__ == "__main__":
    main()
