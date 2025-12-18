"""Rainbow 课程式训练入口。

阶段：单蛇吃食物 -> 两蛇对抗 -> 四蛇互博。
阶段之间继承模型权重（在线/目标网络），经验池与优化器默认重置。

运行：
  python rainbow_curriculum_train.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from agent.rainbow_config import RainbowConfig
from agent.rainbow_trainer import RainbowTrainer


@dataclass(frozen=True)
class Stage:
    name: str
    grid_size: int
    num_snakes: int
    num_food: int
    max_steps: int
    total_frames: int  # 环境步数（每个 env 的一步算 1）
    warmup_frames: int


def train() -> None:
    stages = [
        Stage("stage1_single", grid_size=12, num_snakes=1, num_food=2, max_steps=200, total_frames=400_000, warmup_frames=10_000),
        Stage("stage2_duel", grid_size=14, num_snakes=2, num_food=3, max_steps=300, total_frames=800_000, warmup_frames=20_000),
        # 最终阶段：在 30x30 地图上完成训练
        Stage("stage3_brawl", grid_size=30, num_snakes=4, num_food=10, max_steps=1_200, total_frames=2_800_000, warmup_frames=80_000),
    ]

    base = RainbowConfig(
        grid_size=14,
        num_snakes=2,
        obs_channels=10,
        # A6000：提高并行环境与 batch_size，显著提升吞吐
        num_envs=32,
        num_food=3,
        max_steps=300,
        total_frames=800_000,
        warmup_frames=40_000,
        batch_size=512,
        gamma=0.99,
        multi_step=5,
        replay_capacity=1_500_000,
        lr=1e-4,
        atom_size=51,
        v_min=-3.0,
        v_max=3.0,
        epsilon_start=1.0,
        epsilon_final=0.05,
        epsilon_decay=800_000,
        update_target_interval=5_000,
        train_interval=1,
        log_interval=10_000,
        save_interval=50_000,
        resume=False,
        device="cuda:0",
        # reward
        step_penalty=-0.01,
        food_reward=1.0,
        death_penalty=-1.0,
        kill_reward=0.5,
        distance_shaping_scale=0.01,
    )

    prev_model: Optional[Path] = None

    print("=== 开始 Rainbow 课程式训练 ===")
    print(f"Device: {base.device} | num_envs={base.num_envs} | obs_channels={base.obs_channels}")

    for stage in stages:
        cfg = RainbowConfig(
            **{
                **base.__dict__,
                "grid_size": stage.grid_size,
                "num_snakes": stage.num_snakes,
                "num_food": stage.num_food,
                "max_steps": stage.max_steps,
                "total_frames": stage.total_frames,
                "warmup_frames": stage.warmup_frames,
                "checkpoint_name": f"rainbow_{stage.name}_latest.pth",
                "trainer_checkpoint_name": f"rainbow_{stage.name}_train.pth",
                "resume": False,
            }
        )

        def progress_callback(info: Dict[str, float]) -> None:
            print(
                f"[{stage.name}] Frame {int(info['frame'])} "
                f"Ep={int(info['episode'])} AvgReturn={info['avg_score']:.3f} "
                f"Loss={info.get('loss', 0.0):.4f} Eps={info.get('epsilon', 0.0):.3f} Alive={info.get('alive', 0.0):.2f}"
            )

        trainer = RainbowTrainer(cfg, progress_cb=progress_callback)

        if prev_model is not None and prev_model.exists():
            print(f"[{stage.name}] Load prev weights: {prev_model}")
            trainer.agent.load(prev_model)
            trainer.agent.sync_target()

        start = time.time()
        final_path = trainer.train()
        elapsed = time.time() - start
        print(f"[{stage.name}] Done in {elapsed:.1f}s -> {final_path}")
        prev_model = final_path

    if prev_model is not None:
        print(f"=== 课程训练完成，最终模型: {prev_model} ===")


if __name__ == "__main__":
    train()
