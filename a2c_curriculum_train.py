"""A2C 课程式训练入口。

思路：先在简单任务上学会“找食物”，再逐步提高难度（更多蛇、更大地图、更长局）。
阶段之间继承模型权重（默认不继承优化器状态，便于稳定过渡）。

运行：
  python a2c_curriculum_train.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from agent.a2c_agent import A2CAgent
from agent.a2c_config import A2CConfig
from agent.a2c_trainer import A2CTrainer


@dataclass(frozen=True)
class Stage:
    name: str
    grid_size: int
    num_snakes: int
    num_food: int
    max_steps: int
    total_frames: int  # 环境步数（每个 env 的一步算 1）


def train() -> None:
    # 课程：从“单蛇吃食物”到“多蛇互博”
    stages = [
        Stage("stage1_single", grid_size=12, num_snakes=1, num_food=2, max_steps=200, total_frames=300_000),
        Stage("stage2_duel", grid_size=14, num_snakes=2, num_food=3, max_steps=300, total_frames=600_000),
        # 最终阶段：在 30x30 地图上完成训练
        Stage("stage3_brawl", grid_size=30, num_snakes=4, num_food=10, max_steps=1_200, total_frames=2_400_000),
    ]

    # A6000 建议：提高并行环境数以吃满 GPU
    base = A2CConfig(
        # 这些会被 stage 覆盖
        grid_size=14,
        num_snakes=2,
        num_envs=32,
        obs_channels=10,
        num_food=3,
        max_steps=300,
        total_frames=600_000,
        rollout_length=128,
        lr=3e-4,
        gamma=0.99,
        entropy_coef=0.02,
        value_coef=0.5,
        max_grad_norm=0.5,
        use_gae=True,
        gae_lambda=0.95,
        lr_decay=True,
        lr_min=1e-5,
        log_interval=20_000,
        save_interval=50_000,
        resume=False,
        device="cuda:0",
        # reward（与 env 一致）
        step_penalty=-0.01,
        food_reward=1.0,
        death_penalty=-1.0,
        kill_reward=0.5,
        distance_shaping_scale=0.01,
    )

    prev_model: Optional[Path] = None

    print("=== 开始 A2C 课程式训练 ===")
    print(f"Device: {base.device} | num_envs={base.num_envs} | obs_channels={base.obs_channels}")

    for idx, stage in enumerate(stages, start=1):
        cfg = A2CConfig(
            **{
                **base.__dict__,
                "grid_size": stage.grid_size,
                "num_snakes": stage.num_snakes,
                "num_food": stage.num_food,
                "max_steps": stage.max_steps,
                "total_frames": stage.total_frames,
                "checkpoint_name": f"a2c_{stage.name}_latest.pth",
                "trainer_checkpoint_name": f"a2c_{stage.name}_train.pth",
                "resume": False,
            }
        )

        def progress_callback(info: Dict[str, float]) -> None:
            print(
                f"[{stage.name}] Frame {int(info['frame'])} "
                f"Ep={int(info['episode'])} AvgReturn={info['avg_score']:.3f} "
                f"Loss={info['loss']:.4f} Alive={info['alive']:.2f}"
            )

        trainer = A2CTrainer(cfg, progress_cb=progress_callback)

        # 继承上一阶段的权重（只继承网络参数，优化器重置更稳）
        if prev_model is not None and prev_model.exists():
            print(f"[{stage.name}] Load prev weights: {prev_model}")
            trainer.agent.load(prev_model)

        start = time.time()
        final_path = trainer.train()
        elapsed = time.time() - start
        print(f"[{stage.name}] Done in {elapsed:.1f}s -> {final_path}")
        prev_model = final_path

    # 复制/汇总：最后阶段模型作为总模型
    if prev_model is not None:
        print(f"=== 课程训练完成，最终模型: {prev_model} ===")


if __name__ == "__main__":
    train()
