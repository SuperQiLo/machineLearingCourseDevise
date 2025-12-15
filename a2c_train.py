"""A2C Agent 训练脚本 (Standalone)"""

from __future__ import annotations

import time
from typing import Dict

from agent.a2c_config import A2CConfig
from agent.a2c_trainer import A2CTrainer


def train() -> None:
    config = A2CConfig(
        grid_size=30,
        num_snakes=4,
        total_frames=5_000_000,
        rollout_length=128,
        lr=1e-4,
        gamma=0.99,
        entropy_coef=0.005,
        value_coef=0.5,
        max_grad_norm=10.0,
        log_interval=5_000,
        save_interval=50_000,

        # 奖励塑形（环境 reward=0，Trainer 依赖 events 计算）
        reward_food=50.0,
        reward_kill=150.0,
        reward_death=-80.0,
        reward_survive=-0.005,

        checkpoint_name="a2c_snake_latest.pth",
        trainer_checkpoint_name="a2c_snake_latest_train.pth",
        resume=True,
        device="cuda:0",
    )

    print("=== 开始 A2C Snake 训练 ===")
    print(f"Device: {config.device}")

    def progress_callback(info: Dict[str, float]) -> None:
        print(
            f"[Frame {int(info['frame'])}] "
            f"Ep={int(info['episode'])}, "
            f"AvgScore={info['avg_score']:.2f}, "
            f"Loss={info['loss']:.4f}, "
            f"Alive={int(info['alive'])}"
        )

    trainer = A2CTrainer(config, progress_cb=progress_callback)

    start_time = time.time()
    final_model = trainer.train()
    elapsed = time.time() - start_time

    print(f"=== 训练结束，耗时 {elapsed:.1f}s ===")
    print(f"模型已保存至: {final_model}")


if __name__ == "__main__":
    train()
