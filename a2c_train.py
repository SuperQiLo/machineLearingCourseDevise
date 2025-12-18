"""A2C Agent 训练脚本 (优化版)"""

from __future__ import annotations

import time
from typing import Dict

from agent.a2c_config import A2CConfig
from agent.a2c_trainer import A2CTrainer


def train() -> None:
    # 建议先用小图 + 2 蛇把策略跑通，再逐步加蛇数/地图。
    config = A2CConfig(
        grid_size=14,
        num_snakes=2,
        num_envs=8,
        num_food=3,
        max_steps=300,
        total_frames=2_000_000,
        rollout_length=64,  # 增加rollout长度
        lr=3e-4,  # 优化的学习率
        gamma=0.99,
        entropy_coef=0.02,  # 增加探索
        value_coef=0.5,
        max_grad_norm=0.5,  # 更保守的梯度裁剪
        log_interval=5_000,
        save_interval=50_000,
        
        # GAE 参数
        use_gae=True,
        gae_lambda=0.95,
        
        # 学习率调度
        lr_decay=True,
        lr_min=1e-5,

        # 环境 reward（避免“转圈”局部最优：用 step penalty 替代存活奖励）
        step_penalty=-0.01,
        food_reward=1.0,
        death_penalty=-1.0,
        kill_reward=0.5,
        distance_shaping_scale=0.01,

        checkpoint_name="a2c_snake_latest.pth",
        trainer_checkpoint_name="a2c_snake_latest_train.pth",
        resume=True,
        device="cuda:0",
    )

    print("=== 开始 A2C Snake 训练 (优化版) ===")
    print(f"Device: {config.device}")
    print(f"使用 GAE: {config.use_gae}, Lambda: {config.gae_lambda}")
    print(f"学习率调度: {config.lr_decay}, 初始LR: {config.lr}")

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
