"""Rainbow Agent 训练脚本 (Standalone)"""

from __future__ import annotations

import time
from typing import Dict

from agent.rainbow_config import RainbowConfig
from agent.rainbow_trainer import RainbowTrainer


def train() -> None:
    # -------------------------------------------------------------------------
    # 训练参数配置
    # -------------------------------------------------------------------------
    config = RainbowConfig(
        grid_size=30,
        num_snakes=4,
        total_frames=5_000_000,
        warmup_frames=50_000,
        batch_size=256,
        lr=1e-4,
        replay_capacity=1_000_000,
        train_interval=1,
        update_target_interval=5_000,
        log_interval=5_000,
        save_interval=50_000,
        
        # 探索策略
        epsilon_start=1.0,
        epsilon_final=0.05,
        epsilon_decay=2_000_000,

        # 模型保存名称 (saved to agent/checkpoints/)
        checkpoint_name="rainbow_snake_latest.pth",
        trainer_checkpoint_name="rainbow_snake_latest_train.pth",
        resume=True,

        # A6000 训练建议显式指定
        device="cuda:0",

        # 奖励塑形（环境 reward=0，Trainer 依赖 events 计算）
        reward_food=50.0,
        reward_kill=150.0,
        reward_death=-80.0,
        reward_survive=-0.005,
        
      
    )

    print("=== 开始 Rainbow Snake 训练 ===")
    print(f"Device: {config.device}")
    
    # 定义简单的进度回调
    def progress_callback(info: Dict[str, float]) -> None:
        print(f"[Frame {int(info['frame'])}] "
              f"Ep={int(info['episode'])}, "
              f"AvgScore={info['avg_score']:.2f}, "
              f"Loss={info.get('loss', 0.0):.4f}, "
              f"Eps={info['epsilon']:.3f}, "
              f"Alive={int(info['alive'])}")

    trainer = RainbowTrainer(config, progress_cb=progress_callback)
    
    # 开始训练
    start_time = time.time()
    final_model = trainer.train()
    elapsed = time.time() - start_time
    
    print(f"=== 训练结束，耗时 {elapsed:.1f}s ===")
    print(f"模型已保存至: {final_model}")


if __name__ == "__main__":
    train()
