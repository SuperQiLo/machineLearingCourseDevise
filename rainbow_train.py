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
        total_frames=200_000,
        warmup_frames=5_000,
        batch_size=64,
        lr=5e-4,
        replay_capacity=200_000,
        train_interval=1,
        update_target_interval=1_000,
        log_interval=1_000,
        save_interval=25_000,
        
        # 探索策略
        epsilon_start=1.0,
        epsilon_final=0.05,
        epsilon_decay=120_000,

        # 模型保存名称 (saved to agent/checkpoints/)
        checkpoint_name="rainbow_snake_latest.pth",
        
        # 奖励配置 (由 Trainer/Env 内部处理, 或后续扩展)
        # 目前 Env 默认无奖励，RainbowTrainer 需要修改以传入自定义奖励函数
        # 但为了保持简单，我们先使用 Trainer 的默认逻辑，
        # 注意: 用户之前删除了 Env 的奖励代码，所以我们需要在 Trainer 里实现奖励计算，
        # 或者在 Config 里增加 Reward 并在 Trainer Loop 里计算。
        # 由于 Env 已经变成了纯逻辑 不返回奖励，RainbowTrainer 需要适配。
    )

    print("=== 开始 Rainbow Snake 训练 ===")
    print(f"Device: {config.device}")
    
    # 定义简单的进度回调
    def progress_callback(info: Dict[str, float]) -> None:
        print(f"[Frame {int(info['frame'])}] "
              f"Ep={int(info['episode'])}, "
              f"AvgScore={info['avg_score']:.2f}, "
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
