"""A2C 训练入口 - A6000 优化版。

针对 NVIDIA A6000 (48GB VRAM) 优化的训练配置。
核心策略：利用大显存增加并行环境数和批大小，提高训练吞吐量。
"""

from __future__ import annotations

import time
from typing import Dict

from agent.a2c_config import A2CConfig
from agent.a2c_trainer import A2CTrainer


def train() -> None:
    """A6000 优化配置说明：
    
    1. num_envs=32: 并行环境数量（A6000 可支持更高，但 CPU pygame 渲染可能成为瓶颈）
    2. rollout_length=256: 更长的 rollout 提供更稳定的梯度估计
    3. total_frames=10M: 充足的训练预算，确保收敛
    4. 奖励设计已包含 repetition_penalty，防止转圈圈
    
    如果 GPU 利用率低于 80%：
    - 首先增加 num_envs 到 64
    - 如果仍然低，可能是 CPU 瓶颈（pygame 渲染）
    
    如果出现 OOM：
    - 降低 num_envs 到 16
    - 或减小 rollout_length 到 128
    """
    cfg = A2CConfig(
        # ==================== 环境配置 ====================
        grid_size=30,                  # 棋盘边长
        num_snakes=4,                  # 蛇数量
        num_food=10,                   # 食物数量
        max_steps=1000,                # 每局最大步数
        
        # ==================== 观测配置 ====================
        obs_channels=3,                # RGB 通道
        obs_size=84,                   # 标准 Atari 尺寸
        
        # ==================== A6000 优化参数 ====================
        num_envs=32,                   # 并行环境（A6000 推荐 32-64）
        rollout_length=256,            # 更长的 rollout 更稳定

        # ==================== 显存保护 ====================
        # 将 (T*N) 展平批次拆成 micro-batch 做前向/反向，避免一次性喂入 CNN 导致 cuDNN 申请超大 workspace
        train_micro_batch_size=0,
        
        # ==================== 训练调度 ====================
        total_frames=10_000_000,       # 10M 帧（约 2-4 小时）
        lr=2.5e-4,                     # 学习率
        gamma=0.99,                    # 折扣因子
        entropy_coef=0.02,             # 熵系数（鼓励探索）
        value_coef=0.5,                # 价值损失权重
        max_grad_norm=0.5,             # 梯度裁剪
        
        # ==================== GAE 参数 ====================
        use_gae=True,
        gae_lambda=0.95,
        
        # ==================== 学习率衰减 ====================
        lr_decay=True,
        lr_min=1e-5,
        
        # ==================== 日志与保存 ====================
        log_interval=10_000,           # 每 10K 帧打印一次
        save_interval=500_000,         # 每 500K 帧保存一次
        checkpoint_name="a2c_snake_a6000.pth",
        device="cuda:0",
        
        # ==================== 奖励函数（防止转圈圈） ====================
        step_penalty=-0.01,            # 每步惩罚
        food_reward=1.0,               # 吃到食物奖励
        death_penalty=-1.0,            # 死亡惩罚
        kill_reward=0.5,               # 击杀奖励
        distance_shaping_scale=0.01,   # 距离塑形
        repetition_penalty=-0.05,      # 重复动作惩罚（核心防转圈机制）
    )

    print("=" * 60)
    print("A2C 图像输入训练 - A6000 优化版")
    print("=" * 60)
    print(f"设备: {cfg.device}")
    print(f"并行环境: {cfg.num_envs} | Rollout 长度: {cfg.rollout_length}")
    print(f"观测尺寸: {cfg.obs_channels}x{cfg.obs_size}x{cfg.obs_size}")
    print(f"总帧数: {cfg.total_frames:,}")
    print(f"重复动作惩罚: {cfg.repetition_penalty}")
    print("=" * 60)

    def progress_callback(info: Dict[str, float]) -> None:
        frame = int(info['frame'])
        episode = int(info['episode'])
        avg_score = info['avg_score']
        loss = info['loss']
        alive = info['alive']
        
        # 计算进度百分比
        progress = frame / cfg.total_frames * 100
        
        print(
            f"[{progress:5.1f}%] Frame {frame:>8,} | "
            f"Ep={episode:>5} | AvgReturn={avg_score:>7.3f} | "
            f"Loss={loss:>8.4f} | Alive={alive:.2f}"
        )

    trainer = A2CTrainer(cfg, progress_cb=progress_callback)

    start = time.time()
    print("\n开始训练...")
    final_path = trainer.train()
    elapsed = time.time() - start
    
    print("\n" + "=" * 60)
    print(f"训练完成！用时: {elapsed/3600:.2f} 小时")
    print(f"模型保存至: {final_path}")
    print("=" * 60)


if __name__ == "__main__":
    train()
