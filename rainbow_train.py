"""Rainbow 训练入口 - A6000 优化版。

针对 NVIDIA A6000 (48GB VRAM) 优化的训练配置。
核心策略：利用大显存增加批大小和经验池容量，加速收敛。
"""

from __future__ import annotations

import time
from typing import Dict

from agent.rainbow_config import RainbowConfig
from agent.rainbow_trainer import RainbowTrainer


def train() -> None:
    """A6000 优化配置说明：
    
    1. batch_size=512: 大批量训练更稳定（A6000 可支持 1024+）
    2. num_envs=32: 并行采样加速（CPU pygame 渲染可能是瓶颈）
    3. replay_capacity=2M: 大经验池提供更多样化的训练样本
    4. total_frames=10M: 充足的训练预算
    
    如果 GPU 利用率低于 80%：
    - 增加 batch_size 到 1024
    - 或增加 num_envs 到 64
    
    如果出现 OOM：
    - 降低 batch_size 到 256
    - 或减小 replay_capacity 到 1M
    """
    cfg = RainbowConfig(
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
        batch_size=512,                # 批大小（A6000 可用 512-1024）
        replay_capacity=2_000_000,     # 经验池 2M（A6000 内存充足）
        
        # ==================== 训练调度 ====================
        total_frames=10_000_000,       # 10M 帧（约 3-6 小时）
        warmup_frames=50_000,          # 预热帧数
        train_interval=4,              # 每 4 帧训练一次
        lr=1e-4,                       # 学习率
        gamma=0.99,                    # 折扣因子
        multi_step=5,                  # n-step 回报
        
        # ==================== C51 分布式 RL ====================
        atom_size=51,                  # 价值分布支撑点
        v_min=-10.0,                   # 价值分布最小值
        v_max=10.0,                    # 价值分布最大值
        
        # ==================== 探索策略 ====================
        epsilon_start=1.0,             # 初始探索率
        epsilon_final=0.01,            # 最终探索率
        epsilon_decay=500_000,         # 探索衰减帧数
        
        # ==================== 网络更新 ====================
        update_target_interval=5_000,  # 目标网络同步间隔
        
        # ==================== 日志与保存 ====================
        log_interval=10_000,           # 每 10K 帧打印一次
        save_interval=500_000,         # 每 500K 帧保存一次
        checkpoint_name="rainbow_snake_a6000.pth",
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
    print("Rainbow 图像输入训练 - A6000 优化版")
    print("=" * 60)
    print(f"设备: {cfg.device}")
    print(f"并行环境: {cfg.num_envs} | 批大小: {cfg.batch_size}")
    print(f"经验池容量: {cfg.replay_capacity:,}")
    print(f"观测尺寸: {cfg.obs_channels}x{cfg.obs_size}x{cfg.obs_size}")
    print(f"总帧数: {cfg.total_frames:,}")
    print(f"C51 范围: [{cfg.v_min}, {cfg.v_max}]")
    print(f"重复动作惩罚: {cfg.repetition_penalty}")
    print("=" * 60)

    def progress_callback(info: Dict[str, float]) -> None:
        frame = int(info['frame'])
        episode = int(info['episode'])
        avg_score = info['avg_score']
        loss = info.get('loss', 0.0)
        epsilon = info.get('epsilon', 0.0)
        alive = info.get('alive', 0.0)
        
        # 计算进度百分比
        progress = frame / cfg.total_frames * 100
        
        print(
            f"[{progress:5.1f}%] Frame {frame:>8,} | "
            f"Ep={episode:>5} | AvgReturn={avg_score:>7.3f} | "
            f"Loss={loss:>8.4f} | Eps={epsilon:.3f} | Alive={alive:.2f}"
        )

    trainer = RainbowTrainer(cfg, progress_cb=progress_callback)

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
