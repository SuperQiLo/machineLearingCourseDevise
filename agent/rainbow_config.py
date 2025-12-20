"""Rainbow 智能体/训练器共享的配置结构。

本配置类集中管理 Rainbow DQN 算法的所有超参数，包括：
- 环境相关参数（棋盘尺寸、蛇数量等）
- 网络结构参数（观测尺寸、原子数等）
- 训练调度参数（学习率、批大小、探索策略等）
- 奖励函数参数（由环境层统一计算）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RainbowConfig:
    """Rainbow DQN 算法的核心配置类。

    Rainbow 是一种结合多种 DQN 改进技术的强化学习算法，包括：
    - Dueling DQN：分离状态价值和动作优势
    - NoisyNet：参数化噪声实现探索
    - Prioritized Experience Replay：优先经验回放
    - Multi-step Learning：多步回报估计
    - Distributional RL (C51)：学习价值分布而非期望
    """

    # ==================== 环境参数 ====================
    grid_size: int = 30                # 棋盘边长（正方形地图）
    num_snakes: int = 4                # 同场蛇数量
    obs_channels: int = 3              # 观测图像通道数（RGB）
    obs_size: int = 84                 # 观测图像尺寸（84x84 是标准 Atari 尺寸）
    num_envs: int = 8                  # 并行环境数量
    num_food: int = 6                  # 食物数量
    max_steps: int = 1000              # 每局最大步数

    # ==================== 训练调度参数 ====================
    total_frames: int = 500_000        # 总训练帧数
    warmup_frames: int = 10_000        # 预热帧数（仅收集经验不训练）
    batch_size: int = 128              # 批大小
    gamma: float = 0.99                # 折扣因子（未来奖励的衰减系数）
    multi_step: int = 5                # n-step 回报的步数
    replay_capacity: int = 500_000     # 经验池容量
    lr: float = 1e-4                   # 学习率

    # ==================== C51 分布式 RL 参数 ====================
    atom_size: int = 51                # 价值分布的支撑点数量
    v_min: float = -10.0               # 价值分布的最小值
    v_max: float = 10.0                # 价值分布的最大值

    # ==================== 探索策略参数 (ε-greedy) ====================
    epsilon_start: float = 1.0         # 初始探索率（完全随机）
    epsilon_final: float = 0.01        # 最终探索率
    epsilon_decay: int = 200_000       # 探索率衰减的帧数

    # ==================== 网络更新参数 ====================
    update_target_interval: int = 2_000  # 目标网络同步间隔
    train_interval: int = 4              # 训练间隔（每N帧训练一次）

    # ==================== 日志与保存参数 ====================
    log_interval: int = 1_000          # 日志打印间隔
    save_interval: int = 25_000        # 模型保存间隔
    checkpoint_name: str = "rainbow_snake_latest.pth"
    device: str = "auto"               # 运行设备（auto/cpu/cuda）

    # ==================== 奖励函数参数（由环境统一计算） ====================
    step_penalty: float = -0.01        # 每步惩罚
    food_reward: float = 1.0           # 吃到食物奖励
    death_penalty: float = -1.0        # 死亡惩罚
    kill_reward: float = 0.5           # 击杀奖励
    distance_shaping_scale: float = 0.01  # 距离塑形系数
    repetition_penalty: float = -0.05  # 重复动作惩罚

    def checkpoint_path(self) -> Path:
        """返回用于保存/加载模型的默认路径，并确保目录已创建。"""
        path = Path("agent/checkpoints") / self.checkpoint_name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
