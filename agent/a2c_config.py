"""A2C 智能体/训练器共享的配置结构。

本配置类集中管理 A2C (Advantage Actor-Critic) 算法的所有超参数，包括：
- 环境相关参数（棋盘尺寸、蛇数量等）
- 网络结构参数（观测尺寸等）
- 训练调度参数（学习率、rollout长度、GAE参数等）
- 奖励函数参数（由环境层统一计算）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class A2CConfig:
    """A2C 算法的核心配置类。

    A2C (Advantage Actor-Critic) 是一种同步的策略梯度算法：
    - Actor：学习策略函数 π(a|s)，决定在状态s下选择动作a的概率
    - Critic：学习价值函数 V(s)，评估状态s的好坏
    - Advantage：A(s,a) = Q(s,a) - V(s)，衡量动作a相对于平均水平的优势
    """

    # ==================== 环境参数 ====================
    grid_size: int = 30                # 棋盘边长（正方形地图）
    num_snakes: int = 4                # 同场蛇数量
    obs_channels: int = 3              # 观测图像通道数（RGB）
    obs_size: int = 84                 # 观测图像尺寸
    num_envs: int = 8                  # 并行环境数量（越多越稳定，但内存消耗更大）
    num_food: int = 6                  # 食物数量
    max_steps: int = 1000              # 每局最大步数

    # ==================== 训练调度参数 ====================
    total_frames: int = 200_000        # 总训练帧数
    rollout_length: int = 64           # 每次更新收集的步数（越长梯度越稳定）

    # ==================== 性能/显存参数 ====================
    # 0 表示自动：根据显存粗估初值，并在 OOM 时自动二分回退
    train_micro_batch_size: int = 0  # 将 (T*N) 展平批次分块训练，避免单次前向/反向显存峰值过高

    # ==================== A2C 核心超参数 ====================
    gamma: float = 0.99                # 折扣因子
    lr: float = 3e-4                   # 学习率
    entropy_coef: float = 0.02         # 熵系数（鼓励探索，防止过早收敛）
    value_coef: float = 0.5            # 价值损失权重
    max_grad_norm: float = 0.5         # 梯度裁剪阈值

    # ==================== GAE (广义优势估计) 参数 ====================
    use_gae: bool = True               # 是否使用 GAE
    gae_lambda: float = 0.95           # GAE 的 λ 参数（平衡偏差和方差）

    # ==================== 学习率调度 ====================
    lr_decay: bool = True              # 是否使用学习率衰减
    lr_min: float = 1e-5               # 最小学习率

    # ==================== 日志与保存参数 ====================
    log_interval: int = 1_000          # 日志打印间隔
    save_interval: int = 25_000        # 模型保存间隔
    checkpoint_name: str = "a2c_snake_latest.pth"
    device: str = "auto"               # 运行设备

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
