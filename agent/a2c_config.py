"""A2C 智能体/训练器共享的配置结构 (优化版)。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class A2CConfig:
    """集中管理多蛇环境下 A2C 的关键超参数 (优化版)。"""

    grid_size: int = 30
    num_snakes: int = 4
    obs_channels: int = 10
    num_envs: int = 8
    num_food: int = 6
    max_steps: int = 1000
    total_frames: int = 200_000

    # A2C 核心超参数 - 优化后
    rollout_length: int = 64  # 增加rollout长度以获得更稳定的梯度
    gamma: float = 0.99
    lr: float = 3e-4  # 稍微降低学习率
    entropy_coef: float = 0.02  # 增加熵系数鼓励探索
    value_coef: float = 0.5
    max_grad_norm: float = 0.5  # 降低梯度裁剪阈值，更稳定

    # GAE (Generalized Advantage Estimation) 参数
    use_gae: bool = True
    gae_lambda: float = 0.95

    # 学习率调度
    lr_decay: bool = True
    lr_min: float = 1e-5

    # 训练调度
    log_interval: int = 1_000
    save_interval: int = 25_000
    checkpoint_name: str = "a2c_snake_latest.pth"
    trainer_checkpoint_name: str = "a2c_snake_latest_train.pth"
    resume: bool = True

    # 设备
    device: str = "auto"

    # 环境 reward（统一由 env 层计算）
    step_penalty: float = -0.01
    food_reward: float = 1.0
    death_penalty: float = -1.0
    kill_reward: float = 0.5
    distance_shaping_scale: float = 0.01

    def checkpoint_path(self) -> Path:
        """返回用于保存/加载模型的默认路径，并确保目录已创建。"""

        path = Path("agent/checkpoints") / self.checkpoint_name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def trainer_checkpoint_path(self) -> Path:
        """返回用于断点续训的训练 checkpoint 路径，并确保目录已创建。"""

        path = Path("agent/checkpoints") / self.trainer_checkpoint_name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
