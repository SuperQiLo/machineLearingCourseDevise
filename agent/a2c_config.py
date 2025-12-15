"""A2C 智能体/训练器共享的配置结构。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class A2CConfig:
    """集中管理多蛇环境下 A2C 的关键超参数。"""

    grid_size: int = 30
    num_snakes: int = 4
    total_frames: int = 200_000

    # A2C 核心超参数
    rollout_length: int = 32
    gamma: float = 0.99
    lr: float = 5e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 10.0

    # 训练调度
    log_interval: int = 1_000
    save_interval: int = 25_000
    checkpoint_name: str = "a2c_snake_latest.pth"
    trainer_checkpoint_name: str = "a2c_snake_latest_train.pth"
    resume: bool = True

    # 设备
    device: str = "auto"

    # 奖励塑形（环境 reward 固定为 0，需用 events 计算）
    reward_food: float = 20.0
    reward_kill: float = 100.0
    reward_death: float = -100.0
    reward_survive: float = -1.0

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
