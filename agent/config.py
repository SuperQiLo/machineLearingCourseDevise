"""Rainbow 智能体/训练器共享的配置结构。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RainbowConfig:
    """集中管理多蛇环境下 Rainbow DQN 的关键超参数。"""

    grid_size: int = 30
    num_snakes: int = 4
    total_frames: int = 200_000
    warmup_frames: int = 5_000
    batch_size: int = 64
    gamma: float = 0.99
    multi_step: int = 3
    replay_capacity: int = 200_000
    lr: float = 5e-4
    atom_size: int = 51
    v_min: float = -20.0
    v_max: float = 20.0
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay: int = 120_000
    update_target_interval: int = 1_000
    train_interval: int = 1
    log_interval: int = 1_000
    save_interval: int = 25_000
    checkpoint_name: str = "rainbow_snake_latest.pth"
    device: str = "auto"

    def checkpoint_path(self) -> Path:
        """返回用于保存/加载模型的默认路径，并确保目录已创建。"""

        path = Path("agent/checkpoints") / self.checkpoint_name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
