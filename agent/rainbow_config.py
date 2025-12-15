"""Rainbow 智能体/训练器共享的配置结构。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RainbowConfig:
    """集中管理多蛇环境下 Rainbow DQN 的关键超参数。"""

    grid_size: int = 30  # 棋盘边长，决定观测分辨率
    num_snakes: int = 4  # 同场蛇数量，会影响并行采样和竞争难度
    total_frames: int = 200_000  # 总训练帧数，控制整体迭代时长
    warmup_frames: int = 5_000  # 只收集不训练的帧数，用于填充经验池
    batch_size: int = 64  # 每次优化采样的 transition 数量
    gamma: float = 0.99  # 折扣因子
    multi_step: int = 3  # n-step 回报长度
    replay_capacity: int = 200_000  # 经验池容量
    lr: float = 5e-4  # Adam 学习率
    atom_size: int = 51  # C51 支撑原子数量
    v_min: float = -20.0  # 分布估计的最小回报
    v_max: float = 20.0  # 分布估计的最大回报
    epsilon_start: float = 1.0  # ε-greedy 起始值
    epsilon_final: float = 0.05  # ε-greedy 终值
    epsilon_decay: int = 120_000  # ε 衰减所需帧数
    update_target_interval: int = 1_000  # 目标网络同步间隔
    train_interval: int = 1  # 相邻优化步之间的帧间隔
    log_interval: int = 1_000  # 打印训练日志的帧间隔
    save_interval: int = 25_000  # 滚动保存 latest checkpoint 的帧间隔
    checkpoint_name: str = "rainbow_snake_latest.pth"  # 默认模型名
    trainer_checkpoint_name: str = "rainbow_snake_latest_train.pth"  # 训练断点（包含优化器/回放池等）
    resume: bool = True  # 若训练断点存在则自动续训
    device: str = "auto"  # 运行设备，可设 cpu/cuda:n/auto
    reward_food: float = 20.0  # 吃到食物的奖励
    reward_kill: float = 100.0  # 击杀他蛇的奖励
    reward_death: float = -100.0  # 死亡惩罚
    reward_survive: float = -1.0  # 仅存活一回合的基础损耗

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
