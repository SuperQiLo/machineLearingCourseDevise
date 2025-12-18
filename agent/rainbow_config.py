"""Rainbow 智能体/训练器共享的配置结构 (优化版)。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RainbowConfig:
    """集中管理多蛇环境下 Rainbow DQN 的关键超参数 (优化版)。"""

    grid_size: int = 30  # 棋盘边长，决定观测分辨率
    num_snakes: int = 4  # 同场蛇数量，会影响并行采样和竞争难度
    obs_channels: int = 10
    num_envs: int = 8
    num_food: int = 6
    max_steps: int = 1000
    total_frames: int = 500_000  # 总训练帧数
    warmup_frames: int = 10_000  # 增加预热帧数，收集更多初始经验
    batch_size: int = 128  # 增加批大小以提高训练效率
    gamma: float = 0.99  # 折扣因子
    multi_step: int = 5  # 增加n-step长度
    replay_capacity: int = 500_000  # 增加经验池容量
    lr: float = 1e-4  # 降低学习率以提高稳定性
    atom_size: int = 51  # C51 支撑原子数量
    v_min: float = -50.0  # 扩大分布范围
    v_max: float = 50.0  # 扩大分布范围
    epsilon_start: float = 1.0  # ε-greedy 起始值
    epsilon_final: float = 0.01  # 降低最终探索率
    epsilon_decay: int = 200_000  # 延长ε衰减时间
    update_target_interval: int = 2_000  # 降低目标网络同步频率
    train_interval: int = 4  # 每4帧训练一次
    log_interval: int = 1_000  # 打印训练日志的帧间隔
    save_interval: int = 25_000  # 滚动保存 latest checkpoint 的帧间隔
    checkpoint_name: str = "rainbow_snake_latest.pth"  # 默认模型名
    trainer_checkpoint_name: str = "rainbow_snake_latest_train.pth"  # 训练断点
    resume: bool = True  # 若训练断点存在则自动续训
    device: str = "auto"  # 运行设备
    
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
