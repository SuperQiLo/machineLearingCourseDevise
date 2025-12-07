"""PPO 训练脚本。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from agent.ppo import Memory, PPOAgent
from agent.utils import resolve_device
from env.multi_snake_env import MultiSnakeEnv


@dataclass
class TrainConfig:
    """集中管理 PPO 训练所需的超参数。"""
    grid_size: int = 30
    num_snakes: int = 4
    max_episodes: int = 2000
    max_timesteps: int = 1200
    update_timestep: int = 2400
    log_interval: int = 10
    lr: float = 2e-4
    gamma: float = 0.99
    eps_clip: float = 0.2
    K_epochs: int = 4
    device: str = "auto"


def train(config: Optional[TrainConfig] = None) -> None:
    """根据课程设计跑通多蛇 PPO 训练。"""

    cfg = config or TrainConfig()
    device = resolve_device(cfg.device)
    print(f"[Train] 使用设备: {device}")

    save_dir = Path("agent/checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)

    env = MultiSnakeEnv(width=cfg.grid_size, height=cfg.grid_size, num_snakes=cfg.num_snakes)
    memory = Memory()
    ppo = PPOAgent(
        input_channels=3,
        grid_size=cfg.grid_size,
        action_dim=3,
        lr=cfg.lr,
        gamma=cfg.gamma,
        eps_clip=cfg.eps_clip,
        K_epochs=cfg.K_epochs,
        device=device,
    )

    timestep_counter = 0
    running_scores: List[float] = []

    for episode in range(1, cfg.max_episodes + 1):
        observations = env.reset()
        episode_scores = [0.0 for _ in range(cfg.num_snakes)]

        for _ in range(cfg.max_timesteps):
            alive_flags = [snake["alive"] for snake in env.snakes]
            actions: List[int] = []
            log_probs: List[float] = []

            for idx, alive in enumerate(alive_flags):
                if not alive:
                    actions.append(0)
                    log_probs.append(0.0)
                    continue
                action, log_prob = ppo.select_action(observations[idx])
                actions.append(action)
                log_probs.append(log_prob)

            next_observations, rewards, dones, info = env.step(actions)

            for idx, alive in enumerate(alive_flags):
                if not alive:
                    continue
                memory.add(
                    observations[idx],
                    actions[idx],
                    log_probs[idx],
                    rewards[idx],
                    dones[idx],
                )
                episode_scores[idx] += rewards[idx]

            observations = next_observations
            timestep_counter += 1

            if timestep_counter % cfg.update_timestep == 0:
                ppo.update(memory)
                memory.clear()
                timestep_counter = 0

            if all(dones):
                break

        running_scores.append(sum(episode_scores) / cfg.num_snakes)

        if episode % cfg.log_interval == 0:
            avg_score = float(np.mean(running_scores[-cfg.log_interval:]))
            print(
                f"Episode {episode:04d} | Avg Score: {avg_score:.2f} | Alive: {info['alive_count']} | Steps: {info['steps']}"
            )
            torch.save(ppo.policy.state_dict(), save_dir / "ppo_snake_latest.pth")

    torch.save(ppo.policy.state_dict(), save_dir / "ppo_snake_final.pth")


def prompt_train_config() -> TrainConfig:
    """从命令行输入生成训练配置。"""

    def _int(prompt: str, default: int) -> int:
        """读取整型输入，支持回车使用默认值。"""
        raw = input(f"{prompt} (默认 {default}): ").strip()
        return int(raw) if raw else default

    def _float(prompt: str, default: float) -> float:
        """读取浮点输入，支持回车使用默认值。"""
        raw = input(f"{prompt} (默认 {default}): ").strip()
        return float(raw) if raw else default

    return TrainConfig(
        grid_size=_int("网格大小 (建议 ≥24)", 30),
        num_snakes=_int("蛇数量", 4),
        max_episodes=_int("训练轮数", 2000),
        max_timesteps=_int("每轮最大步数", 1200),
        update_timestep=_int("PPO 更新步长", 2400),
        log_interval=_int("日志间隔", 10),
        lr=_float("学习率", 2e-4),
        gamma=_float("折扣因子", 0.99),
        eps_clip=_float("PPO clip", 0.2),
        K_epochs=_int("每次更新迭代次数", 4),
        device=input("训练设备 (auto/cpu/cuda:0，默认 auto): ").strip() or "auto",
    )


if __name__ == "__main__":
    train(prompt_train_config())
