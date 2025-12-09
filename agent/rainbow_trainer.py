"""封装 Rainbow 训练循环，便于脚本与服务端复用。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from agent.rainbow_config import RainbowConfig
from agent.rainbow_agent import RainbowAgent
from agent.replay_buffer import PrioritizedReplayBuffer
from agent.utils import resolve_device
from env.multi_snake_env import MultiSnakeEnv

ProgressCallback = Callable[[Dict[str, float]], None]


class RainbowTrainer:
    """负责环境交互、经验写入以及训练调度。"""

    def __init__(self, config: RainbowConfig, progress_cb: Optional[ProgressCallback] = None) -> None:
        """根据配置初始化环境、经验池与智能体。"""

        self.cfg = config
        self.device = resolve_device(config.device)
        self.agent = RainbowAgent(config, device=self.device)
        self.replay = PrioritizedReplayBuffer(
            capacity=config.replay_capacity,
            multi_step=config.multi_step,
            gamma=config.gamma,
        )
        self.env = MultiSnakeEnv(
            width=config.grid_size,
            height=config.grid_size,
            num_snakes=config.num_snakes,
            num_food=max(4, config.grid_size // 3),
        )
        self.progress_cb = progress_cb
        self.best_score = -math.inf

    def train(self) -> Path:
        """执行主训练循环，完成后返回最终模型路径。"""

        observations = self.env.reset()
        episode_scores = np.zeros(self.cfg.num_snakes, dtype=np.float32)
        frame = 0
        episode = 1
        epsilon = self.cfg.epsilon_start

        while frame < self.cfg.total_frames:
            alive_flags = [snake["alive"] for snake in self.env.snakes]
            actions: List[int] = []
            for idx, alive in enumerate(alive_flags):
                if not alive:
                    actions.append(0)
                    continue
                actions.append(self.agent.act(observations[idx], epsilon))

            next_obs, _, dones, info = self.env.step(actions)
            rewards = self._derive_rewards(info.get("events"))

            for idx, alive in enumerate(alive_flags):
                if not alive:
                    continue
                self.replay.add(observations[idx], actions[idx], rewards[idx], next_obs[idx], dones[idx])
                episode_scores[idx] += rewards[idx]
            observations = next_obs
            frame += 1
            epsilon = max(
                self.cfg.epsilon_final,
                self.cfg.epsilon_start
                - (self.cfg.epsilon_start - self.cfg.epsilon_final) * frame / self.cfg.epsilon_decay,
            )

            if (
                frame > self.cfg.warmup_frames
                and len(self.replay) >= self.cfg.batch_size
                and frame % self.cfg.train_interval == 0
            ):
                indices, batch, weights = self.replay.sample(self.cfg.batch_size)
                loss, td_errors = self.agent.train_step(batch, weights)
                self.replay.update_priorities(indices, td_errors)

            if frame % self.cfg.update_target_interval == 0:
                self.agent.sync_target()

            if frame % self.cfg.log_interval == 0 and self.progress_cb:
                self.progress_cb(
                    {
                        "frame": frame,
                        "episode": episode,
                        "avg_score": float(np.mean(episode_scores)),
                        "epsilon": epsilon,
                        "alive": info.get("alive_count", self.cfg.num_snakes),
                    }
                )

            if frame % self.cfg.save_interval == 0:
                self.agent.save(self.cfg.checkpoint_path())

            episode_done = all(dones) or info.get("game_over") or info.get("alive_count", 0) <= 0
            if episode_done:
                avg_score = float(np.mean(episode_scores))
                if avg_score > self.best_score:
                    self.best_score = avg_score
                    self.agent.save(self.cfg.checkpoint_path().with_name("rainbow_snake_best.pth"))
                if self.progress_cb:
                    self.progress_cb(
                        {
                            "frame": frame,
                            "episode": episode,
                            "avg_score": avg_score,
                            "epsilon": epsilon,
                            "alive": info.get("alive_count", self.cfg.num_snakes),
                            "steps": info.get("steps", 0),
                        }
                    )
                observations = self.env.reset()
                episode_scores.fill(0.0)
                episode += 1

        final_path = self.cfg.checkpoint_path().with_name("rainbow_snake_final.pth")
        self.agent.save(final_path)
        return final_path

    def _derive_rewards(self, events: Optional[List[Dict]]) -> List[float]:
        rewards = [0.0 for _ in range(self.cfg.num_snakes)]
        if not events:
            return rewards
        for idx, event in enumerate(events):
            if idx >= len(rewards):
                break
            value = 0.0
            if event.get("alive"):
                value += self.cfg.reward_survive
            if event.get("ate_food"):
                value += self.cfg.reward_food
            kills = int(event.get("kills", 0) or 0)
            if kills:
                value += self.cfg.reward_kill * kills
            if event.get("died"):
                value += self.cfg.reward_death
            rewards[idx] = value
        return rewards
