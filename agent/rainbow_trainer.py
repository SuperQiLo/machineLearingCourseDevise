"""封装 Rainbow 训练循环（多环境并行版）。"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional

import numpy as np
import torch

from agent.rainbow_config import RainbowConfig
from agent.rainbow_agent import RainbowAgent
from agent.rainbow_replay_buffer import PrioritizedReplayBuffer
from agent.utils import resolve_device
from env.config import EnvConfig
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

        env_cfg = EnvConfig(
            width=config.grid_size,
            height=config.grid_size,
            num_snakes=config.num_snakes,
            num_food=config.num_food,
            max_steps=config.max_steps,
            step_penalty=config.step_penalty,
            food_reward=config.food_reward,
            death_penalty=config.death_penalty,
            kill_reward=config.kill_reward,
            distance_shaping_scale=config.distance_shaping_scale,
        )
        self.envs: List[MultiSnakeEnv] = [
            MultiSnakeEnv(config=env_cfg, seed=12345 + i) for i in range(config.num_envs)
        ]
        self.progress_cb = progress_cb
        self.best_score = -math.inf

        self._episode_returns: Deque[float] = deque(maxlen=200)

    def train(self) -> Path:
        """执行主训练循环，完成后返回最终模型路径。"""

        obs_envs = [env.reset() for env in self.envs]
        episode_scores = np.zeros((self.cfg.num_envs, self.cfg.num_snakes), dtype=np.float32)
        frame = 0  # 以“环境步数”计数（每个 env 的一步算 1）
        episode = 0
        epsilon = self.cfg.epsilon_start
        last_loss = 0.0

        if self.cfg.resume:
            loaded = self._try_load_training_checkpoint()
            if loaded is not None:
                frame, episode, epsilon, self.best_score, last_loss = loaded
                obs_envs = [env.reset() for env in self.envs]
                episode_scores.fill(0.0)

        while frame < self.cfg.total_frames:
            # 先为每个 env 生成动作，再 step，写 replay
            for e, env in enumerate(self.envs):
                alive_flags = [snake.get("alive", True) for snake in env.snakes]
                actions: List[int] = []
                for s, alive in enumerate(alive_flags):
                    if not alive:
                        actions.append(0)
                    else:
                        actions.append(self.agent.act(obs_envs[e][s], epsilon))

                next_obs, rewards, dones, info = env.step(actions)

                for s, alive in enumerate(alive_flags):
                    if not alive:
                        continue
                    self.replay.add(
                        obs_envs[e][s],
                        actions[s],
                        float(rewards[s]),
                        next_obs[s],
                        bool(dones[s]),
                    )
                    episode_scores[e, s] += float(rewards[s])

                obs_envs[e] = next_obs

                episode_done = bool(info.get("game_over")) or bool(np.all(np.asarray(dones, dtype=bool)))
                if episode_done:
                    self._episode_returns.append(float(np.mean(episode_scores[e])))
                    episode_scores[e].fill(0.0)
                    episode += 1
                    obs_envs[e] = env.reset()

            frame += self.cfg.num_envs
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
                last_loss = float(loss)
                self.replay.update_priorities(indices, td_errors)

            if frame % self.cfg.update_target_interval == 0:
                self.agent.sync_target()

            if frame % self.cfg.log_interval == 0 and self.progress_cb:
                avg_return = float(np.mean(self._episode_returns)) if self._episode_returns else 0.0
                alive_total = 0
                for env in self.envs:
                    alive_total += sum(1 for s in env.snakes if s.get("alive", False))
                alive_avg = alive_total / max(1, self.cfg.num_envs)
                self.progress_cb(
                    {
                        "frame": float(frame),
                        "episode": float(episode),
                        "avg_score": avg_return,
                        "loss": float(last_loss),
                        "epsilon": float(epsilon),
                        "alive": float(alive_avg),
                    }
                )

            if frame % self.cfg.save_interval == 0:
                self.agent.save(self.cfg.checkpoint_path())
                self._save_training_checkpoint(frame, episode, epsilon, last_loss)

        final_path = self._final_path()
        self.agent.save(final_path)
        self._save_training_checkpoint(frame, episode, epsilon, last_loss)
        return final_path

    def _prefix(self) -> str:
        name = Path(self.cfg.checkpoint_name).name
        stem = Path(name).stem
        if stem.endswith("_latest"):
            stem = stem[: -len("_latest")]
        return stem

    def _final_path(self) -> Path:
        return self.cfg.checkpoint_path().with_name(f"{self._prefix()}_final.pth")

    def _save_training_checkpoint(self, frame: int, episode: int, epsilon: float, last_loss: float) -> None:
        path = self.cfg.trainer_checkpoint_path()
        payload = {
            "frame": int(frame),
            "episode": int(episode),
            "epsilon": float(epsilon),
            "best_score": float(self.best_score),
            "last_loss": float(last_loss),
            "online_state": self.agent.online_net.state_dict(),
            "target_state": self.agent.target_net.state_dict(),
            "optimizer_state": self.agent.optimizer.state_dict(),
            "replay_state": self.replay.state_dict(),
        }
        torch.save(payload, path)

    def _try_load_training_checkpoint(self) -> Optional[tuple[int, int, float, float, float]]:
        path = self.cfg.trainer_checkpoint_path()
        if not path.exists():
            return None

        kwargs = {"map_location": self.device}
        try:
            payload = torch.load(path, weights_only=False, **kwargs)  # type: ignore[arg-type]
        except TypeError:
            payload = torch.load(path, **kwargs)

        if not isinstance(payload, dict):
            return None

        self.agent.online_net.load_state_dict(payload.get("online_state", {}))
        self.agent.target_net.load_state_dict(payload.get("target_state", {}))
        opt_state = payload.get("optimizer_state")
        if opt_state:
            self.agent.optimizer.load_state_dict(opt_state)
        replay_state = payload.get("replay_state")
        if isinstance(replay_state, dict):
            self.replay.load_state_dict(replay_state)

        frame = int(payload.get("frame", 0))
        episode = int(payload.get("episode", 1))
        epsilon = float(payload.get("epsilon", self.cfg.epsilon_start))
        best_score = float(payload.get("best_score", self.best_score))
        last_loss = float(payload.get("last_loss", 0.0))
        return frame, episode, epsilon, best_score, last_loss

        
