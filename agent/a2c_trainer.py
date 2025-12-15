"""封装 A2C 训练循环，便于脚本与服务端复用。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from agent.a2c_agent import A2CAgent, A2CRolloutBatch
from agent.a2c_config import A2CConfig
from agent.utils import resolve_device
from env.multi_snake_env import MultiSnakeEnv

ProgressCallback = Callable[[Dict[str, float]], None]


class A2CTrainer:
    """负责环境交互、rollout 采样以及 A2C 更新调度。"""

    def __init__(self, config: A2CConfig, progress_cb: Optional[ProgressCallback] = None) -> None:
        self.cfg = config
        self.device = resolve_device(config.device)
        self.agent = A2CAgent(config, device=self.device)
        self.env = MultiSnakeEnv(
            width=config.grid_size,
            height=config.grid_size,
            num_snakes=config.num_snakes,
            num_food=max(4, config.grid_size // 3),
        )
        self.progress_cb = progress_cb
        self.best_score = -math.inf

    def train(self) -> Path:
        observations = self.env.reset()
        episode_scores = np.zeros(self.cfg.num_snakes, dtype=np.float32)
        frame = 0
        episode = 1
        last_loss = 0.0

        if self.cfg.resume:
            loaded = self._try_load_training_checkpoint()
            if loaded is not None:
                frame, episode, self.best_score, last_loss = loaded
                observations = self.env.reset()
                episode_scores.fill(0.0)

        # A2C：训练模式默认按策略采样动作
        self.agent.net.train()

        while frame < self.cfg.total_frames:
            batch, observations, episode_done = self._collect_rollout(observations)

            loss, _ = self.agent.train_step(batch)
            last_loss = float(loss)
            frame += self.cfg.rollout_length

            # 更新 episode 统计（按 rollout 内的 reward 汇总）
            episode_scores += batch.rewards.sum(axis=0)

            alive_count = sum(1 for s in self.env.snakes if s.get("alive", False))
            game_over = alive_count <= 1 or getattr(self.env, "steps", 0) >= getattr(self.env, "max_steps", 0)

            if self.progress_cb and frame % self.cfg.log_interval == 0:
                self.progress_cb(
                    {
                        "frame": float(frame),
                        "episode": float(episode),
                        "avg_score": float(np.mean(episode_scores)),
                        "loss": float(last_loss),
                        "alive": float(alive_count),
                    }
                )

            if frame % self.cfg.save_interval == 0:
                self.agent.save(self.cfg.checkpoint_path())
                self._save_training_checkpoint(frame, episode, last_loss)

            if episode_done:
                avg_score = float(np.mean(episode_scores))
                if avg_score > self.best_score:
                    self.best_score = avg_score
                    self.agent.save(self.cfg.checkpoint_path().with_name("a2c_snake_best.pth"))
                if self.progress_cb:
                    self.progress_cb(
                        {
                            "frame": float(frame),
                            "episode": float(episode),
                            "avg_score": avg_score,
                            "loss": float(last_loss),
                            "alive": float(alive_count),
                            "steps": float(getattr(self.env, "steps", 0)),
                        }
                    )
                observations = self.env.reset()
                episode_scores.fill(0.0)
                episode += 1

        final_path = self.cfg.checkpoint_path().with_name("a2c_snake_final.pth")
        self.agent.save(final_path)
        self._save_training_checkpoint(frame, episode, last_loss)
        return final_path

    def _save_training_checkpoint(self, frame: int, episode: int, last_loss: float) -> None:
        path = self.cfg.trainer_checkpoint_path()
        payload = {
            "frame": int(frame),
            "episode": int(episode),
            "best_score": float(self.best_score),
            "last_loss": float(last_loss),
            "net_state": self.agent.net.state_dict(),
            "optimizer_state": self.agent.optimizer.state_dict(),
        }
        torch.save(payload, path)

    def _try_load_training_checkpoint(self) -> Optional[tuple[int, int, float, float]]:
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

        self.agent.net.load_state_dict(payload.get("net_state", {}))
        opt_state = payload.get("optimizer_state")
        if opt_state:
            self.agent.optimizer.load_state_dict(opt_state)

        frame = int(payload.get("frame", 0))
        episode = int(payload.get("episode", 1))
        best_score = float(payload.get("best_score", self.best_score))
        last_loss = float(payload.get("last_loss", 0.0))
        return frame, episode, best_score, last_loss

    def _alive_to_done(self) -> List[bool]:
        return [not snake.get("alive", False) for snake in self.env.snakes]

    def _collect_rollout(self, observations: List[np.ndarray]) -> Tuple[A2CRolloutBatch, List[np.ndarray], bool]:
        T = self.cfg.rollout_length
        N = self.cfg.num_snakes

        states = np.zeros((T, N, 3, self.cfg.grid_size, self.cfg.grid_size), dtype=np.float32)
        actions = np.zeros((T, N), dtype=np.int64)
        rewards = np.zeros((T, N), dtype=np.float32)
        dones = np.zeros((T, N), dtype=np.float32)
        values = np.zeros((T, N), dtype=np.float32)
        log_probs = np.zeros((T, N), dtype=np.float32)
        masks = np.zeros((T, N), dtype=np.float32)

        episode_done = False
        last_done_flags = np.zeros((N,), dtype=np.float32)

        for t in range(T):
            alive_flags = [snake.get("alive", True) for snake in self.env.snakes]

            # 记录状态，并对存活蛇采样动作
            for i in range(N):
                if i < len(observations):
                    states[t, i] = observations[i]
                if not alive_flags[i]:
                    actions[t, i] = 0
                    values[t, i] = 0.0
                    log_probs[t, i] = 0.0
                    masks[t, i] = 0.0
                    continue

                masks[t, i] = 1.0

                obs_t = observations[i]
                x = np.expand_dims(obs_t, axis=0)
                x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    logits, v = self.agent.net(x_t)
                    dist = torch.distributions.Categorical(logits=logits)
                    a = dist.sample()
                    lp = dist.log_prob(a)

                actions[t, i] = int(a.item())
                values[t, i] = float(v.item())
                log_probs[t, i] = float(lp.item())

            next_obs, _, step_dones, info = self.env.step(actions[t].tolist())
            shaped = self._derive_rewards(info.get("events"))

            rewards[t] = np.asarray(shaped, dtype=np.float32)
            dead_flags = np.asarray([not s.get("alive", False) for s in self.env.snakes], dtype=np.float32)
            done_flags = np.maximum(np.asarray(step_dones, dtype=np.float32), dead_flags)
            dones[t] = done_flags
            last_done_flags = done_flags

            observations = next_obs
            episode_done = all(done_flags.astype(bool)) or bool(info.get("game_over")) or info.get("alive_count", 0) <= 1
            if episode_done:
                break

        last_state = np.stack(observations, axis=0).astype(np.float32)
        last_done = last_done_flags.astype(np.float32)

        return (
            A2CRolloutBatch(
            states=states,
            actions=actions,
            rewards=rewards,
            dones=dones,
            values=values,
            log_probs=log_probs,
            masks=masks,
            last_state=last_state,
            last_done=last_done,
            ),
            observations,
            bool(episode_done),
        )

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
