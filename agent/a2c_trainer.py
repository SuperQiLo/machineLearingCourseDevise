"""封装 A2C 训练循环（多环境并行版）。"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

from agent.a2c_agent import A2CAgent, A2CRolloutBatch
from agent.a2c_config import A2CConfig
from agent.utils import resolve_device
from env.config import EnvConfig
from env.multi_snake_env import MultiSnakeEnv

ProgressCallback = Callable[[Dict[str, float]], None]


class A2CTrainer:
    def __init__(self, config: A2CConfig, progress_cb: Optional[ProgressCallback] = None) -> None:
        self.cfg = config
        self.device = resolve_device(config.device)
        self.agent = A2CAgent(config, device=self.device)

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
        obs_envs = [env.reset() for env in self.envs]
        frame = 0
        episode = 0
        last_loss = 0.0

        if self.cfg.resume:
            loaded = self._try_load_training_checkpoint()
            if loaded is not None:
                frame, episode, self.best_score, last_loss = loaded
                obs_envs = [env.reset() for env in self.envs]

        self.agent.net.train()

        while frame < self.cfg.total_frames:
            batch, obs_envs, finished_eps = self._collect_rollout(obs_envs)
            loss, _ = self.agent.train_step(batch)
            last_loss = float(loss)

            # 以“环境步数”计数：每个 env 的一步算 1
            frame += self.cfg.rollout_length * self.cfg.num_envs
            episode += finished_eps

            if self.progress_cb and frame % self.cfg.log_interval == 0:
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
                        "alive": float(alive_avg),
                    }
                )

            if frame % self.cfg.save_interval == 0:
                self.agent.save(self.cfg.checkpoint_path())
                self._save_training_checkpoint(frame, episode, last_loss)

            avg_return = float(np.mean(self._episode_returns)) if self._episode_returns else -math.inf
            if avg_return > self.best_score:
                self.best_score = avg_return
                self.agent.save(self._best_path())

        final_path = self._final_path()
        self.agent.save(final_path)
        self._save_training_checkpoint(frame, episode, last_loss)
        return final_path

    def _prefix(self) -> str:
        name = Path(self.cfg.checkpoint_name).name
        stem = Path(name).stem
        if stem.endswith("_latest"):
            stem = stem[: -len("_latest")]
        return stem

    def _best_path(self) -> Path:
        return self.cfg.checkpoint_path().with_name(f"{self._prefix()}_best.pth")

    def _final_path(self) -> Path:
        return self.cfg.checkpoint_path().with_name(f"{self._prefix()}_final.pth")

    def _save_training_checkpoint(self, frame: int, episode: int, last_loss: float) -> None:
        path = self.cfg.trainer_checkpoint_path()
        payload = {
            "frame": int(frame),
            "episode": int(episode),
            "best_score": float(self.best_score),
            "last_loss": float(last_loss),
            "net_state": self.agent.net.state_dict(),
            "optimizer_state": self.agent.optimizer.state_dict(),
            "scheduler_state": self.agent.scheduler.state_dict() if self.agent.scheduler is not None else None,
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
        sch_state = payload.get("scheduler_state")
        if self.agent.scheduler is not None and sch_state:
            self.agent.scheduler.load_state_dict(sch_state)

        frame = int(payload.get("frame", 0))
        episode = int(payload.get("episode", 0))
        best_score = float(payload.get("best_score", self.best_score))
        last_loss = float(payload.get("last_loss", 0.0))
        return frame, episode, best_score, last_loss

    def _collect_rollout(
        self, obs_envs: List[List[np.ndarray]]
    ) -> Tuple[A2CRolloutBatch, List[List[np.ndarray]], int]:
        T = self.cfg.rollout_length
        E = self.cfg.num_envs
        S = self.cfg.num_snakes
        B = E * S
        C = int(getattr(self.cfg, "obs_channels", 10))
        G = self.cfg.grid_size

        states = np.zeros((T, B, C, G, G), dtype=np.float32)
        actions = np.zeros((T, B), dtype=np.int64)
        rewards = np.zeros((T, B), dtype=np.float32)
        dones = np.zeros((T, B), dtype=np.float32)
        values = np.zeros((T, B), dtype=np.float32)
        log_probs = np.zeros((T, B), dtype=np.float32)
        masks = np.zeros((T, B), dtype=np.float32)

        ep_returns = np.zeros((E, S), dtype=np.float32)
        finished_eps = 0

        for t in range(T):
            flat_obs: List[np.ndarray] = []
            alive_mask: List[bool] = []
            for e, env in enumerate(self.envs):
                for s in range(S):
                    o = obs_envs[e][s]
                    flat_obs.append(o)
                    alive = bool(env.snakes[s].get("alive", True))
                    alive_mask.append(alive)

            states[t] = np.stack(flat_obs, axis=0)

            # 采样动作（仅对存活个体）
            x = torch.as_tensor(states[t], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                logits, v = self.agent.net(x)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                lp = dist.log_prob(a)

            a_np = a.detach().cpu().numpy().astype(np.int64)
            v_np = v.detach().cpu().numpy().astype(np.float32).reshape(-1)
            lp_np = lp.detach().cpu().numpy().astype(np.float32)

            for i in range(B):
                if not alive_mask[i]:
                    a_np[i] = 0
                    v_np[i] = 0.0
                    lp_np[i] = 0.0
                    masks[t, i] = 0.0
                else:
                    masks[t, i] = 1.0

            actions[t] = a_np
            values[t] = v_np
            log_probs[t] = lp_np

            # 按 env 切片执行 step
            next_obs_envs: List[List[np.ndarray]] = []
            for e, env in enumerate(self.envs):
                a_slice = actions[t, e * S : (e + 1) * S].tolist()
                next_obs, r, d, info = env.step(a_slice)
                next_obs_envs.append(next_obs)

                r_arr = np.asarray(r, dtype=np.float32)
                d_arr = np.asarray(d, dtype=np.float32)
                rewards[t, e * S : (e + 1) * S] = r_arr
                dones[t, e * S : (e + 1) * S] = d_arr
                ep_returns[e] += r_arr

                if bool(info.get("game_over")) or bool(np.all(d_arr.astype(bool))):
                    # 记录一局的平均回报（按 snake 平均），并重置
                    self._episode_returns.append(float(np.mean(ep_returns[e])))
                    ep_returns[e].fill(0.0)
                    finished_eps += 1
                    next_obs_envs[e] = env.reset()

            obs_envs = next_obs_envs

        last_state = np.stack([o for env_obs in obs_envs for o in env_obs], axis=0).astype(np.float32)
        last_done = np.zeros((B,), dtype=np.float32)

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
            obs_envs,
            finished_eps,
        )
