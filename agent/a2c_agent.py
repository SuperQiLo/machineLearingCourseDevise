"""A2C 推理/训练用的智能体实现。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from agent.a2c_config import A2CConfig
from agent.a2c_model import A2CBackbone
from agent.base_agent import BaseAgent
from agent.utils import resolve_device


@dataclass
class A2CRolloutBatch:
    """按 rollout 打包的数据（T, N, ...），由 Trainer 构造。"""

    states: np.ndarray  # (T, N, C, W, H)
    actions: np.ndarray  # (T, N)
    rewards: np.ndarray  # (T, N)
    dones: np.ndarray  # (T, N)
    values: np.ndarray  # (T, N)
    log_probs: np.ndarray  # (T, N)
    masks: np.ndarray  # (T, N) 1 表示有效(当步 alive)
    last_state: np.ndarray  # (N, C, W, H)
    last_done: np.ndarray  # (N,)


class A2CAgent(BaseAgent):
    """共享策略的 A2C 智能体（Actor-Critic）。"""

    def __init__(self, config: A2CConfig, device: Optional[torch.device] = None) -> None:
        self.cfg = config
        self.device = device or resolve_device(config.device)
        self.action_dim = 3

        self.net = A2CBackbone(
            input_channels=3,
            grid_size=config.grid_size,
            action_dim=self.action_dim,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.lr, eps=1.5e-4)

    # ------------------------------------------------------------------
    # 推理接口
    # ------------------------------------------------------------------
    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """默认：训练模式下按策略采样；评估模式下取 argmax。"""

        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)

        x = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.net(x)
            if self.net.training:
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, num_samples=1)
                return int(action.item())
            return int(torch.argmax(logits, dim=-1).item())

    def load(self, path: Path) -> None:  # type: ignore[override]
        kwargs = {"map_location": self.device}
        try:
            state_dict = torch.load(path, weights_only=True, **kwargs)  # type: ignore[arg-type]
        except TypeError:
            state_dict = torch.load(path, **kwargs)
        self.net.load_state_dict(state_dict)

    def save(self, path: Path) -> None:  # type: ignore[override]
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)

    # ------------------------------------------------------------------
    # 训练相关
    # ------------------------------------------------------------------
    def train_step(self, batch: Any, weights: Any = None) -> Tuple[float, Optional[np.ndarray]]:
        """对一个 rollout 做一次 A2C 更新。"""

        if not isinstance(batch, A2CRolloutBatch):
            raise TypeError("batch must be an A2CRolloutBatch")

        device = self.device
        gamma = self.cfg.gamma

        states = torch.as_tensor(batch.states, dtype=torch.float32, device=device)  # (T,N,C,W,H)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=device)  # (T,N)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=device)  # (T,N)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=device)  # (T,N)
        masks = torch.as_tensor(batch.masks, dtype=torch.float32, device=device)  # (T,N)

        last_state = torch.as_tensor(batch.last_state, dtype=torch.float32, device=device)  # (N,C,W,H)
        last_done = torch.as_tensor(batch.last_done, dtype=torch.float32, device=device)  # (N,)

        T, N = states.shape[0], states.shape[1]

        flat_states = states.view(T * N, *states.shape[2:])
        logits, values_pred = self.net(flat_states)
        logits = logits.view(T, N, -1)
        values_pred = values_pred.view(T, N)

        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)  # (T,N)
        entropy = dist.entropy()  # (T,N)

        with torch.no_grad():
            _, last_value = self.net(last_state)
            bootstrap = last_value * (1.0 - last_done)

            returns = torch.zeros((T, N), dtype=torch.float32, device=device)
            running = bootstrap
            for t in range(T - 1, -1, -1):
                running = rewards[t] + gamma * running * (1.0 - dones[t])
                returns[t] = running

            advantages = returns - values_pred

        valid = masks.view(-1) > 0.0
        if valid.sum().item() == 0:
            return 0.0, None

        flat_log_probs = log_probs.view(-1)[valid]
        flat_adv = advantages.view(-1)[valid]
        flat_entropy = entropy.view(-1)[valid]
        flat_values = values_pred.view(-1)[valid]
        flat_returns = returns.view(-1)[valid]

        policy_loss = -(flat_log_probs * flat_adv.detach()).mean()
        value_loss = F.mse_loss(flat_values, flat_returns)
        entropy_loss = -flat_entropy.mean()

        loss = policy_loss + self.cfg.value_coef * value_loss + self.cfg.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        return float(loss.item()), None

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """可选的评估接口：返回 (log_probs, entropy, values)。"""

        logits, values = self.net(states)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values
