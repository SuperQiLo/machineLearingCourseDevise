"""A2C 推理/训练用的智能体实现 (增强版)。"""

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
    """共享策略的 A2C 智能体（Actor-Critic），支持 GAE。"""

    def __init__(self, config: A2CConfig, device: Optional[torch.device] = None) -> None:
        self.cfg = config
        self.device = device or resolve_device(config.device)
        self.action_dim = 3

        self.net = A2CBackbone(
            input_channels=int(getattr(config, "obs_channels", 10)),
            grid_size=config.grid_size,
            action_dim=self.action_dim,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.net.parameters(), 
            lr=config.lr, 
            eps=1e-5,
            weight_decay=1e-4  # L2正则化
        )
        
        # 学习率调度器
        if config.lr_decay:
            updates = max(1, int(config.total_frames // (config.rollout_length * max(1, getattr(config, "num_envs", 1)))))
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=updates,
                eta_min=config.lr_min
            )
        else:
            self.scheduler = None

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
    # GAE 计算
    # ------------------------------------------------------------------
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算 Generalized Advantage Estimation。
        
        Args:
            rewards: (T, N) 奖励
            values: (T, N) 价值估计
            dones: (T, N) 结束标志
            last_value: (N,) 最后状态的价值
            gamma: 折扣因子
            gae_lambda: GAE lambda参数
            
        Returns:
            advantages: (T, N) GAE优势
            returns: (T, N) 目标回报
        """
        T, N = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(N, device=rewards.device)
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    # 训练相关
    # ------------------------------------------------------------------
    def train_step(self, batch: Any, weights: Any = None) -> Tuple[float, Optional[np.ndarray]]:
        """对一个 rollout 做一次 A2C 更新，使用 GAE。"""

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
            last_value = last_value * (1.0 - last_done)

            if self.cfg.use_gae:
                # 使用GAE计算优势
                advantages, returns = self.compute_gae(
                    rewards, values_pred.detach(), dones, last_value,
                    gamma, self.cfg.gae_lambda
                )
            else:
                # 传统的n-step回报
                returns = torch.zeros((T, N), dtype=torch.float32, device=device)
                running = last_value
                for t in range(T - 1, -1, -1):
                    running = rewards[t] + gamma * running * (1.0 - dones[t])
                    returns[t] = running
                advantages = returns - values_pred.detach()

            # 标准化优势 - 减少方差
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        valid = masks.view(-1) > 0.0
        if valid.sum().item() == 0:
            return 0.0, None

        flat_log_probs = log_probs.view(-1)[valid]
        flat_adv = advantages.view(-1)[valid]
        flat_entropy = entropy.view(-1)[valid]
        flat_values = values_pred.view(-1)[valid]
        flat_returns = returns.view(-1)[valid]

        policy_loss = -(flat_log_probs * flat_adv).mean()
        value_loss = F.smooth_l1_loss(flat_values, flat_returns)  # 使用Huber loss更稳定
        entropy_loss = -flat_entropy.mean()

        loss = policy_loss + self.cfg.value_coef * value_loss + self.cfg.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()
        
        # 更新学习率
        if self.scheduler is not None:
            self.scheduler.step()

        return float(loss.item()), None

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """可选的评估接口：返回 (log_probs, entropy, values)。"""

        logits, values = self.net(states)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values
