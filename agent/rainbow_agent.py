"""Rainbow DQN 推理/训练用的智能体实现。"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from agent.base_agent import BaseAgent
from agent.rainbow_config import RainbowConfig
from agent.rainbow_model import RainbowBackbone
from agent.replay_buffer import TransitionBatch
from agent.utils import resolve_device


class RainbowAgent(BaseAgent):
    """包含在线/目标网络的 Rainbow 智能体，既可训练也可用于推理。"""

    def __init__(self, config: RainbowConfig, device: Optional[torch.device] = None) -> None:
        """初始化在线/目标网络、优化器等核心组件。"""

        self.cfg = config
        self.device = device or resolve_device(config.device)
        self.action_dim = 3

        self.online_net = RainbowBackbone(
            input_channels=3,
            grid_size=config.grid_size,
            action_dim=self.action_dim,
            atom_size=config.atom_size,
        ).to(self.device)
        self.target_net = RainbowBackbone(
            input_channels=3,
            grid_size=config.grid_size,
            action_dim=self.action_dim,
            atom_size=config.atom_size,
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=config.lr, eps=1.5e-4)
        self.support = torch.linspace(config.v_min, config.v_max, config.atom_size, device=self.device)
        self.delta_z = (config.v_max - config.v_min) / (config.atom_size - 1)

    # ------------------------------------------------------------------
    # 推理接口
    # ------------------------------------------------------------------
    def act(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        """以 eps-greedy 策略选择动作，既可用于训练也可部署。"""

        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)

        state = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            advantage, value = self.online_net(state)
            dist = self._logits_to_dist(advantage, value)
            q_values = torch.sum(dist * self.support, dim=-1)
            action = torch.argmax(q_values, dim=1)
        return int(action.item())

    def load(self, checkpoint: Path) -> None:
        """从给定路径加载参数到在线/目标网络。"""

        kwargs = {"map_location": self.device}
        try:
            state_dict = torch.load(checkpoint, weights_only=True, **kwargs)  # type: ignore[arg-type]
        except TypeError:
            state_dict = torch.load(checkpoint, **kwargs)
        self.online_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)

    def save(self, checkpoint: Path) -> None:
        """将当前在线网络的参数保存至指定路径。"""

        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.online_net.state_dict(), checkpoint)

    # ------------------------------------------------------------------
    # 训练相关
    # ------------------------------------------------------------------
    def train_step(
        self,
        batch: Any,
        weights: Any = None,
    ) -> Tuple[float, Optional[np.ndarray]]:
        """执行一次分布式 Bellman 更新，返回 loss 与 TD 误差估计。"""

        device = self.device
        # Ensure batch is TransitionBatch
        if not isinstance(batch, TransitionBatch):
            raise TypeError("batch must be a TransitionBatch")
        
        start_states = torch.as_tensor(batch.states, dtype=torch.float32, device=device)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=device).unsqueeze(-1)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=device)
        next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=device)
        
        # Handle weights
        if weights is None:
            weights_t = torch.ones(len(batch.states), device=device)
        elif isinstance(weights, torch.Tensor):
            weights_t = weights.to(device)
        else:
            weights_t = torch.as_tensor(weights, dtype=torch.float32, device=device)

        self.online_net.reset_noise()
        self.target_net.reset_noise()

        # 当前动作对应的分布
        advantage, value = self.online_net(start_states)
        dist = self._logits_to_dist(advantage, value)
        action_dist = dist.gather(1, actions.unsqueeze(-1).expand(-1, 1, self.cfg.atom_size)).squeeze(1)
        action_dist = torch.clamp(action_dist, min=1e-6)

        with torch.no_grad():
            next_advantage, next_value = self.online_net(next_states)
            next_dist = self._logits_to_dist(next_advantage, next_value)
            next_q = torch.sum(next_dist * self.support, dim=-1)
            next_actions = next_q.argmax(dim=1, keepdim=True)

            target_advantage, target_value = self.target_net(next_states)
            target_dist_full = self._logits_to_dist(target_advantage, target_value)
            target_dist = target_dist_full.gather(
                1,
                next_actions.unsqueeze(-1).expand(-1, 1, self.cfg.atom_size),
            ).squeeze(1)

            t_z = rewards.unsqueeze(1) + (self.cfg.gamma**self.cfg.multi_step) * (1.0 - dones.unsqueeze(1)) * self.support
            t_z = t_z.clamp(self.cfg.v_min, self.cfg.v_max)
            b = (t_z - self.cfg.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            l = l.clamp(0, self.cfg.atom_size - 1)
            u = u.clamp(0, self.cfg.atom_size - 1)

            proj_dist = torch.zeros_like(target_dist)
            batch_range = torch.arange(target_dist.size(0), device=device).unsqueeze(1) * self.cfg.atom_size
            proj_dist.view(-1).index_add_(0, (l + batch_range).view(-1), (target_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + batch_range).view(-1), (target_dist * (b - l.float())).view(-1))

        loss = -torch.sum(proj_dist * torch.log(action_dist), dim=1)
        loss = (loss * weights_t).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        td_errors = torch.abs(torch.sum((proj_dist - action_dist), dim=1)).detach()

        return loss.item(), td_errors

    def sync_target(self) -> None:
        """将在线网络的权重复制到目标网络。"""

        self.target_net.load_state_dict(self.online_net.state_dict())

    def _logits_to_dist(self, advantage: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """根据 Duelling 结构组合 Value/Advantage 并归一化为分布。"""

        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        return F.softmax(q_atoms, dim=-1)