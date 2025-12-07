"""PPO 算法实现。"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent.model import ActorCritic


class PPOAgent:
    """共享策略的 PPO 智能体。"""

    def __init__(
        self,
        input_channels: int = 3,
        grid_size: int = 20,
        action_dim: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        K_epochs: int = 4,
        device: torch.device | None = None,
    ) -> None:
        """初始化 PPO 智能体，构建策略/价值网络与优化器。"""
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = ActorCritic(input_channels, grid_size, action_dim).to(self.device)
        self.policy_old = ActorCritic(input_channels, grid_size, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()

    # ------------------------------------------------------------------
    # 交互接口
    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray) -> tuple[int, float]:
        """基于旧策略采样动作，返回动作与 log_prob。"""

        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            action, log_prob = self.policy_old.act(state_tensor)
        return action, float(log_prob.item())

    def predict(self, state: np.ndarray) -> int:
        """部署阶段使用贪心动作。"""

        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            probs, _ = self.policy(state_tensor)
            action = torch.argmax(probs, dim=-1)
        return int(action.item())

    # ------------------------------------------------------------------
    # 更新逻辑
    # ------------------------------------------------------------------
    def update(self, memory: "Memory") -> None:
        """使用采样到的记忆批次执行若干轮 PPO 更新。"""
        if not memory.states:
            return

        rewards = self._compute_discounted_rewards(memory.rewards, memory.is_terminals)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        states = torch.as_tensor(np.stack(memory.states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(memory.actions, dtype=torch.long, device=self.device)
        old_logprobs = torch.as_tensor(memory.logprobs, dtype=torch.float32, device=self.device)

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(states, actions)
            ratios = torch.exp(logprobs - old_logprobs)

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.mse_loss(state_values, rewards)
                - 0.01 * dist_entropy
            )

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

    def _compute_discounted_rewards(self, rewards: List[float], terminals: List[bool]) -> torch.Tensor:
        """按 episode 终止重置折扣回报，用于 advantage 计算。"""
        discounted_rewards: List[float] = []
        discounted_sum = 0.0
        for reward, done in zip(reversed(rewards), reversed(terminals)):
            if done:
                discounted_sum = 0.0
            discounted_sum = reward + self.gamma * discounted_sum
            discounted_rewards.insert(0, discounted_sum)
        return torch.as_tensor(discounted_rewards, dtype=torch.float32, device=self.device)


class Memory:
    """存放一段轨迹数据的工具类。"""

    def __init__(self) -> None:
        """初始化空的动作、状态、概率与奖励缓存。"""
        self.actions: List[int] = []
        self.states: List[np.ndarray] = []
        self.logprobs: List[float] = []
        self.rewards: List[float] = []
        self.is_terminals: List[bool] = []

    def add(self, state: np.ndarray, action: int, logprob: float, reward: float, done: bool) -> None:
        """将单步交互结果压入记忆。"""
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.is_terminals.append(done)

    def clear(self) -> None:
        """在完成一次 PPO 更新后清空缓存。"""
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()
