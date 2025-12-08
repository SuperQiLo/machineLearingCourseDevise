"""带有 n-step 功能的优先经验回放实现。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class Transition:
    """单条 n-step 经验记录，用于写入经验池。"""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class TransitionBatch:
    """按批次打包的 Transition，方便一次性转成张量。"""

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray


class PrioritizedReplayBuffer:
    """支持 n-step 回报和重要性采样的优先经验池。"""

    def __init__(
        self,
        capacity: int,
        multi_step: int,
        gamma: float,
        *,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 200_000,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / max(1, beta_frames)
        self.multi_step = multi_step
        self.gamma = gamma

        self.buffer: List[Transition] = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.n_step_buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=multi_step)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """写入一条原始 transition，内部会自动累计 n-step。"""

        transition = (state, action, reward, next_state, done)
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) == self.multi_step:
            reward_sum, next_state_n, done_n = self._collect_n_step()
            state_0, action_0 = self.n_step_buffer[0][:2]
            self._store_transition(state_0, action_0, reward_sum, next_state_n, done_n)
            self.n_step_buffer.popleft()

        if done:
            while self.n_step_buffer:
                reward_sum, next_state_n, done_n = self._collect_n_step()
                state_0, action_0 = self.n_step_buffer[0][:2]
                self._store_transition(state_0, action_0, reward_sum, next_state_n, done_n)
                self.n_step_buffer.popleft()

    def sample(self, batch_size: int) -> Tuple[np.ndarray, TransitionBatch, torch.Tensor]:
        """按优先级采样一批样本，并返回重要性权重。"""

        if len(self.buffer) == 0:
            raise ValueError("空的经验池无法采样")

        priorities = self.priorities if len(self.buffer) == self.capacity else self.priorities[: self.pos]
        scaled = priorities**self.alpha
        probs = scaled / scaled.sum()
        indices = np.random.choice(len(priorities), batch_size, p=probs)

        samples = [self.buffer[idx] for idx in indices]
        batch = TransitionBatch(
            states=np.stack([sample.state for sample in samples]).astype(np.float32),
            actions=np.array([sample.action for sample in samples], dtype=np.int64),
            rewards=np.array([sample.reward for sample in samples], dtype=np.float32),
            next_states=np.stack([sample.next_state for sample in samples]).astype(np.float32),
            dones=np.array([sample.done for sample in samples], dtype=np.float32),
        )

        total = len(priorities)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)

        return indices, batch, torch.as_tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices: Sequence[int], priorities: torch.Tensor, eps: float = 1e-3) -> None:
        """训练完毕后回写 TD-Error，用于提升下一次采样质量。"""

        values = priorities.detach().cpu().numpy() + eps
        for idx, priority in zip(indices, values):
            self.priorities[idx] = priority

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    def _collect_n_step(self) -> Tuple[float, np.ndarray, bool]:
        """将当前 n-step 队列折算为单条 Transition。"""

        reward, next_state, done = self.n_step_buffer[-1][2:]
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_state, step_done = transition[2], transition[3], transition[4]
            reward = r + self.gamma * reward * (1 - step_done)
            next_state = n_state if step_done else next_state
            done = step_done or done
        return reward, next_state, done

    def _store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """写入经验池，并维护优先级数组与游标。"""

        transition = Transition(state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        current_max = self.priorities.max() if self.buffer else 1.0
        self.priorities[self.pos] = max(current_max, 1.0)
        self.pos = (self.pos + 1) % self.capacity
