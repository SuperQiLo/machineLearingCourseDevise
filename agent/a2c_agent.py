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
            input_channels=int(getattr(config, "obs_channels", 3)),
            obs_size=getattr(config, "obs_size", 84),
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

        # 注意：这里的 (T*N) 可能很大（例如 256*128=32768），如果一次性喂入 CNN，
        # cuDNN 可能为卷积选择需要超大 workspace 的算法而导致 OOM。
        # 因此将前向/反向拆成 micro-batch 累积梯度。

        # states 留在 CPU，按块搬运到 GPU，显著降低峰值显存
        states_cpu = torch.as_tensor(batch.states, dtype=torch.float32, device="cpu")  # (T,N,C,W,H)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=device)  # (T,N)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=device)  # (T,N)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=device)  # (T,N)
        masks = torch.as_tensor(batch.masks, dtype=torch.float32, device=device)  # (T,N)

        last_state = torch.as_tensor(batch.last_state, dtype=torch.float32, device=device)  # (N,C,W,H)
        last_done = torch.as_tensor(batch.last_done, dtype=torch.float32, device=device)  # (N,)

        T, N = int(states_cpu.shape[0]), int(states_cpu.shape[1])
        total = T * N
        flat_states_cpu = states_cpu.view(total, *states_cpu.shape[2:])

        def _initial_micro_bs() -> int:
            cfg_mb = int(getattr(self.cfg, "train_micro_batch_size", 0))
            if cfg_mb > 0:
                return min(cfg_mb, total)

            # auto：用可用显存给一个保守初值，再依赖 OOM 回退兜底
            # 这里不尝试精确建模激活/工作区，只做“合理起步 + 失败自动二分”。
            start = min(total, 2048)
            if device.type == "cuda" and torch.cuda.is_available():
                try:
                    free_bytes, _ = torch.cuda.mem_get_info(device)
                    free_gb = float(free_bytes) / (1024.0 ** 3)
                    # 经验：每 GB 给 ~256 个样本的起步上限，并做硬上限保护
                    start = min(total, max(start, int(free_gb * 256)))
                    start = min(start, 8192)
                except Exception:
                    pass
            return max(64, int(start))

        def _empty_cuda_cache() -> None:
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        def _compute_values_pred(micro_bs: int) -> torch.Tensor:
            values_pred_flat = torch.empty((total,), dtype=torch.float32, device=device)
            with torch.no_grad():
                for start in range(0, total, micro_bs):
                    end = min(total, start + micro_bs)
                    x = flat_states_cpu[start:end].to(device, non_blocking=False)
                    _, v = self.net(x)
                    values_pred_flat[start:end] = v
            return values_pred_flat.view(T, N)

        def _backward_update(
            micro_bs: int,
            advantages: torch.Tensor,
            returns: torch.Tensor,
            valid: torch.Tensor,
            valid_count: int,
        ) -> float:
            flat_actions = actions.view(-1)
            flat_adv = advantages.view(-1)
            flat_returns = returns.view(-1)

            self.optimizer.zero_grad(set_to_none=True)
            total_loss_value = 0.0
            denom = float(valid_count)

            for start in range(0, total, micro_bs):
                end = min(total, start + micro_bs)
                mb_valid = valid[start:end]
                if not bool(mb_valid.any().item()):
                    continue

                x = flat_states_cpu[start:end].to(device, non_blocking=False)
                logits_mb, values_mb = self.net(x)
                dist_mb = torch.distributions.Categorical(logits=logits_mb)

                act_mb = flat_actions[start:end]
                adv_mb = flat_adv[start:end]
                ret_mb = flat_returns[start:end]

                logp_mb = dist_mb.log_prob(act_mb)
                ent_mb = dist_mb.entropy()

                policy_term = -((logp_mb[mb_valid] * adv_mb[mb_valid]).sum() / denom)
                value_term = F.smooth_l1_loss(values_mb[mb_valid], ret_mb[mb_valid], reduction="sum") / denom
                entropy_term = -(ent_mb[mb_valid].sum() / denom)

                loss_mb = policy_term + self.cfg.value_coef * value_term + self.cfg.entropy_coef * entropy_term
                total_loss_value += float(loss_mb.detach().item())
                loss_mb.backward()

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            return float(total_loss_value)

        # -------------------- 自适应 micro-batch：OOM 时自动二分回退重试 --------------------
        micro_bs = _initial_micro_bs()
        last_oom: Exception | None = None
        cfg_mb_raw = int(getattr(self.cfg, "train_micro_batch_size", 0))

        for _ in range(8):
            try:
                values_pred = _compute_values_pred(micro_bs)

                with torch.no_grad():
                    _, last_value = self.net(last_state)
                    last_value = last_value * (1.0 - last_done)

                    if self.cfg.use_gae:
                        advantages, returns = self.compute_gae(
                            rewards, values_pred, dones, last_value, gamma, self.cfg.gae_lambda
                        )
                    else:
                        returns = torch.zeros((T, N), dtype=torch.float32, device=device)
                        running = last_value
                        for t in range(T - 1, -1, -1):
                            running = rewards[t] + gamma * running * (1.0 - dones[t])
                            returns[t] = running
                        advantages = returns - values_pred

                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                valid = masks.view(-1) > 0.0
                valid_count = int(valid.sum().item())
                if valid_count == 0:
                    return 0.0, None

                loss_value = _backward_update(micro_bs, advantages, returns, valid, valid_count)
                mode = "auto" if cfg_mb_raw <= 0 else "fixed"
                print(f"[A2C] micro_bs({mode}) = {micro_bs}", flush=True)
                return float(loss_value), None

            except torch.OutOfMemoryError as e:
                last_oom = e
                _empty_cuda_cache()
                # 二分回退；确保最终至少为 1
                micro_bs = max(1, micro_bs // 2)
                if micro_bs == 1:
                    break

        # 如果自动回退也失败，抛出最后一次 OOM
        if last_oom is not None:
            raise last_oom
        raise RuntimeError("A2C train_step failed unexpectedly")

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """可选的评估接口：返回 (log_probs, entropy, values)。"""

        logits, values = self.net(states)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values
