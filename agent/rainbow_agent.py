"""Rainbow DQN 推理/训练用的智能体实现 (增强版)。"""

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
from agent.rainbow_replay_buffer import TransitionBatch
from agent.utils import resolve_device


class RainbowAgent(BaseAgent):
    """包含在线/目标网络的 Rainbow 智能体，既可训练也可用于推理。"""

    def __init__(self, config: RainbowConfig, device: Optional[torch.device] = None) -> None:
        """初始化在线/目标网络、优化器等核心组件。"""

        self.cfg = config
        self.device = device or resolve_device(config.device)
        self.action_dim = 3

        self.online_net = RainbowBackbone(
            input_channels=int(getattr(config, "obs_channels", 3)),
            obs_size=getattr(config, "obs_size", 84),
            action_dim=self.action_dim,
            atom_size=config.atom_size,
        ).to(self.device)
        self.target_net = RainbowBackbone(
            input_channels=int(getattr(config, "obs_channels", 3)),
            obs_size=getattr(config, "obs_size", 84),
            action_dim=self.action_dim,
            atom_size=config.atom_size,
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = torch.optim.AdamW(
            self.online_net.parameters(), 
            lr=config.lr, 
            eps=1.5e-4,
            weight_decay=1e-5
        )
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
        
        # states/next_states 可能很大；为降低峰值显存，先放 CPU，再按 micro-batch 搬运
        start_states_cpu = torch.as_tensor(batch.states, dtype=torch.float32, device="cpu")
        next_states_cpu = torch.as_tensor(batch.next_states, dtype=torch.float32, device="cpu")

        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=device).unsqueeze(-1)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=device)
        
        # Handle weights
        if weights is None:
            weights_t = torch.ones(len(batch.states), device=device)
        elif isinstance(weights, torch.Tensor):
            weights_t = weights.to(device)
        else:
            weights_t = torch.as_tensor(weights, dtype=torch.float32, device=device)

        batch_size = int(start_states_cpu.shape[0])
        atom_size = int(self.cfg.atom_size)

        def _initial_micro_bs() -> int:
            cfg_mb = int(getattr(self.cfg, "train_micro_batch_size", 0))
            if cfg_mb > 0:
                return min(cfg_mb, batch_size)
            start = min(batch_size, 512)
            if device.type == "cuda" and torch.cuda.is_available():
                try:
                    free_bytes, _ = torch.cuda.mem_get_info(device)
                    free_gb = float(free_bytes) / (1024.0 ** 3)
                    start = min(batch_size, max(start, int(free_gb * 128)))
                    start = min(start, 4096)
                except Exception:
                    pass
            return max(32, int(start))

        def _empty_cuda_cache() -> None:
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        micro_bs = _initial_micro_bs()
        cfg_mb_raw = int(getattr(self.cfg, "train_micro_batch_size", 0))
        last_oom: Exception | None = None

        for _ in range(8):
            try:
                # 固定一次 update 内的噪声（不要在 micro-batch 内重复 reset）
                self.online_net.reset_noise()
                self.target_net.reset_noise()

                self.optimizer.zero_grad(set_to_none=True)
                td_errors = torch.empty((batch_size,), dtype=torch.float32, device=device)

                # 以全 batch 的 mean 为目标：sum(loss_i * w_i) / B
                denom = float(batch_size)
                total_loss_value = 0.0

                for start in range(0, batch_size, micro_bs):
                    end = min(batch_size, start + micro_bs)
                    mb = end - start

                    s_x = start_states_cpu[start:end].to(device, non_blocking=False)
                    a_mb = actions[start:end]

                    # 当前动作分布（带梯度）
                    adv, val = self.online_net(s_x)
                    dist = self._logits_to_dist(adv, val)
                    action_dist = dist.gather(1, a_mb.unsqueeze(-1).expand(-1, 1, atom_size)).squeeze(1)
                    action_dist = torch.clamp(action_dist, min=1e-6)

                    # 目标投影（无梯度）
                    with torch.no_grad():
                        ns_x = next_states_cpu[start:end].to(device, non_blocking=False)
                        r_mb = rewards[start:end]
                        d_mb = dones[start:end]
                        w_mb = weights_t[start:end]

                        next_adv, next_val = self.online_net(ns_x)
                        next_dist = self._logits_to_dist(next_adv, next_val)
                        next_q = torch.sum(next_dist * self.support, dim=-1)
                        next_actions = next_q.argmax(dim=1, keepdim=True)

                        tar_adv, tar_val = self.target_net(ns_x)
                        tar_dist_full = self._logits_to_dist(tar_adv, tar_val)
                        tar_dist = tar_dist_full.gather(
                            1,
                            next_actions.unsqueeze(-1).expand(-1, 1, atom_size),
                        ).squeeze(1)

                        t_z = r_mb.unsqueeze(1) + (self.cfg.gamma**self.cfg.multi_step) * (1.0 - d_mb.unsqueeze(1)) * self.support
                        t_z = t_z.clamp(self.cfg.v_min, self.cfg.v_max)
                        b = (t_z - self.cfg.v_min) / self.delta_z
                        l = b.floor().long().clamp(0, atom_size - 1)
                        u = b.ceil().long().clamp(0, atom_size - 1)

                        proj_dist = torch.zeros((mb, atom_size), dtype=torch.float32, device=device)
                        batch_range = torch.arange(mb, device=device).unsqueeze(1) * atom_size
                        proj_dist.view(-1).index_add_(0, (l + batch_range).view(-1), (tar_dist * (u.float() - b)).view(-1))
                        proj_dist.view(-1).index_add_(0, (u + batch_range).view(-1), (tar_dist * (b - l.float())).view(-1))

                    # 交叉熵：-sum(proj * log(p))，并按 weights 加权
                    per_sample = -torch.sum(proj_dist * torch.log(action_dist), dim=1)
                    loss_mb = (per_sample * w_mb).sum() / denom
                    total_loss_value += float(loss_mb.detach().item())
                    loss_mb.backward()

                    td_errors[start:end] = torch.abs(torch.sum(proj_dist - action_dist.detach(), dim=1))

                torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
                self.optimizer.step()

                mode = "auto" if cfg_mb_raw <= 0 else "fixed"
                print(f"[Rainbow] micro_bs({mode}) = {micro_bs}", flush=True)

                return float(total_loss_value), td_errors.detach().cpu().numpy()

            except torch.OutOfMemoryError as e:
                last_oom = e
                _empty_cuda_cache()
                micro_bs = max(1, micro_bs // 2)
                if micro_bs == 1:
                    break

        if last_oom is not None:
            raise last_oom
        raise RuntimeError("Rainbow train_step failed unexpectedly")

    def sync_target(self) -> None:
        """将在线网络的权重复制到目标网络。"""

        self.target_net.load_state_dict(self.online_net.state_dict())

    def _logits_to_dist(self, advantage: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """根据 Duelling 结构组合 Value/Advantage 并归一化为分布。"""

        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        return F.softmax(q_atoms, dim=-1)