"""train_dqn_variants.py

【中文说明】
这是 DQN 家族（DQN / DDQN / PER / Dueling）统一训练入口。

- 训练模式：
    - `--single`：单蛇（更偏“找食物/导航”的 shaping），用于 curriculum 的 Phase 1。
    - 默认（不加 `--single`）：多蛇对战 + 自博弈（battle），用于 curriculum 的 Phase 2。
- 主要功能：并行环境采样、经验回放（含 PER）、目标网络软更新（tau）、混合精度（AMP）、
    以及对手模型分组推断（减少重复推断，提高 FPS）。
- 性能/稳定性相关环境变量：
    - `DQN_CPU_THREADS`：限制 CPU 线程数，减少多进程/多环境时的线程争用。
    - `RIVAL_UPDATE_INTERVAL`：battle 模式下刷新“活跃对手池”的间隔（步数）。

历史说明（原英文注释的中文化）：
统一的 DQN 变体训练器（偏重 battle 性能优化）。支持 DQN/DDQN/PER/Dueling-PER，
包含“按对手模型分组的批量推断”、目标网络软更新，以及针对不同算法的默认超参。
"""

import math
import random
import time
import threading
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# cuDNN autotune：输入形状固定（20x20 grid）的卷积网络可加速
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# 可选：限制 CPU 线程数，减少 AsyncVectorEnv 多进程 worker 与 PyTorch 线程池争用。
# 常见建议：1（或 2），具体取决于 CPU 核数与并行环境数量。
_cpu_threads_env = os.getenv("DQN_CPU_THREADS")
if _cpu_threads_env:
    try:
        _n = max(1, int(_cpu_threads_env))
        torch.set_num_threads(_n)
        # Inter-op threads can be smaller to reduce oversubscription
        try:
            torch.set_num_interop_threads(min(4, _n))
        except Exception:
            pass
        # Best-effort for libraries used in workers; user can still override in shell.
        os.environ.setdefault("OMP_NUM_THREADS", str(_n))
        os.environ.setdefault("MKL_NUM_THREADS", str(_n))
    except Exception:
        pass

# V31.1：对齐 PPO：对“动态 batch”（对手分组推断大小会变）跳过 CUDAGraphs
if hasattr(torch, '_inductor'):
    import torch._inductor.config as inductor_config
    inductor_config.triton.cudagraph_skip_dynamic_graphs = True

# V18.4：开启 TF32（Ampere+）以使用 Tensor Core 加速 matmul/conv
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

from env.battle_snake_env import BattleSnakeEnv, BattleSnakeConfig
from env.gymnasium_wrapper import make_gymnasium_env
import gymnasium as gym

from agent.dqn import DQNNet, DQNAgent
from agent.ddqn import DDQNNet, DDQNAgent
from agent.per_dqn import PERDQNNet, PERDQNAgent
from agent.dueling_dqn import DuelingDQNNet, DuelingDQNAgent
from utils.self_play import SelfPlayManager


def _normalize_variant(v: str) -> str:
    """Normalize user-facing variant names to internal canonical names.

    中文：将命令行/用户输入的各种写法（如 `ddqn-per`、`ddqn_per_dueling`）统一归一化，
    便于后续根据 variant 选择网络结构、回放缓冲区与训练超参。

    Canonical variants:
      - dqn
      - ddqn
      - per        (DDQN + PER)
      - dueling    (DDQN + PER + Dueling)
    """
    v0 = (v or "").strip().lower()
    v1 = v0.replace("+", "_").replace("-", "_")
    if v1 in {"dqn"}:
        return "dqn"
    if v1 in {"ddqn"}:
        return "ddqn"
    if v1 in {"per", "ddqn_per", "ddqn_per_nodueling", "ddqnper"}:
        return "per"
    if v1 in {"dueling", "ddqn_per_dueling", "per_dueling", "ddqnperdueling"}:
        return "dueling"
    raise ValueError(
        f"Unknown variant '{v}'. Use one of: dqn, ddqn, per(=ddqn+per), dueling(=ddqn+per+dueling)."
    )

# V17.4: AMP for faster GPU training (Updated API for PyTorch 2.x)
from torch.amp import autocast, GradScaler

def log(msg):
    """打印训练日志（强制 flush，便于重定向/监控）。"""
    print(msg, flush=True)

# --- Buffer Implementations ---

class FastReplayBuffer:
    """标准 DQN 的快速回放缓冲区（GPU 侧存储）。

    中文要点：
    - 该实现把 transition 的主要张量直接放在 GPU 上，减少 host<->device 同步开销。
    - `push_batch()` 支持一次写入一批并行环境采样的数据。
    - `sample()` 直接在 GPU 上生成随机索引并取样。
    """
    def __init__(self, capacity, grid_shape, vector_dim, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        
        # V28.0: Full GPU Zero-Sync Buffer for Standard DQN
        self.grids = torch.zeros((capacity, *grid_shape), dtype=torch.uint8, device=device)
        self.vectors = torch.zeros((capacity, vector_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_grids = torch.zeros((capacity, *grid_shape), dtype=torch.uint8, device=device)
        self.next_vectors = torch.zeros((capacity, vector_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
        
        # GPU state trackers
        self.register_buffer("_ptr_val", torch.zeros(1, dtype=torch.long, device=device))
        self.register_buffer("_size_val", torch.zeros(1, dtype=torch.long, device=device))
        
        # CPU Shadow variables for zero-sync indexing
        self._ptr = 0
        self._size = 0

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    @property
    def ptr(self): return self._ptr

    @property
    def size(self): return self._size

    def push_batch(self, grids, vecs, actions, rewards, next_grids, next_vecs, dones):
        num = len(grids)
        idx_range = (torch.arange(self._ptr, self._ptr + num, device=self.device) % self.capacity)
        
        self.grids[idx_range] = torch.as_tensor(grids, dtype=torch.uint8, device=self.device)
        self.vectors[idx_range] = torch.as_tensor(vecs, dtype=torch.float32, device=self.device)
        self.actions[idx_range] = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        self.rewards[idx_range] = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        self.next_grids[idx_range] = torch.as_tensor(next_grids, dtype=torch.uint8, device=self.device)
        self.next_vectors[idx_range] = torch.as_tensor(next_vecs, dtype=torch.float32, device=self.device)
        self.dones[idx_range] = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
        
        # Update trackers (Shadowed)
        self._ptr = (self._ptr + num) % self.capacity
        self._size = min(self._size + num, self.capacity)
        self._ptr_val[0] = self._ptr
        self._size_val[0] = self._size

    def sample(self):
        # V28.0: Pure indexing on GPU with Zero-Sync
        batch = torch.randint(0, self._size, (self.batch_size,), device=self.device)
        
        return (
            { "grid": self.grids[batch].float(), "vector": self.vectors[batch] },
            self.actions[batch], self.rewards[batch],
            { "grid": self.next_grids[batch].float(), "vector": self.next_vectors[batch] },
            self.dones[batch],
            None, # weights
            None  # indices
        )

    def update_priorities(self, idxs, td_errors):
        # Optional: Standard DQN ignores this, but we keep it for API compatibility
        pass



# V25.0：已弃用 TorchSumTree（改为线性 GPU priorities + multinomial 采样）

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay（PER）回放缓冲区（GPU 侧存储）。

    中文要点：
    - `priorities` 存储的是 $(|TD|+\\epsilon)^\\alpha$（已做 alpha 幂）。
    - 采样使用 `torch.multinomial` 按优先级抽样，并返回重要性采样权重（IS weights）。
    - `update_priorities()` 用 TD-error 更新优先级；beta 会在训练中逐步增大以减小偏差。
    """
    def __init__(self, capacity, grid_shape, vector_dim, batch_size, device, alpha=0.5, beta=0.4):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-6
        
        # Buffers on GPU
        self.grids = torch.zeros((capacity, *grid_shape), dtype=torch.uint8, device=device)
        self.vectors = torch.zeros((capacity, vector_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_grids = torch.zeros((capacity, *grid_shape), dtype=torch.uint8, device=device)
        self.next_vectors = torch.zeros((capacity, vector_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.priorities = torch.zeros(capacity, dtype=torch.float32, device=device)

        # Maintain running sum of priorities on GPU to avoid O(capacity) reductions during sampling.
        self.register_buffer("_total_priority", torch.zeros(1, dtype=torch.float32, device=device))
        
        # GPU state trackers
        self.register_buffer("_ptr_val", torch.zeros(1, dtype=torch.long, device=device))
        self.register_buffer("_size_val", torch.zeros(1, dtype=torch.long, device=device))
        self.register_buffer("_max_pri_tensor", torch.ones(1, dtype=torch.float32, device=device))
        
        # V25.2: CPU Shadow variables for zero-sync indexing
        self._ptr = 0
        self._size = 0

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    @property
    def ptr(self):
        return self._ptr

    @property
    def size(self):
        return self._size

    def push_batch(self, grids, vecs, actions, rewards, next_grids, next_vecs, dones):
        num = len(grids)
        idx_range = (torch.arange(self._ptr, self._ptr + num, device=self.device) % self.capacity)
        
        self.grids[idx_range] = torch.as_tensor(grids, dtype=torch.uint8, device=self.device)
        self.vectors[idx_range] = torch.as_tensor(vecs, dtype=torch.float32, device=self.device)
        self.actions[idx_range] = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        self.rewards[idx_range] = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        self.next_grids[idx_range] = torch.as_tensor(next_grids, dtype=torch.uint8, device=self.device)
        self.next_vectors[idx_range] = torch.as_tensor(next_vecs, dtype=torch.float32, device=self.device)
        self.dones[idx_range] = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
        
       # O(1) GPU Priority Update (No host sync)
        # NOTE: `self.priorities` stores *already alpha-exponentiated* priorities (|TD|^alpha).
        # `update_priorities()` computes (|TD| + eps)^alpha, so we must NOT apply `^alpha` again here.
        old_sum = self.priorities[idx_range].sum()
        new_p = self._max_pri_tensor[0]
        self.priorities[idx_range] = new_p
        self._total_priority[0] += (new_p * num) - old_sum
        
        # Update trackers (Both GPU and CPU shadow)
        self._ptr = (self._ptr + num) % self.capacity
        self._size = min(self._size + num, self.capacity)
        self._ptr_val[0] = self._ptr
        self._size_val[0] = self._size

    def update_priorities(self, idxs, td_errors):
        ps = (td_errors.detach().abs() + self.epsilon).pow(self.alpha)

        # idxs may contain duplicates (sampling with replacement). Aggregate by max priority per index.
        if hasattr(ps, "scatter_reduce_"):
            uniq, inv = torch.unique(idxs, return_inverse=True)
            agg = torch.zeros((uniq.shape[0],), device=self.device, dtype=ps.dtype)
            agg.scatter_reduce_(0, inv, ps, reduce="amax", include_self=False)
            old = self.priorities[uniq]
            self.priorities[uniq] = agg
            self._total_priority[0] += (agg - old).sum()
            new_max = agg.max()
        else:
            # Fallback: best-effort without exact duplicate handling
            old = self.priorities[idxs]
            self.priorities[idxs] = ps
            self._total_priority[0] += (ps - old).sum()
            new_max = ps.max()

        # Update max_priority on GPU
        self._max_pri_tensor[0] = torch.max(self._max_pri_tensor[0], new_max)

    def sample(self, beta: float):
        # V25.1: Zero-Sync GPU Multinomial Sampling
        curr_size = self.size
        # Slicing is okay, but we use the fixed capacity if we want to avoid host-syncing 'size'
        # pr = self.priorities[:curr_size]
        # To truly avoid host-sync, one could pad priorities with 0 for unused indices.
        # But for multinomial, we need a 1D tensor of weights.
        
        pr = self.priorities[:curr_size]
        idxs = torch.multinomial(pr, self.batch_size, replacement=True)
        
        # Importance weights calculation
        total_p = self._total_priority[0]
        probs = pr[idxs] / (total_p + 1e-8)
        weights = (curr_size * probs).pow(-beta)
        weights = weights / (weights.max() + 1e-8)
        
        return (
            { "grid": self.grids[idxs].float(), "vector": self.vectors[idxs] },
            self.actions[idxs], self.rewards[idxs],
            { "grid": self.next_grids[idxs].float(), "vector": self.next_vectors[idxs] },
            self.dones[idxs], weights, idxs
        )

# --- Main Trainer ---

@dataclass
class TrainConfig:
    """训练配置（TrainConfig）。

    中文字段说明（只写关键项）：
    - `variant`：算法变体（dqn/ddqn/per/dueling），会先经过 `_normalize_variant()` 归一化。
    - `total_frames`：总交互步数（跨所有并行环境累计）。
    - `num_envs`/`num_envs_override`：并行环境数量；battle 和 single 会有不同默认值。
    - `num_snakes`：环境中蛇的数量；当 `single_snake=True` 时会强制为 1。
    - `self_play_prob_*`：自博弈概率的“前高后低”日程，用于提升泛化。
    - `pool_dir`：自博弈模型池目录（用于抽取/保存历史对手）。
    - `save_path`/`load_path`：best 模型保存路径 / 可选加载路径（Phase 2 常用于微调）。
    """
    variant: str = "dqn" # dqn, ddqn, per(=ddqn+per), dueling(=ddqn+per+dueling)
    total_frames: int = 5_000_000 
    num_envs: int = 64 # V26.0: Optimal IPC for A6000 Phase 2
    num_envs_override: Optional[int] = None
    batch_size: int = 512 
    lr: float = 1e-4
    eps_decay: int = 0  
    tau: float = 0.005 
    num_snakes: int = 4
    pool_dir: str = "agent/pool/dqn"
    save_path: str = "agent/checkpoints/dqn_best.pth"
    load_path: Optional[str] = None
    # Recommended defaults for best performance in battle/self-play
    self_play_prob: float = 0.6
    buffer_size: int = 200_000 # V17.4: Reduced to 200k for much faster sampling
    single_snake: bool = False
    # Exploration (can override via CLI)
    eps_start: Optional[float] = None
    eps_min: Optional[float] = None

    # Self-play probability schedule (front-high then low)
    # Defaults tuned for stronger generalization: more self-play early, more randomness later.
    self_play_prob_start: Optional[float] = 0.7
    self_play_prob_end: Optional[float] = 0.4
    self_play_prob_frac: float = 0.30

    # Fine-tune learning rate multiplier when --load is used.
    finetune_lr_mult: float = 0.5

class DQNVariantTrainer:
    """DQN 统一训练器（按 `TrainConfig.variant` 切换实现）。

    中文概览：
    - single 模式更偏“稳定学习导航能力”（更高探索、更密集 shaping）。
    - battle 模式引入自博弈与对手池，训练更偏“对抗/稳定胜率”。
    - 末尾会额外保存一个 `.final.pth` 快照，避免 best 与 final 混淆。
    """
    def __init__(self, cfg: TrainConfig):
        # Normalize variants early (support DDQN+PER naming requested by user)
        cfg.variant = _normalize_variant(cfg.variant)
        self.cfg = cfg  # FIXED: Restored self.cfg assignment

        # Curriculum correctness: when running Phase 1 ("single"), force a true single-snake env.
        # This keeps the observation/action spaces consistent while matching intended training.
        if cfg.single_snake:
            self.cfg.num_snakes = 1
        # Variant-aware exploration schedule.
        # Use percentage-based decay to adapt to any total_frames count (1M, 5M, etc.)
        total = max(1, int(cfg.total_frames))
        
        # Decide Decay Duration: 85% of total steps for single, 90% for battle (need more exploration)
        # V35.0: Universal Compile Config - Skip dynamic graphs to stay stable with groups
        if hasattr(torch, '_inductor'):
            torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Restore Decay Steps (V35.1 Fix)
        if cfg.single_snake:
            decay_ratio = 0.80 
        else:
            decay_ratio = 0.90
        self.decay_steps = int(total * decay_ratio)
        self.cfg.eps_decay = self.decay_steps

        # V35.0 Aggressive Optimization: Reduce overhead
        # Respect explicit overrides for A6000 benchmarking.
        if cfg.num_envs_override is not None:
            self.cfg.num_envs = int(cfg.num_envs_override)
        else:
            if not cfg.single_snake:
                self.cfg.num_envs = 32 # Reduced from 64 to save CPU/Memory
            else:
                self.cfg.num_envs = 128

        if not cfg.single_snake:
            self.active_rivals = [] # Global Active Rival models (paths)
            # Reduce FPS spikes from frequent model refresh/load.
            # Can override via env var for experiments.
            self.rival_update_interval = int(os.getenv("RIVAL_UPDATE_INTERVAL", "50000"))
            self.next_rival_update = 0
        
        # Opponent Grouping Cache
        self.cached_opp_groups = {}
        self.opp_model_paths = [] 
        
        v = cfg.variant
        num_snakes = cfg.num_snakes
        if cfg.single_snake:
            # Phase 1: High Exploration & Robust Learning
            default_eps_start, default_eps_min = 1.0, 0.02
            self.batch_size = 1024 # V21.1: Increased for GPU saturation
            if v == "dqn":
                self.lr, self.tau = 1.5e-4, 0.002
            else:
                self.lr, self.tau = 2.0e-4, 0.005
        else:
            # Phase 2: High Stability & Combat Precision
            default_eps_start, default_eps_min = 0.6, 0.05
            if v in ("per", "dueling"):
                # PER/Dueling are more brittle in battle; reduce exploration noise to avoid policy degradation.
                default_eps_start, default_eps_min = 0.40, 0.02
            self.batch_size = 1024 # V21.1: Increased
            if v == "dqn":
                self.lr, self.tau = 1.0e-4, 0.002
            elif v == "ddqn":
                # DDQN benefits from the same conservative target updates as DQN in noisy battle.
                self.lr, self.tau = 1.0e-4, 0.002
            elif v == "per":
                # PER is stable with conservative tau; keep LR aligned to DQN.
                self.lr, self.tau = 1.0e-4, 0.002
            elif v == "dueling":
                # Dueling is more sensitive in battle; keep target updates conservative.
                self.lr, self.tau = 1.0e-4, 0.002
            else:
                self.lr, self.tau = 8.0e-5, 0.005

        # Allow overriding eps schedule from config/CLI
        self.eps_start = float(cfg.eps_start) if cfg.eps_start is not None else float(default_eps_start)
        self.eps_min = float(cfg.eps_min) if cfg.eps_min is not None else float(default_eps_min)

        # Self-play probability schedule (front-high then low)
        self.sp_prob_start = float(cfg.self_play_prob_start) if cfg.self_play_prob_start is not None else float(cfg.self_play_prob)
        self.sp_prob_end = float(cfg.self_play_prob_end) if cfg.self_play_prob_end is not None else float(cfg.self_play_prob)
        self.sp_prob_frac = float(cfg.self_play_prob_frac) if cfg.self_play_prob_frac is not None else 0.30

        # If self-play is fully disabled, avoid requesting full observations and avoid opponent model inference.
        self.use_model_opps = (
            (not cfg.single_snake)
            and (cfg.num_snakes > 1)
            and (max(self.sp_prob_start, self.sp_prob_end, float(cfg.self_play_prob)) > 0.0)
        )
            
        self.grad_clip = 0.5 
        
        # Override algorithm-specific if needed
        if cfg.variant == "per":
            # PER tuning:
            # - Phase 1: keep higher alpha for fast learning.
            # - Phase 2: slightly higher alpha than before to learn faster, and higher beta_start
            #   to reduce sampling bias earlier.
            self.per_alpha = 0.6 if cfg.single_snake else 0.5
            self.per_beta_start = 0.4 if cfg.single_snake else 0.6
        elif cfg.variant == "dueling":
            # Keep dueling PER more conservative by default.
            self.per_alpha = 0.6 if cfg.single_snake else 0.4
            self.per_beta_start = 0.4
        
        # V12.0: Buffer size managed via TrainConfig for A6000 visibility
        self.buffer_size = cfg.buffer_size

        # V45.0: DDQN Fix - Remove LR penalty that caused 26% performance loss
        # Analysis: V44.6 halved LR (8e-5->4e-5), causing DDQN Ph1 reward=101 vs DQN=136
        # Fix: Restore full LR 8e-5 to match DQN convergence speed
        # Removed: if cfg.variant == "ddqn": self.lr = self.lr * 0.5

        log(f">>> [V8.0 Asymmetric-Tuning] Variant: {cfg.variant.upper()} | LR: {self.lr} | Tau: {self.tau} | GradClip: {self.grad_clip} | Buf: {self.buffer_size}")
        log(f">>> [Explore] eps_start={self.eps_start:.2f} eps_min={self.eps_min:.2f} | [SelfPlay] prob={self.sp_prob_start:.2f}->{self.sp_prob_end:.2f} @ {self.sp_prob_frac:.2f}")
        
        env_cfg = BattleSnakeConfig(width=20, height=20, num_snakes=cfg.num_snakes)
        if cfg.num_snakes == 1:
            # Phase 1: High focus on navigation (V18.3 Turbo Sync)
            env_cfg.closer_reward = 0.05
            env_cfg.farther_penalty = -0.04
            env_cfg.food_reward = 1.0
            env_cfg.death_penalty = -3.0
            env_cfg.step_penalty = -0.01
            env_cfg.self_collision_penalty = -4.0
            env_cfg.min_food = 5
            log(f">>> [V18.3 Turbo] PHASE 1 (Single) | FoodDensity: {env_cfg.min_food} | FoodRew: {env_cfg.food_reward}")
        else:
            # Phase 2: V9.0 Combat (High Aggression)
            env_cfg.closer_reward = 0.05
            env_cfg.farther_penalty = -0.04
            env_cfg.step_penalty = -0.01
            env_cfg.death_penalty = -3.0
            env_cfg.kill_reward = 2.0
            env_cfg.food_reward = 1.2
            env_cfg.self_collision_penalty = -4.0
            env_cfg.win_reward = 5.0
            env_cfg.loss_penalty = -2.0
            env_cfg.min_food = 2
            log(f">>> [V9.0 Battle] PHASE 2 | Kill: {env_cfg.kill_reward} | FoodDensity: {env_cfg.min_food}")
        
        log(f">>> [V18.3 Turbo] Initializing {cfg.num_envs} Parallel Environments...")
        def env_fn():
             extra_shaping = {}
             unsafe_move_pen = -0.10
             if cfg.variant in ("per", "dueling") and (not cfg.single_snake):
                 # Stronger safety + anti-loop shaping for brittle variants in battle mode.
                 unsafe_move_pen = -0.12
                 extra_shaping = {
                     "unsafe_move2_penalty": -0.04,
                     "revisit_penalty": -0.01,
                     "revisit_window": 24,
                 }
             # Convert BattleSnakeConfig to gymnasium wrapper keywords
             return make_gymnasium_env(
                 num_snakes=cfg.num_snakes, 
                 grid_size=20,
                 min_food=env_cfg.min_food,
                 closer_reward=env_cfg.closer_reward,
                 farther_penalty=env_cfg.farther_penalty,
                 food_reward=env_cfg.food_reward,
                 death_penalty=env_cfg.death_penalty,
                 step_penalty=env_cfg.step_penalty,
                 self_collision_penalty=env_cfg.self_collision_penalty,
                 # Only passed if present in BattleSnakeConfig
                 kill_reward=getattr(env_cfg, 'kill_reward', 150.0),
                 win_reward=getattr(env_cfg, 'win_reward', 800.0),
                 loss_penalty=getattr(env_cfg, 'loss_penalty', -200.0),
                 # Learn game rules via score delta (length-scaled scoring), with a bit of dense shaping retained.
                 use_score_delta_reward=True,
                 score_reward_coef=0.001,
                 env_reward_coef=0.2,
                 # Encourage purposeful dash (score gain soon after dash) instead of spamming.
                 dash_effect_window=6,
                 dash_success_bonus=0.2,
                 dash_fail_penalty=-0.2,
                 # Safety shaping: penalize obviously unsafe moves (wall/body) and risky/invalid dash.
                 unsafe_move_penalty=unsafe_move_pen,
                 unsafe_dash_penalty=-0.10,
                 invalid_dash_penalty=-0.02,
                 **extra_shaping,
                 return_full_obs=self.use_model_opps,
             )
        self.envs = gym.vector.AsyncVectorEnv([env_fn for _ in range(cfg.num_envs)])
        log(">>> [V18.3 Turbo] Environments Ready.")
        
        # Select Architecture
        if cfg.variant == "dqn": self.net_cls = DQNNet
        elif cfg.variant == "ddqn": self.net_cls = DDQNNet
        elif cfg.variant == "per": 
            from agent.per_dqn import PERDQNNet
            self.net_cls = PERDQNNet 
        elif cfg.variant == "dueling": self.net_cls = DuelingDQNNet
        else: raise ValueError(f"Unknown variant {cfg.variant}")
        
        self.policy_net = self.net_cls(vector_dim=28).to(self.device)
        self.target_net = self.net_cls(vector_dim=28).to(self.device)

        # Memory format optimization for Conv2d on CUDA
        if self.device.type == "cuda":
            self.policy_net = self.policy_net.to(memory_format=torch.channels_last)
            self.target_net = self.target_net.to(memory_format=torch.channels_last)
        
        # 1. Load weights (V37.0: Dynamic stripping of _orig_mod prefixes)
        if cfg.load_path and Path(cfg.load_path).exists():
            log(f">>> Loading weights from {cfg.load_path}...")
            sd = torch.load(cfg.load_path, map_location=self.device, weights_only=True)
            # Clean Prefix: Handle models saved with torch.compile enabled
            sd = { k.replace("_orig_mod.", ""): v for k, v in sd.items() }
            self.policy_net.load_state_dict(sd)
            self.lr = self.lr * float(getattr(cfg, "finetune_lr_mult", 0.25))
            log(f">>> [V22.0] Fine-tuning mode: Lowered LR to {self.lr:.2e}")
            
        # 2. Sync target net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 3. Initialize Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # 4. Final Compilation (Only on Linux, after loading/syncing)
        if os.name != 'nt' and hasattr(torch, 'compile'):
            try:
                self.policy_net = torch.compile(self.policy_net, mode='reduce-overhead')
                log(">>> [DQN-Turbo] torch.compile() enabled.")
            except Exception as e:
                log(f">>> [DQN-Turbo] torch.compile() failed: {e}")
        else:
            log(">>> [DQN-Turbo] torch.compile() skipped (Windows/Unsupported).")
        
        # V28.0: Revert DQN to vanilla sampling while keeping PER for variants
        # V28.1: dqn and ddqn use uniform sampling; per and dueling use prioritized
        if cfg.variant in ["dqn", "ddqn"]:
            self.memory = FastReplayBuffer(self.buffer_size, (5, 20, 20), 28, self.batch_size, self.device)
        else:
            alpha = getattr(self, 'per_alpha', 0.6)
            self.memory = PrioritizedReplayBuffer(self.buffer_size, (5, 20, 20), 28, self.batch_size, self.device, alpha=alpha)
            
        self.steps = 0
        log(f">>> [V13.3 Heartbeat] Setup Finished. Buffer: {self.buffer_size} | Device: {self.device}")
        self.sp_manager = SelfPlayManager(cfg.pool_dir)
        
        # Self-Play Manager (V6.7 Model Cache)
        # Store model paths for opponents. None means random.
        self.opp_model_paths = [[None]*cfg.num_snakes for _ in range(cfg.num_envs)]
        self.loaded_opp_models: Dict[str, nn.Module] = {}

        # Async prefetch for opponent weights (CPU) to avoid I/O spikes in battle mode.
        # Main thread will materialize GPU modules lazily when weights are ready.
        self._prefetch_queue: "queue.SimpleQueue[str]" = queue.SimpleQueue()
        self._prefetch_lock = threading.Lock()
        self._prefetch_pending: set[str] = set()
        self._prefetched_sd: Dict[str, Tuple[float, Dict[str, torch.Tensor]]] = {}
        self._prefetch_stop = threading.Event()
        self._prefetch_max_items = int(os.getenv("PREFETCH_MAX_ITEMS", "64"))
        self._prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._prefetch_thread.start()
        
        # Best-checkpoint tracking (battle uses S0/Win%, single uses avg reward)
        self.best_reward = -float('inf')
        self.best_s0 = -float('inf')
        self.best_win = -float('inf')
        
        # V39.0: Unified Gamma (Dueling was 0.995 -> unstable)
        # All variants now use 0.99 for stable value estimation
        self.gamma = 0.99
        # V17.2: Reduced Update Frequency to 1
        self.updates_per_step = 1
        
        # V17.4: AMP Scaler for mixed precision training
        self.scaler = GradScaler('cuda')
        self.use_amp = True

    def save_model(self, path):
        """保存当前策略网络到磁盘（原子写入，跨平台尽量稳）。

        中文：
        - 训练中可能随时中断，因此先写入 `.tmp` 再替换目标文件。
        - 如果启用了 `torch.compile`，会存在 `_orig_mod` 包装，这里保存“干净”的 state_dict。
        - 统一保存 CPU Tensor，避免序列化 CUDA Tensor 带来的兼容性问题。
        """
        # V37.0: Always save the clean state_dict (stripping torch.compile wrappers)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = p.with_suffix(p.suffix + ".tmp")
        
        # 访问原始 module（如果被 torch.compile 包装）
        raw_net = self.policy_net._orig_mod if hasattr(self.policy_net, "_orig_mod") else self.policy_net
        # 保存 CPU tensor：更便携，避免把 CUDA 张量写进权重文件
        sd_cpu = {k: v.detach().cpu() for k, v in raw_net.state_dict().items()}
        torch.save(sd_cpu, tmp_path)
        
        if os.path.exists(p):
            try:
                os.replace(tmp_path, p)
            except OSError:
                # Windows 下少数情况下 replace 可能失败：退化为删除再重命名
                os.remove(p)
                os.rename(tmp_path, p)
        else:
            os.rename(tmp_path, p)

    def _get_opp_model(self, path: str) -> nn.Module:
        """获取对手模型（优先缓存/预取结果），失败则返回 None。

        中文：battle 自博弈下，对手模型来自历史池（磁盘上的权重文件）。
        - 直接在训练主线程频繁 `torch.load` 会造成 I/O 抖动；因此先在后台线程预取到 CPU。
        - 主线程只在“CPU 权重已准备好”时才 materialize 到 GPU 并进入缓存。
        - 如果文件缺失/损坏/尚未预取完成：返回 None，调用方会回退到随机动作。
        """
        if path in self.loaded_opp_models:
            return self.loaded_opp_models.get(path)

        # 如果后台已经把权重预取到 CPU，则在这里快速创建 GPU 模型并 load
        try:
            mtime = os.stat(path).st_mtime
        except OSError:
            return None

        cached = self._prefetched_sd.get(path)
        if cached is not None and cached[0] == mtime:
            try:
                model = self.net_cls(vector_dim=28).to(self.device).eval()
                if self.device.type == "cuda":
                    model = model.to(memory_format=torch.channels_last)
                sd = cached[1]
                model.load_state_dict(sd)
                for p in model.parameters():
                    p.requires_grad = False
                self.loaded_opp_models[path] = model
                if len(self.loaded_opp_models) > 500:
                    key_to_del = next(iter(self.loaded_opp_models))
                    del self.loaded_opp_models[key_to_del]
                return model
            except Exception:
                return None

        # 预取尚未完成：把路径丢进队列，暂时跳过（调用方回退随机动作）
        self._enqueue_prefetch(path)
        return None

    def _enqueue_prefetch(self, path: Optional[str]) -> None:
        if not path:
            return
        with self._prefetch_lock:
            if path in self._prefetch_pending:
                return
            self._prefetch_pending.add(path)
        self._prefetch_queue.put(path)

    def _prefetch_worker(self) -> None:
        """后台线程：把对手权重文件预取到 CPU 内存。

        中文：
        - 只做 CPU 侧 `torch.load`，不在后台创建 GPU 模型，避免和训练抢 GPU。
        - 通过 mtime 判断是否需要更新缓存。
        """
        while not self._prefetch_stop.is_set():
            try:
                path = self._prefetch_queue.get(timeout=0.5)
            except Exception:
                continue

            with self._prefetch_lock:
                self._prefetch_pending.discard(path)

            try:
                mtime = os.stat(path).st_mtime
            except OSError:
                continue

            cached = self._prefetched_sd.get(path)
            if cached is not None and cached[0] == mtime:
                continue

            try:
                sd = torch.load(path, map_location="cpu", weights_only=True)
                sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
                self._prefetched_sd[path] = (mtime, sd)

                # simple LRU-ish eviction
                if len(self._prefetched_sd) > self._prefetch_max_items:
                    self._prefetched_sd.pop(next(iter(self._prefetched_sd)))
            except Exception:
                continue

    def train(self):
        """主训练循环。

        中文结构（每步）：
        1) 按步数更新 epsilon（探索率）
        2) 选择动作：learner 用 policy_net（epsilon-greedy）；对手随机或模型推断（分组批推断）
        3) env.step(all_actions) 与环境交互
        4) 写入回放（只存 learner 的 transition）
        5) episode 结束统计与 self-play 对手刷新
        6) 达到 batch_size 后更新网络（update）
        7) 软更新 target_net（Polyak）
        8) 周期日志 + best 保存 + 追加历史池快照
        """
        log(f">>> Starting {self.cfg.variant.upper()} Training (V21.0 Flash-Batch)...")
        obs_dict, info = self.envs.reset()
        ep_rewards = [0.0] * self.cfg.num_envs
        recent_rewards = []
        recent_wins = []
        recent_score0 = []
        last_log_time = time.time()
        
        total_frames = max(1, int(self.cfg.total_frames))
        num_snakes = self.cfg.num_snakes
        total_snakes = self.cfg.num_envs * num_snakes

        # 预计算索引与 pinned CPU buffer：
        # - `learner_map`：在 flatten 的 total_snakes 维度中，定位每个 env 的 learner（蛇0）
        # - `actions_cpu`：pinned memory 可加速 GPU->CPU 拷贝，便于喂给 envs.step()
        learner_map = torch.arange(0, total_snakes, num_snakes, device=self.device)
        actions_cpu = torch.empty((total_snakes,), dtype=torch.int32, device="cpu", pin_memory=True)

        # 复用张量：避免每 step 分配导致的 GPU allocator 抖动
        current_actions = torch.empty(total_snakes, dtype=torch.int32, device=self.device)
        
        # V26.0 Initialization
        if self.use_model_opps:
            self.opp_model_paths = [([None] * num_snakes) for _ in range(self.cfg.num_envs)]
            self._rebuild_opp_groups()
        
        while self.steps < total_frames:
            self.steps += self.cfg.num_envs
            completion = min(1.0, self.steps / self.decay_steps)
            eps = max(self.eps_min, self.eps_start - completion * (self.eps_start - self.eps_min))

            # ====== 1) 动作选择（epsilon-greedy）======
            # - learner：用 Q(s,·) 选 argmax；以 eps 概率随机动作
            # - 对手：battle 时可来自历史池模型（self-play），否则随机
            if self.use_model_opps:
                # battle 模式：从 info 取全体蛇的 full_obs（由 wrapper 提供）
                # - grid 用 uint8 传到 GPU，减少 PCIe 带宽；forward 前再 cast
                full_grids = info.get("full_obs_grids") if isinstance(info, dict) else None
                full_vecs = info.get("full_obs_vecs") if isinstance(info, dict) else None
                if full_grids is None or full_vecs is None:
                    raise KeyError(
                        "Missing 'full_obs_grids'/'full_obs_vecs' in env info for multi-snake mode. "
                        "Ensure the Gymnasium wrapper returns full observations (return_full_obs=True)."
                    )

                all_g_u8 = torch.from_numpy(full_grids).view(total_snakes, 5, 20, 20)
                all_v_f32 = torch.from_numpy(full_vecs).view(total_snakes, -1)

                t_all_g_u8 = all_g_u8.to(self.device, non_blocking=True)
                t_all_v = all_v_f32.to(self.device, non_blocking=True)

                with torch.inference_mode(), autocast('cuda', enabled=self.use_amp):
                    grid_dtype = torch.float16 if (self.device.type == "cuda") else torch.float32
                    t_all_g = t_all_g_u8.to(dtype=grid_dtype)
                    if self.device.type == "cuda":
                        t_all_g = t_all_g.contiguous(memory_format=torch.channels_last)

                    # A) learner（蛇0）：对每个 env 计算一次 Q，并 epsilon-greedy
                    q_vals = self.policy_net(t_all_g[learner_map], t_all_v[learner_map])
                    la = q_vals.argmax(dim=1).to(torch.int32)
                    if eps > 0:
                        rv = torch.rand(self.cfg.num_envs, device=self.device)
                        la = torch.where(
                            rv < eps,
                            torch.randint(0, 4, (self.cfg.num_envs,), device=self.device, dtype=torch.int32),
                            la,
                        )
                    current_actions[learner_map] = la

                    # B) opponents：按“模型路径”分组推断
                    # - m_path=None 表示随机对手
                    # - m_path!=None 尝试取缓存/预取模型；取不到则保持默认（稍后由调用方看到缺口）
                    for m_path, idxs in self.cached_opp_groups.items():
                        if m_path is None:
                            current_actions[idxs] = torch.randint(
                                0, 4, (len(idxs),), device=self.device, dtype=torch.int32
                            )
                        else:
                            m = self._get_opp_model(m_path)
                            if m:
                                current_actions[idxs] = m(t_all_g[idxs], t_all_v[idxs]).argmax(dim=1).to(torch.int32)

                    # 将 total_snakes 的动作一次性拷贝到 CPU，再 reshape 成 (num_envs, num_snakes)
                    actions_cpu.copy_(current_actions, non_blocking=True)
                    all_actions_flat_np = actions_cpu.numpy()

                all_actions = all_actions_flat_np.reshape(self.cfg.num_envs, num_snakes)
            else:
                # single/无 self-play：只用 learner 的 obs；如果是多蛇也让对手随机
                t_g = torch.from_numpy(obs_dict["grid"]).to(self.device, non_blocking=True)
                t_v = torch.from_numpy(obs_dict["vector"]).to(self.device, non_blocking=True)
                with torch.inference_mode(), autocast('cuda', enabled=self.use_amp):
                    grid_dtype = torch.float16 if (self.device.type == "cuda") else torch.float32
                    t_g = t_g.to(dtype=grid_dtype)
                    if self.device.type == "cuda":
                        t_g = t_g.contiguous(memory_format=torch.channels_last)
                    q_vals = self.policy_net(t_g, t_v)
                    la = q_vals.argmax(dim=1).to(torch.int32)
                    if eps > 0:
                        rv = torch.rand(self.cfg.num_envs, device=self.device)
                        la = torch.where(
                            rv < eps,
                            torch.randint(0, 4, (self.cfg.num_envs,), device=self.device, dtype=torch.int32),
                            la,
                        )
                if num_snakes == 1:
                    all_actions = la.view(self.cfg.num_envs, 1).cpu().numpy()
                else:
                    opp = torch.randint(0, 4, (self.cfg.num_envs, num_snakes - 1), device="cpu", dtype=torch.int32)
                    all_actions = torch.cat([la.cpu().view(self.cfg.num_envs, 1), opp], dim=1).numpy()

            # ====== 2) 与环境交互 ======
            # 中文：Gymnasium wrapper 约定：当 action 是 (num_envs, num_snakes) 时，表示全体蛇动作。
            next_obs_dict, rews, terminated, truncated, next_info = self.envs.step(all_actions)
            
            # ====== 3) 写入回放（只存 learner transition）======
            # 中文：对手动作/观测不进入 learner 的回放；训练目标是 learner 的 Q-learning。
            self.memory.push_batch(
                obs_dict["grid"], obs_dict["vector"],
                all_actions[:, 0], rews,
                next_obs_dict["grid"], next_obs_dict["vector"],
                terminated
            )
            
            # ====== 4) episode 统计 + self-play 对手刷新 ======
            # 中文：episode 结束时，记录 return；battle 模式额外记录 win-rate 与 score0。
            should_rebuild = False
            for e_idx in range(self.cfg.num_envs):
                ep_rewards[e_idx] += rews[e_idx]
                if terminated[e_idx] or truncated[e_idx]:
                    recent_rewards.append(ep_rewards[e_idx])
                    if len(recent_rewards) > 100: recent_rewards.pop(0)
                    ep_rewards[e_idx] = 0.0

                    # battle 的额外统计：winner_idx / score0 来自 wrapper 的 info
                    if (not self.cfg.single_snake) and num_snakes > 1 and isinstance(next_info, dict):
                        try:
                            winner = int(next_info.get("winner_idx", [-1])[e_idx])
                        except Exception:
                            winner = -1
                        recent_wins.append(1 if winner == 0 else 0)
                        if len(recent_wins) > 200:
                            recent_wins.pop(0)
                        try:
                            recent_score0.append(int(next_info.get("score0", [0])[e_idx]))
                        except Exception:
                            recent_score0.append(0)
                        if len(recent_score0) > 200:
                            recent_score0.pop(0)

                    # self-play 概率两段式：前期更高（多样性/探索），后期更低（稳定收敛）
                    if not self.cfg.single_snake and num_snakes > 1:
                        prog = self.steps / float(total_frames)
                        sp_prob = self.sp_prob_start if prog < self.sp_prob_frac else self.sp_prob_end
                        for s in range(1, num_snakes):
                            old = self.opp_model_paths[e_idx][s]
                            if self.active_rivals and (random.random() < sp_prob):
                                new = random.choice(self.active_rivals)
                            else:
                                new = None
                            if new != old:
                                self.opp_model_paths[e_idx][s] = new
                                should_rebuild = True

            # ====== 5) 全局对手池刷新（active_rivals）======
            # 中文：定期从 SelfPlayManager 采样若干模型路径，作为“当前活跃对手集合”，
            # 再由每个 env 在 episode 结束时随机选择其中一个作为对手。
            if self.use_model_opps and self.steps >= self.next_rival_update:
                self.next_rival_update = self.steps + self.rival_update_interval
                new_rivals = []
                for _ in range(4):
                    m = self.sp_manager.sample_model()
                    if m: new_rivals.append(str(m))
                if new_rivals:
                    self.active_rivals = new_rivals
                    for p in self.active_rivals:
                        self._enqueue_prefetch(p)
                
            if self.use_model_opps and should_rebuild:
                self._rebuild_opp_groups()
            
            obs_dict, info = next_obs_dict, next_info

            # ====== 6) 学习率退火（有下限）======
            # 中文：学习率从 lr 退火到 10%*lr，避免后期彻底停学。
            progress = self.steps / total_frames
            frac = max(0.0, 1.0 - progress)
            current_lr = self.lr * (0.10 + 0.90 * frac) 
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr

            # ====== 7) 兼容遗留的“阶段阈值”变量（不直接影响逻辑）======
            warmup1, warmup2 = 50000, 150000
            late_half = total_frames * 0.5
            late_40 = total_frames * 0.4
            
            # ====== 8) 网络更新（从回放采样 + 反向传播）======
            # 中文：当回放样本数达到 batch_size 后开始 update。
            # 这里固定每步更新 1 次，是“吞吐/稳定/FPS”之间的折中（并行环境本身就会快速填充回放）。
            if self.memory.size >= self.batch_size:
                updates = 1
                for _ in range(updates):
                    self.update()

            # ====== 9) 目标网络软更新（Polyak）======
            # 中文：target <- (1-tau)*target + tau*policy。
            # 同时根据训练进度与 variant，在后期把 tau 降低以减少抖动。
            tau_eff = self.tau
            if self.cfg.variant == "ddqn":
                late_cut = late_half if self.cfg.single_snake else late_40
                if self.steps >= late_cut:
                    tau_eff = self.tau * 0.5
            if self.cfg.variant == "dqn" and (not self.cfg.single_snake) and self.steps >= late_half:
                tau_eff = self.tau * 0.5
            if self.cfg.variant == "per" and self.steps >= late_half:
                tau_eff = self.tau * 0.5
            if self.cfg.variant == "dueling" and self.steps >= late_half:
                tau_eff = self.tau * 0.5
                
            with torch.no_grad():
                for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                    target_param.mul_(1.0 - tau_eff).add_(policy_param, alpha=tau_eff)
                
            # ====== 10) 心跳日志 + best 保存 + 周期性池快照 ======
            if self.steps % 1024 < self.cfg.num_envs:
                elapsed = time.time() - last_log_time
                fps = (1024) / (elapsed + 1e-6)
                avg_r = np.mean(recent_rewards) if recent_rewards else 0
                if (not self.cfg.single_snake) and num_snakes > 1:
                    win_rate = (np.mean(recent_wins) if recent_wins else 0.0) * 100.0
                    avg_s0 = np.mean(recent_score0) if recent_score0 else 0.0
                    log(
                        f"Step: {self.steps} | EPS: {eps:.2f} | Rew: {avg_r:.2f} | Win%: {win_rate:.1f} | S0: {avg_s0:.1f} | FPS: {fps:.1f} | Var: {self.cfg.variant}"
                    )

                    # 保存 best（battle）：优先 S0，其次 Win%。
                    # 中文：加阈值是为了减少频繁写盘（尤其是 SSD/网络盘）。
                    improved = False
                    if avg_s0 > self.best_s0 + 5.0:
                        improved = True
                    elif abs(avg_s0 - self.best_s0) <= 5.0 and win_rate > self.best_win + 1.0:
                        improved = True
                    elif self.best_s0 == -float('inf') and (recent_score0 or recent_wins):
                        improved = True
                    if improved:
                        self.best_s0 = float(avg_s0)
                        self.best_win = float(win_rate)
                        self.save_model(self.cfg.save_path)
                        log(f">>> [Best] Saved -> {self.cfg.save_path} | Win% {self.best_win:.1f} | S0 {self.best_s0:.1f}")
                else:
                    log(f"Step: {self.steps} | EPS: {eps:.2f} | Rew: {avg_r:.2f} | FPS: {fps:.1f} | Var: {self.cfg.variant}")

                    # 保存 best（single）：按 avg episode reward
                    improved = False
                    if avg_r > self.best_reward + 0.5:
                        improved = True
                    elif self.best_reward == -float('inf') and recent_rewards:
                        improved = True
                    if improved:
                        self.best_reward = float(avg_r)
                        self.save_model(self.cfg.save_path)
                        log(f">>> [Best] Saved -> {self.cfg.save_path} | Rew {self.best_reward:.2f}")
                last_log_time = time.time()
                
                # 周期性追加历史池快照：即使不是 best，也能增加对手多样性
                pool_interval = max(150_000, int(total_frames * 0.03))
                if self.steps % pool_interval < self.cfg.num_envs:
                    raw_net = self.policy_net._orig_mod if hasattr(self.policy_net, "_orig_mod") else self.policy_net
                    sd_cpu = {k: v.detach().cpu() for k, v in raw_net.state_dict().items()}
                    self.sp_manager.add_model(sd_cpu, f"{self.cfg.variant}_step_{self.steps}")
            
        # ====== 训练结束：保证 best 存在，并写 final 快照 ======
        # - `save_path` 始终代表 best（便于 GUI/推理直接使用）
        # - 额外写 `.final.pth` 作为训练末尾的快照
        best_path = Path(self.cfg.save_path)
        if not best_path.exists():
            self.save_model(self.cfg.save_path)
            log(f">>> [Best] (fallback) Saved -> {self.cfg.save_path}")

        final_path = str(best_path.with_suffix(".final.pth"))
        self.save_model(final_path)
        log(f">>> [Final] Saved -> {final_path}")

    def _rebuild_opp_groups(self):
        """按对手模型路径分组 env-index，用于批量推断减少重复计算。"""
        # 中文：把 {path -> [snake_idx,...]} 转成 GPU tensor，推断时无需频繁 Python list 操作。
        temp_groups = {}
        for e in range(self.cfg.num_envs):
            for s in range(1, self.cfg.num_snakes):
                m_path = self.opp_model_paths[e][s]
                idx = e * self.cfg.num_snakes + s
                temp_groups.setdefault(m_path, []).append(idx)

        # Async prefetch any non-random opponent models
        for m in temp_groups.keys():
            if m is not None:
                self._enqueue_prefetch(m)
        
        # Convert to cached tensors for constant-time GPU usage
        self.cached_opp_groups = {
            m: torch.tensor(idxs, device=self.device) for m, idxs in temp_groups.items()
        }

    def update(self):
        """从回放缓冲区采样并执行一次梯度更新（含 PER/AMP/梯度裁剪）。"""
        # ====== PER 的 beta 退火 ======
        # 中文：beta 从 beta_start -> 1.0（随训练进度增大），逐步抵消优先采样带来的偏差。
        frac = self.steps / self.cfg.total_frames
        beta_start = getattr(self, "per_beta_start", 0.4)
        current_beta = min(1.0, beta_start + frac * (1.0 - beta_start))
        
        if "per" in self.cfg.variant or "dueling" in self.cfg.variant:
            states, actions, rewards, next_states, dones, weights, idxs = self.memory.sample(current_beta)
        else:
            states, actions, rewards, next_states, dones, weights, idxs = self.memory.sample()
        
        # ====== 计算 TD target 与 loss（AMP）======
        # 中文：
        # - DQN：target 用 target_net 的 max(Q')
        # - DDQN/PER/Dueling：用 policy_net 选 best_action，再用 target_net 评估（减小过估计）
        # - PER：用 IS weights 加权每个样本的 loss
        with autocast('cuda', enabled=self.use_amp):
            q_curr = self.policy_net(states['grid'], states['vector']).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            with torch.no_grad():
                if self.cfg.variant == "dqn":
                    q_next = self.target_net(next_states['grid'], next_states['vector']).max(1)[0]
                else:
                    best_actions = self.policy_net(next_states['grid'], next_states['vector']).argmax(1)
                    q_next = self.target_net(next_states['grid'], next_states['vector']).gather(1, best_actions.unsqueeze(1)).squeeze(1)
                target = rewards + self.gamma * q_next * (1.0 - dones.float())
                
            td_errors = q_curr - target
            if weights is not None:
                per_sample = nn.SmoothL1Loss(reduction="none")(q_curr, target)
                loss = (weights * per_sample).mean()
            else:
                loss = nn.SmoothL1Loss()(q_curr, target)
        
        self.optimizer.zero_grad()
        # AMP scaler：缩放梯度以降低 fp16 下溢风险；unscale 后再做梯度裁剪
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        if weights is not None:
            # 用 TD-error 更新优先级（PER）
            self.memory.update_priorities(idxs, td_errors.detach())

if __name__ == "__main__":
    from argparse import ArgumentParser
    p = ArgumentParser()
    # 中文：命令行参数尽量保持“短而直观”，更多默认行为由 `DQNVariantTrainer` 按阶段/variant 自动设定。
    p.add_argument(
        "--variant",
        type=str,
        default="dqn",
        choices=["dqn", "ddqn", "per", "dueling", "ddqn_per", "ddqn_per_dueling"],
        help="Variants: dqn, ddqn, per(=ddqn+per), dueling(=ddqn+per+dueling)",
    )
    p.add_argument("--steps", type=int, default=1000000)
    p.add_argument("--single", action="store_true")
    p.add_argument("--load", type=str, default=None)
    p.add_argument("--save", type=str, default="agent/checkpoints/dqn_best.pth")
    p.add_argument("--num-envs", type=int, default=None, help="Override number of parallel envs")
    p.add_argument("--sp-prob", type=float, default=0.6)
    p.add_argument("--finetune-lr-mult", type=float, default=0.5, help="LR multiplier when --load is used (default 0.5)")

    # Optional knobs (battle/single)
    p.add_argument("--eps-start", type=float, default=None, help="Override epsilon start (default depends on phase)")
    p.add_argument("--eps-min", type=float, default=None, help="Override epsilon min (default depends on phase)")

    # Self-play probability schedule: early high then low
    p.add_argument("--sp-prob-start", type=float, default=0.7, help="Self-play prob in early phase (default 0.7)")
    p.add_argument("--sp-prob-end", type=float, default=0.4, help="Self-play prob after switch (default 0.4)")
    p.add_argument("--sp-prob-frac", type=float, default=0.30, help="Switch point as fraction of total steps (default 0.30)")
    args = p.parse_args()

    v_norm = _normalize_variant(args.variant)
    p_dir = f"agent/pool/{v_norm}"
    
    cfg = TrainConfig(
        variant=v_norm,
        total_frames=args.steps,
        single_snake=args.single,
        load_path=args.load,
        save_path=args.save,
        num_envs_override=args.num_envs,
        self_play_prob=args.sp_prob,
        eps_start=args.eps_start,
        eps_min=args.eps_min,
        self_play_prob_start=args.sp_prob_start,
        self_play_prob_end=args.sp_prob_end,
        self_play_prob_frac=args.sp_prob_frac,
        finetune_lr_mult=args.finetune_lr_mult,
        pool_dir=p_dir
    )
    DQNVariantTrainer(cfg).train()
