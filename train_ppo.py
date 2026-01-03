"""train_ppo.py

【中文说明】
这是 BattleSnake 的 PPO 训练入口。

- 使用 `gymnasium.vector.AsyncVectorEnv` 做多进程并行采样。
- 默认会启用 AMP/TF32（GPU 上）以提升吞吐。
- 支持 battle 自博弈：当 `num_snakes > 1` 且 `self_play_prob > 0` 时，会从 `SelfPlayManager`
    的历史池中抽取对手模型参与对抗（并在训练过程中持续向池中写入快照）。

V18.1: Turbo Performance Edition
- Gymnasium AsyncVectorEnv (Parallel Processing)
- Pinned Memory & Pre-allocated Buffers
- torch.compile() (Fused Kernels)
- Mixed Precision Training (AMP)
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.amp import autocast, GradScaler

# V18.3: Silence CUDAGraph dynamic shape warnings for grouped inference
if hasattr(torch, '_inductor'):
    import torch._inductor.config as inductor_config
    inductor_config.triton.cudagraph_skip_dynamic_graphs = True

# V18.4: Enable TF32 for Tensor Core acceleration (Ampere+)
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

from agent.ppo import ActorCritic
from env.gymnasium_wrapper import make_gymnasium_env
from utils.self_play import SelfPlayManager


def log(msg: str) -> None:
    """统一日志输出（flush=True 便于训练时实时看到进度）。"""
    print(msg, flush=True)


def _atomic_torch_save(state_dict, path: Path) -> None:
    """原子方式保存权重文件。

    中文：先写入临时文件，再用 `os.replace` 原子替换目标文件，
    可避免训练中断导致 checkpoint 半写入/损坏。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state_dict, tmp)
    os.replace(tmp, path)


def train_ppo(
    *,
    num_envs: int,
    num_snakes: int,
    total_timesteps: int,
    load_path: Optional[str],
    checkpoint_path: str,
    pool_dir: str,
    self_play_prob: float,
    lr: float,
    target_kl: float,
    finetune_lr: Optional[float] = None,
    finetune_lr_mult: float = 0.25,
    finetune_target_kl: float = 0.030,
    rollout_steps: Optional[int] = None,
    update_epochs: Optional[int] = None,
    minibatch_size: Optional[int] = None,
    seed: int = 0,
) -> None:
    """训练 PPO。

    参数说明（中文，挑关键项）：
    - `num_envs`：并行环境数量（AsyncVectorEnv 的 worker 数），越大采样越快但更吃 CPU/RAM。
    - `num_snakes`：环境中的蛇数量；1 表示单蛇导航，>1 表示对战。
    - `total_timesteps`：总训练步数（跨所有环境累计）。
    - `load_path`：可选加载已有权重（Phase 2 微调常用）。
    - `checkpoint_path`：best 模型保存路径（会原子替换，防止中途写坏文件）。
    - `pool_dir`：自博弈历史池目录（保存/抽取对手快照）。
    - `self_play_prob`：对战时使用“模型对手”的概率；否则使用随机/规则对手（由 wrapper 决定）。
    - `lr`/`target_kl`：基础学习率 / KL 目标阈值（过大可能导致策略崩）。
    - `finetune_lr*`/`finetune_target_kl`：当 `load_path` 不为空时的微调超参（更稳）。
    - `rollout_steps`/`update_epochs`/`minibatch_size`：PPO 更新计算量的主要来源；调大更稳但更慢。
    - `seed`：随机种子（尽量可复现）。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 设备选择：优先 CUDA，否则 CPU。
    # AMP/TF32 只在 CUDA 下启用。

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f">>> [PPO-Turbo] Device={device} | snakes={num_snakes} | envs={num_envs} | steps={total_timesteps}")

    obs_dtype = torch.float16 if device.type == 'cuda' else torch.float32
    use_amp = (device.type == 'cuda')

    # 训练输出目录：
    # - `agent/checkpoints/`：best / final
    # - `pool_dir`：自博弈历史池（对手快照）

    Path("agent/checkpoints").mkdir(parents=True, exist_ok=True)
    sp_manager = SelfPlayManager(pool_dir)

    best_path = Path(checkpoint_path)
    best_score0 = -float('inf')
    best_win = -float('inf')
    best_ret = -float('inf')

    # V18.0: Base Model
    model = ActorCritic(vector_dim=28, grid_shape=(5, 20, 20), action_dim=4).to(device)
    
    if load_path:
        p = Path(load_path)
        if p.exists():
            log(f">>> [PPO] Loading weights from {load_path}")
            state_dict = torch.load(str(p), map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
        else:
            log(f">>> [Warning] load_path not found: {load_path}")

    # V18.0: torch.compile for massive throughput boost on modern GPUs
    # Note: Using mode='reduce-overhead' for best performance balance
    original_model = model
    if hasattr(torch, 'compile') and os.name != 'nt': # Windows support for compile is still spotty
        try:
            model = torch.compile(model, mode='reduce-overhead')
            log(">>> [PPO-Turbo] torch.compile() enabled.")
        except Exception as e:
            log(f">>> [PPO-Turbo] torch.compile() failed, falling back: {e}")
            model = original_model
    else:
        log(">>> [PPO-Turbo] torch.compile() skipped (not supported or Windows).")

    # Put an initial snapshot into the pool
    # 中文：即使一开始没有“best”，也先把初始策略放进池，
    # 这样 battle 自博弈时对手至少不是空池。
    sp_manager.add_model({k: v.detach().cpu() for k, v in original_model.state_dict().items()}, name="ppo_init")

    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))

    # PPO hyperparams (Recommended defaults: stable + lower memory/CPU pressure)
    # Note: PPO is update-heavy; reducing rollout/update cost prevents OOM/-9 kills on smaller machines.
    num_steps = (256 if num_snakes == 1 else 512) if rollout_steps is None else int(rollout_steps)
    update_epochs = 2 if update_epochs is None else int(update_epochs)
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.20
    ent_start = 0.08 if num_snakes == 1 else 0.12 
    ent_min = 0.02
    lr_init = lr
    target_kl = float(target_kl)
    vf_coef = 0.5            # Value function coefficient
    ent_coef_weight = 0.01   # Policy entropy coefficient weight
    max_grad_norm = 0.5      # Gradient clipping
    norm_adv = True          # Advantage normalization

    # 中文：fine-tune 模式（通常是 Phase2）
    # - 只要传入 load_path，就视为微调：降低有效 LR、提高熵、放宽 KL，避免策略崩。

    # V29.0: Auto-tune for fine-tuning mode (--load)
    # Prevent Policy Collapse in Phase 2 by boosting entropy and relaxing KL
    is_finetune = load_path is not None
    if is_finetune:
        if finetune_lr is not None:
            lr_init = float(finetune_lr)
        else:
            lr_init = lr * float(finetune_lr_mult)  # Lower LR for stable fine-tuning
        ent_start = 0.12         # V29.0: Boosted from 0.05 to force exploration in Battle Mode
        target_kl = float(finetune_target_kl)        # V29.0: Relaxed from 0.010 to allow adaptation to new dynamics
        log(f">>> [PPO-Turbo] Fine-tuning mode: LR={lr_init:.2e}, Ent={ent_start}, KL={target_kl}")

    # Keep user-provided num_envs; no auto-bump by default (stability-first).

    batch_size = num_envs * num_steps
    minibatch_size = 4096 if minibatch_size is None else int(minibatch_size)
    if minibatch_size > batch_size:
        minibatch_size = batch_size

    log(f">>> [PPO-Turbo] Initializing Parallel Environments...")

    use_model_opps = (num_snakes > 1 and float(self_play_prob) > 0.0)
    
    def env_creator():
        """创建单个环境实例（用于 AsyncVectorEnv worker）。

        中文：
        - single：更偏导航 shaping（min_food 更高，击杀/胜负奖励不重要）
        - battle：对齐 DQN battle 的 reward 设定，强调击杀/胜负与稳定存活
        """
        # Configuration for battle or navigation
        reward_cfg = {}
        if num_snakes == 1:
            reward_cfg = {
                "width": 20, "height": 20, "num_snakes": 1,
                "min_food": 5,
                "closer_reward": 0.05,
                "farther_penalty": -0.04,
                "food_reward": 1.0,
                "death_penalty": -3.0,
                "step_penalty": -0.01,
                "self_collision_penalty": -4.0,
            }
        else:
            # Align with DQN battle preset (train_dqn_variants.py PHASE 2)
            reward_cfg = {
                "closer_reward": 0.05,
                "farther_penalty": -0.04,
                "step_penalty": -0.01,
                "min_food": 2,
                "death_penalty": -3.0,
                "kill_reward": 2.0,
                "food_reward": 1.2,
                "self_collision_penalty": -4.0,
                "win_reward": 5.0,
                "loss_penalty": -2.0,
            }
        # Unpack reward_cfg to avoid conflicts with make_gymnasium_env defaults
        mfn = reward_cfg.pop("min_food", 2)
        gs = reward_cfg.pop("width", 20)
        ns = reward_cfg.pop("num_snakes", num_snakes) # Ensure no double-passing
        return make_gymnasium_env(
            num_snakes=ns,
            grid_size=gs,
            min_food=mfn,
            return_full_obs=use_model_opps,
            # Learn game rules via score delta (length-scaled scoring), with a bit of dense shaping retained.
            use_score_delta_reward=True,
            score_reward_coef=0.001,
            env_reward_coef=0.2,
            # Encourage purposeful dash (score gain soon after dash) instead of spamming.
            dash_effect_window=6,
            dash_success_bonus=0.2,
            dash_fail_penalty=-0.2,
            # Safety shaping: penalize obviously unsafe moves (wall/body) and risky/invalid dash.
            unsafe_move_penalty=-0.10,
            unsafe_dash_penalty=-0.10,
            invalid_dash_penalty=-0.02,
            **reward_cfg,
        )

    # Use AsyncVectorEnv for true multi-process parallelism
    envs = gym.vector.AsyncVectorEnv([env_creator for _ in range(num_envs)])
    
    log(f">>> [PPO-Turbo] Environments Ready. Resetting...")
    obs, info = envs.reset(seed=seed)
    
    # Store opponent models for self-play
    # { env_idx: [model_path_snake1, model_path_snake2, ... ] }
    opp_model_paths: List[List[Optional[str]]] = [[None] * num_snakes for _ in range(num_envs)]
    
    def assign_opps(e_idx: int):
        """为某个 env 的对手分配策略。

        规则：对每个对手蛇（1..num_snakes-1），以 `self_play_prob` 概率从历史池采样模型；
        否则为 None（表示随机对手）。
        """
        if num_snakes <= 1: return
        for s_idx in range(1, num_snakes):
            if random.random() < self_play_prob:
                m = sp_manager.sample_model(chaos_prob=0.1)
                opp_model_paths[e_idx][s_idx] = str(m) if m else None
            else:
                opp_model_paths[e_idx][s_idx] = None
    
    for i in range(num_envs):
        if use_model_opps:
            assign_opps(i)

    # Model cache for opponents to avoid reloading
    opp_model_cache: Dict[str, Tuple[float, nn.Module]] = {}

    def get_cached_model(path: str) -> Optional[nn.Module]:
        """按文件 mtime 缓存对手模型，避免重复 load。

        返回：
        - nn.Module：可用的对手模型（eval 模式、requires_grad=False）
        - None：文件不存在/损坏/加载失败
        """
        try:
            mtime = os.stat(path).st_mtime
        except OSError:
            return None
        if path in opp_model_cache and opp_model_cache[path][0] == mtime:
            return opp_model_cache[path][1]
        
        try:
            m = ActorCritic(vector_dim=28, grid_shape=(5, 20, 20), action_dim=4).to(device).eval()
            sd = torch.load(path, map_location=device, weights_only=True)
            m.load_state_dict(sd)
            for p in m.parameters(): p.requires_grad = False
            opp_model_cache[path] = (mtime, m)
            if len(opp_model_cache) > 40: del opp_model_cache[next(iter(opp_model_cache))]
            return m
        except: return None

    # V18.0: Pre-allocate Rollout Buffers (directly on device)
    # PERF: Keep grid as uint8 on GPU to avoid CPU-side uint8->fp16 conversion on every step.
    # Cast to fp16 on GPU via a reusable temp buffer right before model forward.
    b_obs_grid_u8 = torch.zeros((num_steps, num_envs, 5, 20, 20), dtype=torch.uint8, device=device)
    b_obs_vec = torch.zeros((num_steps, num_envs, 28), dtype=obs_dtype, device=device)
    b_actions = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)
    b_logprobs = torch.zeros((num_steps, num_envs), device=device)
    b_rewards = torch.zeros((num_steps, num_envs), device=device)
    b_dones = torch.zeros((num_steps, num_envs), device=device)
    b_values = torch.zeros((num_steps, num_envs), device=device)

    # Reusable cast buffer for grid (avoid per-step allocations)
    tmp_grid = torch.empty((num_envs, 5, 20, 20), dtype=obs_dtype, device=device)

    # Rolling Return Tracking
    ep_returns: List[float] = []
    running_return = np.zeros(num_envs, dtype=np.float32)

    # Battle diagnostics (win-rate / score) for Phase 2
    ep_wins: List[int] = []
    ep_end_scores: List[int] = []

    global_step = 0
    update = 0
    total_snakes = num_envs * num_snakes

    log(">>> [PPO-Turbo] Setup finished. Starting training loop...")

    while global_step < total_timesteps:
        update += 1
        # ====== 1) 退火（Annealing）======
        # 中文：按 update 进度逐步降低学习率，并把 entropy 系数从 ent_start 退火到 ent_min。
        # 这样前期探索更强，后期更稳定。
        frac = max(0.0, 1.0 - (update - 1) / (total_timesteps / (num_envs * num_steps)))
        lr_now = lr_init * (0.1 + 0.9 * frac)
        for pg in optimizer.param_groups: pg["lr"] = lr_now
        ent_coef = max(0.02, ent_min + (ent_start - ent_min) * (0.10 + 0.90 * frac))

        model.eval()
        for step in range(num_steps):
            global_step += num_envs
            
            # ====== 2) 采样（Rollout 收集）======
            # obs 形状：
            # - obs['grid']   : (num_envs, 5, 20, 20) uint8
            # - obs['vector'] : (num_envs, 28) float32
            # 中文：grid 保持 uint8 存 GPU，forward 前再 cast，可减少 CPU 侧转换开销。
            with torch.inference_mode():
                cpu_grid = torch.from_numpy(obs['grid'])
                cpu_vec = torch.from_numpy(obs['vector'])
                # Avoid CPU-side dtype conversion for grid: copy as uint8, cast on GPU.
                b_obs_grid_u8[step].copy_(cpu_grid)
                b_obs_vec[step].copy_(cpu_vec)

                # Get Actions for Learner
                with autocast('cuda', enabled=use_amp):
                    tmp_grid.copy_(b_obs_grid_u8[step])
                    a, lp, _, v = model.get_action_and_value(tmp_grid, b_obs_vec[step])
                b_actions[step].copy_(a)
                b_logprobs[step].copy_(lp)
                b_values[step].copy_(v.squeeze(-1))

            # ====== 3) 对手动作生成（battle 模式）======
            # - learner（蛇0）的动作来自 PPO policy。
            # - 对手蛇：要么随机，要么从自博弈池抽模型推断。
            # - 为加速：按“同一个模型文件路径”把对手索引分组，批量推断（减少 forward 次数）。
            all_actions = np.zeros((num_envs, num_snakes), dtype=np.int32)
            all_actions[:, 0] = a.cpu().numpy()
            
            if num_snakes > 1:
                if use_model_opps and ("full_obs_grids" in info) and ("full_obs_vecs" in info):
                    # Only move opponent observations needed for model inference.
                    # Random opponents don't need GPU copies.
                    full_grids = info["full_obs_grids"].reshape(total_snakes, 5, 20, 20)
                    full_vecs = info["full_obs_vecs"].reshape(total_snakes, -1)

                    groups: Dict[nn.Module, List[int]] = {}
                    for e_idx in range(num_envs):
                        for s_idx in range(1, num_snakes):
                            idx = e_idx * num_snakes + s_idx
                            m_path = opp_model_paths[e_idx][s_idx]
                            if not m_path:
                                all_actions[e_idx, s_idx] = random.randint(0, 3)
                                continue
                            m = get_cached_model(m_path)
                            if not m:
                                all_actions[e_idx, s_idx] = random.randint(0, 3)
                                continue
                            groups.setdefault(m, []).append(idx)

                    # Per-model batched inference; only copy the required rows.
                    for m, idx_list in groups.items():
                        with torch.inference_mode(), autocast('cuda', enabled=use_amp):
                            g = torch.as_tensor(full_grids[idx_list], dtype=obs_dtype, device=device)
                            v = torch.as_tensor(full_vecs[idx_list], dtype=obs_dtype, device=device)
                            logits, _ = m(g, v)
                            acts = logits.argmax(dim=1).to(dtype=torch.int32).cpu().numpy()
                            for i, idx in enumerate(idx_list):
                                e_i = idx // num_snakes
                                s_i = idx % num_snakes
                                all_actions[e_i, s_i] = int(acts[i])
                else:
                    for e_idx in range(num_envs):
                        for s_idx in range(1, num_snakes):
                            all_actions[e_idx, s_idx] = random.randint(0, 3)

            # ====== 4) 与环境交互（AsyncVectorEnv.step）======
            # - AsyncVectorEnv 的每个 worker 都是子进程，因此不能通过“直接调用 wrapper 方法”传对手动作。
            # - 本项目 wrapper 约定：如果 `step()` 收到的是 array/list，则按“全体蛇动作”解释。
            #   所以这里直接把 `all_actions`（num_envs, num_snakes）传进去。
            next_obs, next_rewards, next_terminated, next_truncated, next_info = envs.step(all_actions)
            
            # Record rewards and dones for learner (snake 0)
            b_rewards[step].copy_(torch.as_tensor(next_rewards, device=device))
            b_dones[step].copy_(torch.as_tensor(next_terminated, device=device))
            
            # Episode Tracking
            # 中文：收集每个 env 的 episode return；battle 时额外统计 win-rate/score0。
            for e_idx in range(num_envs):
                running_return[e_idx] += next_rewards[e_idx]
                if next_terminated[e_idx] or next_truncated[e_idx]:
                    ep_returns.append(float(running_return[e_idx]))
                    running_return[e_idx] = 0.0

                    # Extra diagnostics for battle mode (meaningful across policies)
                    if num_snakes > 1 and isinstance(next_info, dict):
                        try:
                            winner = int(next_info.get("winner_idx", [-1])[e_idx])
                        except Exception:
                            winner = -1
                        ep_wins.append(1 if winner == 0 else 0)
                        try:
                            ep_end_scores.append(int(next_info.get("score0", [0])[e_idx]))
                        except Exception:
                            ep_end_scores.append(0)
                    if use_model_opps:
                        assign_opps(e_idx) # Re-assign for next episode

            obs, info = next_obs, next_info

        # ====== 5) Bootstrap value（用于 GAE 的最后一项）======
        with torch.inference_mode(), autocast('cuda', enabled=use_amp):
            t_grid_u8 = torch.as_tensor(obs['grid'], dtype=torch.uint8, device=device)
            t_grid = t_grid_u8.to(dtype=obs_dtype)
            t_vec = torch.as_tensor(obs['vector'], dtype=obs_dtype, device=device)
            next_value = model.get_value(t_grid, t_vec).squeeze(-1)

        # ====== 6) GAE（Generalized Advantage Estimation）======
        # 中文：优势函数用递推计算：
        #   $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$
        #   $A_t = \delta_t + \gamma\lambda A_{t+1}$（遇到 terminal 则截断）
        advantages = torch.zeros_like(b_rewards)
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                # V18.2: Use next_terminated (the done flag associated with the bootstrap state next_obs)
                nextnonterminal = 1.0 - torch.as_tensor(next_terminated, dtype=torch.float32, device=device)
                nextvalues = next_value
            else:
                # V18.2 FIX: Use b_dones[t] which corresponds to the state transition at step t
                # Old code used b_dones[t+1], which was off-by-one
                nextnonterminal = 1.0 - b_dones[t]
                nextvalues = b_values[t + 1]
            delta = b_rewards[t] + gamma * nextvalues * nextnonterminal - b_values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + b_values

        # ====== 7) 展平 batch（num_steps*num_envs）======
        flat_grid_u8 = b_obs_grid_u8.reshape(-1, 5, 20, 20)
        flat_vec = b_obs_vec.reshape(-1, 28)
        flat_actions = b_actions.reshape(-1)
        flat_logprobs = b_logprobs.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_values = b_values.reshape(-1)

        # Advantage 归一化（可显著提升稳定性）
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        # ====== 8) PPO 优化（多 epoch、多 minibatch）======
        # 中文：每个 minibatch 计算 policy loss / value loss / entropy bonus。
        # KL 超过阈值时 early stop，避免策略更新过猛。
        model.train()
        inds = np.arange(batch_size)
        stop_early = False
        pg_loss_sum = 0.0
        v_loss_sum = 0.0
        ent_sum = 0.0
        approx_kl_sum = 0.0
        clipfrac_sum = 0.0
        n_minibatches = 0
        for epoch in range(update_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, minibatch_size):
                mb = inds[start : start + minibatch_size]
                
                with autocast('cuda', enabled=(device.type == 'cuda')):
                    mb_grid = flat_grid_u8[mb].to(dtype=obs_dtype)
                    _, newlogprob, entropy, newvalue = model.get_action_and_value(
                        mb_grid, flat_vec[mb], flat_actions[mb]
                    )
                    newvalue = newvalue.squeeze(-1)
                    logratio = newlogprob - flat_logprobs[mb]
                    ratio = logratio.exp()

                    # Policy Loss
                    pg_loss1 = -flat_advantages[mb] * ratio
                    pg_loss2 = -flat_advantages[mb] * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value Loss
                    v_loss = 0.5 * (newvalue - flat_returns[mb]).pow(2).mean()

                    # Entropy Loss
                    ent_loss = entropy.mean()

                    loss = pg_loss + vf_coef * v_loss - ent_coef * ent_loss

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()

                    pg_loss_sum += float(pg_loss.detach().item())
                    v_loss_sum += float(v_loss.detach().item())
                    ent_sum += float(ent_loss.detach().item())
                    approx_kl_sum += float(approx_kl.detach().item())
                    clipfrac_sum += float(clipfrac.detach().item())
                    n_minibatches += 1

                    if approx_kl > target_kl:
                        stop_early = True
                        break
            if stop_early:
                break

        # ====== 9) 诊断指标（explained variance 等）======
        with torch.no_grad():
            v_y = flat_returns
            v_y_pred = flat_values
            var_y = torch.var(v_y, unbiased=False)
            explained_var = torch.tensor(0.0, device=device)
            if float(var_y.item()) > 1e-8:
                explained_var = 1.0 - torch.var(v_y - v_y_pred, unbiased=False) / (var_y + 1e-8)

        mb_denom = max(1, int(n_minibatches))
        pg_loss_mean = pg_loss_sum / mb_denom
        v_loss_mean = v_loss_sum / mb_denom
        ent_mean = ent_sum / mb_denom
        kl_mean = approx_kl_sum / mb_denom
        clipfrac_mean = clipfrac_sum / mb_denom

        # ====== 10) 日志与 best 保存 ======
        # battle：优先按 score0 提升保存，其次看 win_rate。
        # single：按 avg return 提升保存。
        if update <= 10 or update % 10 == 0:
            avg_ret = np.mean(ep_returns[-100:]) if ep_returns else 0
            if num_snakes > 1:
                win_rate = (np.mean(ep_wins[-200:]) if ep_wins else 0.0) * 100.0
                avg_score0 = np.mean(ep_end_scores[-200:]) if ep_end_scores else 0.0
                log(
                    f">>> [PPO-Turbo] Upd {update} | Step {global_step}/{total_timesteps} ({global_step/total_timesteps:.1%}) "
                    f"| Ret {avg_ret:.2f} | Win% {win_rate:.1f} | Score0 {avg_score0:.1f} "
                    f"| pg_loss {pg_loss_mean:.3f} | v_loss {v_loss_mean:.3f} | ent {ent_mean:.3f} "
                    f"| kl {kl_mean:.4f}/{target_kl:.4f} | clipfrac {clipfrac_mean:.3f} | ev {float(explained_var.item()):.3f} "
                    f"| lr {lr_now:.2e} | ent_coef {ent_coef:.3f}" + (" | early_stop" if stop_early else "")
                )

                # 保存 best（battle）：优先 Score0，其次 Win%
                improved = False
                if avg_score0 > best_score0 + 5.0:
                    improved = True
                elif abs(avg_score0 - best_score0) <= 5.0 and win_rate > best_win + 1.0:
                    improved = True
                elif best_score0 == -float('inf') and (ep_end_scores or ep_wins):
                    improved = True
                if improved:
                    best_score0 = float(avg_score0)
                    best_win = float(win_rate)
                    cpu_state = {k: v.detach().cpu() for k, v in original_model.state_dict().items()}
                    _atomic_torch_save(cpu_state, best_path)
                    log(f">>> [PPO-Turbo][Best] Saved -> {best_path} | Win% {best_win:.1f} | Score0 {best_score0:.1f}")
            else:
                log(
                    f">>> [PPO-Turbo] Upd {update} | Step {global_step}/{total_timesteps} ({global_step/total_timesteps:.1%}) "
                    f"| Ret {avg_ret:.2f} "
                    f"| pg_loss {pg_loss_mean:.3f} | v_loss {v_loss_mean:.3f} | ent {ent_mean:.3f} "
                    f"| kl {kl_mean:.4f}/{target_kl:.4f} | clipfrac {clipfrac_mean:.3f} | ev {float(explained_var.item()):.3f} "
                    f"| lr {lr_now:.2e} | ent_coef {ent_coef:.3f}" + (" | early_stop" if stop_early else "")
                )

                # 保存 best（single）：按 avg return
                improved = False
                if avg_ret > best_ret + 0.5:
                    improved = True
                elif best_ret == -float('inf') and ep_returns:
                    improved = True
                if improved:
                    best_ret = float(avg_ret)
                    cpu_state = {k: v.detach().cpu() for k, v in original_model.state_dict().items()}
                    _atomic_torch_save(cpu_state, best_path)
                    log(f">>> [PPO-Turbo][Best] Saved -> {best_path} | Ret {best_ret:.2f}")

        # ====== 11) 周期性写入自博弈池 ======
        # 中文：不一定是 best，但能增加对手多样性（训练时作为历史对手采样）。
        if global_step >= 500_000 and global_step % 500_000 < num_envs * num_steps:
             cpu_state = {k: v.detach().cpu() for k, v in original_model.state_dict().items()}
             sp_manager.add_model(cpu_state, name=f"ppo_step_{global_step}")

    # ====== 12) 训练结束：确保 best 存在，并写 final 快照 ======
    if not best_path.exists():
        cpu_state = {k: v.detach().cpu() for k, v in original_model.state_dict().items()}
        _atomic_torch_save(cpu_state, best_path)
        log(f">>> [PPO-Turbo][Best] (fallback) Saved -> {best_path}")

    # Final Save
    final_path = best_path.with_suffix(".final.pth")
    cpu_state = {k: v.detach().cpu() for k, v in original_model.state_dict().items()}
    _atomic_torch_save(cpu_state, final_path)
    log(f">>> [PPO-Turbo] Finished. Saved to {final_path}")
    envs.close()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--single", action="store_true")
    p.add_argument("--load", type=str, default=None)
    p.add_argument("--steps", type=int, default=10_000_000) # V18.3: 10M for deep mastery
    p.add_argument("--envs", type=int, default=None, help="Number of parallel envs (overrides default).")
    p.add_argument("--rollout-steps", type=int, default=None, help="Rollout steps per env before each PPO update.")
    p.add_argument("--update-epochs", type=int, default=None, help="PPO update epochs per rollout (more = slower, more reuse).")
    p.add_argument("--minibatch-size", type=int, default=None, help="Minibatch size for PPO updates.")
    p.add_argument("--lr", type=float, default=None, help="Base learning rate (before fine-tune scaling).")
    p.add_argument("--target-kl", type=float, default=0.015, help="KL early-stop threshold for PPO updates.")
    p.add_argument("--finetune-lr", type=float, default=None, help="Override fine-tune LR directly (only when --load is set).")
    p.add_argument("--finetune-lr-mult", type=float, default=0.25, help="Fine-tune LR multiplier applied to --lr (only when --load is set, ignored if --finetune-lr is set).")
    p.add_argument("--finetune-target-kl", type=float, default=0.030, help="Fine-tune KL early-stop threshold (only when --load is set).")
    p.add_argument("--self-play-prob", type=float, default=None, help="Battle self-play probability for opponent snakes (0 disables self-play).")
    args = p.parse_args()

    # Recommended defaults (stable across GPUs/CPUs)
    num_snakes = 1 if args.single else 4
    num_envs = (64 if args.single else 64) if args.envs is None else int(args.envs)
    base_lr = (2.0e-4 if args.single else 1.5e-4) if args.lr is None else float(args.lr)
    ckpt = "agent/checkpoints/ppo_best.pth" if args.single else "agent/checkpoints/ppo_battle_best.pth"

    self_play_prob = (0.3 if num_snakes > 1 else 0.0) if args.self_play_prob is None else float(args.self_play_prob)

    train_ppo(
        num_envs=num_envs,
        num_snakes=num_snakes,
        total_timesteps=int(args.steps),
        load_path=args.load,
        checkpoint_path=ckpt,
        pool_dir="agent/pool/ppo",
        self_play_prob=self_play_prob,
        lr=base_lr,
        target_kl=float(args.target_kl),
        finetune_lr=args.finetune_lr,
        finetune_lr_mult=float(args.finetune_lr_mult),
        finetune_target_kl=float(args.finetune_target_kl),
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
    )
