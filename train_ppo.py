"""train_ppo.py

PPO training for BattleSnake.
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
    print(msg, flush=True)


def _atomic_torch_save(state_dict, path: Path) -> None:
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
    seed: int = 0,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f">>> [PPO-Turbo] Device={device} | snakes={num_snakes} | envs={num_envs} | steps={total_timesteps}")

    Path("agent/checkpoints").mkdir(parents=True, exist_ok=True)
    sp_manager = SelfPlayManager(pool_dir)

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
    sp_manager.add_model({k: v.detach().cpu() for k, v in original_model.state_dict().items()}, name="ppo_init")

    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))

    # PPO hyperparams (Optimized for Throughput)
    num_steps = 512 if num_snakes == 1 else 1024 
    update_epochs = 4 # V18.2: Increased to 4 for better sample reuse
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.20
    ent_start = 0.08 if num_snakes == 1 else 0.12 
    ent_min = 0.02
    lr_init = lr
    target_kl = 0.015 
    vf_coef = 0.5            # Value function coefficient
    ent_coef_weight = 0.01   # Policy entropy coefficient weight
    max_grad_norm = 0.5      # Gradient clipping
    norm_adv = True          # Advantage normalization

    # V29.0: Auto-tune for fine-tuning mode (--load)
    # Prevent Policy Collapse in Phase 2 by boosting entropy and relaxing KL
    is_finetune = load_path is not None
    if is_finetune:
        lr_init = lr * 0.25      # Lower LR for stable fine-tuning
        ent_start = 0.12         # V29.0: Boosted from 0.05 to force exploration in Battle Mode
        target_kl = 0.030        # V29.0: Relaxed from 0.010 to allow adaptation to new dynamics
        log(f">>> [PPO-Turbo] Fine-tuning mode: LR={lr_init:.2e}, Ent={ent_start}, KL={target_kl}")

    # V19.0: Increase num_envs for multi-snake to boost FPS
    if num_snakes > 1:
        num_envs = 96 # Increased from 64
        log(f">>> [PPO-Turbo] Multi-snake mode: Increased envs to {num_envs} for FPS")

    batch_size = num_envs * num_steps
    minibatch_size = 2048 
    if minibatch_size > batch_size:
        minibatch_size = batch_size

    log(f">>> [PPO-Turbo] Initializing Parallel Environments...")
    
    def env_creator():
        # Configuration for battle or navigation
        reward_cfg = {}
        if num_snakes == 1:
            reward_cfg = {
                "width": 20, "height": 20, "num_snakes": 1,
                "min_food": 5, # V18.3: Increased density for better signal
                "closer_reward": 0.15, # V18.3: Stronger guidance
                "farther_penalty": -0.12,
                "food_reward": 50.0, # V18.3: Stronger positive reinforcement
                "death_penalty": -50.0,
                "step_penalty": -0.05, # V18.3: Discourage idling/looping
                "self_collision_penalty": -60.0,
            }
        else:
            # Align with DQN battle preset (train_dqn_variants.py PHASE 2)
            reward_cfg = {
                "closer_reward": 0.15,
                "farther_penalty": -0.10,
                "step_penalty": -0.05,
                "min_food": 2,
                "death_penalty": -100.0,
                "kill_reward": 150.0,
                "food_reward": 80.0,
                "self_collision_penalty": -150.0,
                "win_reward": 500.0,
                "loss_penalty": -200.0,
            }
        # Unpack reward_cfg to avoid conflicts with make_gymnasium_env defaults
        mfn = reward_cfg.pop("min_food", 2)
        gs = reward_cfg.pop("width", 20)
        ns = reward_cfg.pop("num_snakes", num_snakes) # Ensure no double-passing
        return make_gymnasium_env(num_snakes=ns, grid_size=gs, min_food=mfn, **reward_cfg)

    # Use AsyncVectorEnv for true multi-process parallelism
    envs = gym.vector.AsyncVectorEnv([env_creator for _ in range(num_envs)])
    
    log(f">>> [PPO-Turbo] Environments Ready. Resetting...")
    obs, info = envs.reset(seed=seed)
    
    # Store opponent models for self-play
    # { env_idx: [model_path_snake1, model_path_snake2, ... ] }
    opp_model_paths: List[List[Optional[str]]] = [[None] * num_snakes for _ in range(num_envs)]
    
    def assign_opps(e_idx: int):
        if num_snakes <= 1: return
        for s_idx in range(1, num_snakes):
            if random.random() < self_play_prob:
                m = sp_manager.sample_model(chaos_prob=0.1)
                opp_model_paths[e_idx][s_idx] = str(m) if m else None
            else:
                opp_model_paths[e_idx][s_idx] = None
    
    for i in range(num_envs):
        assign_opps(i)

    # Model cache for opponents to avoid reloading
    opp_model_cache: Dict[str, Tuple[float, nn.Module]] = {}

    def get_cached_model(path: str) -> Optional[nn.Module]:
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
    # Using float32 for observation grids to match network input
    # (Memory usage: 128 envs * 512 steps * 5*20*20 * 4 bytes ≈ 130MB, well within GPU memory)
    b_obs_grid = torch.zeros((num_steps, num_envs, 5, 20, 20), device=device)
    b_obs_vec = torch.zeros((num_steps, num_envs, 28), device=device)
    b_actions = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)
    b_logprobs = torch.zeros((num_steps, num_envs), device=device)
    b_rewards = torch.zeros((num_steps, num_envs), device=device)
    b_dones = torch.zeros((num_steps, num_envs), device=device)
    b_values = torch.zeros((num_steps, num_envs), device=device)

    # Rolling Return Tracking
    ep_returns: List[float] = []
    running_return = np.zeros(num_envs, dtype=np.float32)

    global_step = 0
    update = 0
    start_time = time.time()
    total_snakes = num_envs * num_snakes

    log(">>> [PPO-Turbo] Setup finished. Starting training loop...")

    while global_step < total_timesteps:
        update += 1
        # Annealing
        frac = max(0.0, 1.0 - (update - 1) / (total_timesteps / (num_envs * num_steps)))
        lr_now = lr_init * (0.1 + 0.9 * frac)
        for pg in optimizer.param_groups: pg["lr"] = lr_now
        ent_coef = max(0.02, ent_min + (ent_start - ent_min) * (0.10 + 0.90 * frac))

        model.eval()
        for step in range(num_steps):
            global_step += num_envs
            
            # Map observations to tensors (Optimized with pre-allocated buffer)
            # obs['grid'] is (num_envs, 5, 20, 20), obs['vector'] is (num_envs, 28)
            with torch.no_grad():
                t_grid = torch.as_tensor(obs['grid'], dtype=torch.float32, device=device)
                t_vec = torch.as_tensor(obs['vector'], dtype=torch.float32, device=device)
                
                b_obs_grid[step].copy_(t_grid)
                b_obs_vec[step].copy_(t_vec)
                
                # Get Actions for Learner
                a, lp, _, v = model.get_action_and_value(t_grid, t_vec)
                b_actions[step].copy_(a)
                b_logprobs[step].copy_(lp)
                b_values[step].copy_(v.squeeze(-1))

            # Handle Opponents (Grouped Inference - V30.0 High Speed Matrix mode)
            all_actions = np.zeros((num_envs, num_snakes), dtype=np.int32)
            all_actions[:, 0] = a.cpu().numpy()
            
            if num_snakes > 1:
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
                    with torch.no_grad():
                        g = torch.as_tensor(full_grids[idx_list], dtype=torch.float32, device=device)
                        v = torch.as_tensor(full_vecs[idx_list], dtype=torch.float32, device=device)
                        logits, _ = m(g, v)
                        acts = logits.argmax(dim=1).to(dtype=torch.int32).cpu().numpy()
                        for i, idx in enumerate(idx_list):
                            e_i = idx // num_snakes
                            s_i = idx % num_snakes
                            all_actions[e_i, s_i] = int(acts[i])

            # Step Environments (Async)
            # Note: We need to pass the multi-agent actions. 
            # AsyncVectorEnv.step expects (num_envs, action_space)
            # Our gymnasium wrapper expects [learner_action, *opp_actions]
            # Since we provide ONLY snake 0 as action_space in wrapper,
            # we use the set_opponent_actions trick or we can modify wrapper to take multi-actions.
            # Here, let's use the simplest path: modify envs if they are in same process, 
            # BUT they are in sub-processes. So we MUST pass ALL actions through the step call.
            
            # WORKAROUND: In our Gymnasium wrapper, if we pass an array to step, 
            # it uses it for all snakes.
            next_obs, next_rewards, next_terminated, next_truncated, next_info = envs.step(all_actions)
            
            # Record rewards and dones for learner (snake 0)
            b_rewards[step].copy_(torch.as_tensor(next_rewards, device=device))
            b_dones[step].copy_(torch.as_tensor(next_terminated, device=device))
            
            # Episode Tracking
            for e_idx in range(num_envs):
                running_return[e_idx] += next_rewards[e_idx]
                if next_terminated[e_idx] or next_truncated[e_idx]:
                    ep_returns.append(float(running_return[e_idx]))
                    running_return[e_idx] = 0.0
                    assign_opps(e_idx) # Re-assign for next episode

            obs, info = next_obs, next_info

        # Bootstrap Value
        with torch.no_grad():
            t_grid = torch.as_tensor(obs['grid'], dtype=torch.float32, device=device)
            t_vec = torch.as_tensor(obs['vector'], dtype=torch.float32, device=device)
            next_value = model.get_value(t_grid, t_vec).squeeze(-1)

        # GAE
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

        # Flatten Buffers
        flat_grid = b_obs_grid.reshape(-1, 5, 20, 20)
        flat_vec = b_obs_vec.reshape(-1, 28)
        flat_actions = b_actions.reshape(-1)
        flat_logprobs = b_logprobs.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_values = b_values.reshape(-1)

        # Norm advantages
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        # Optimization
        model.train()
        inds = np.arange(batch_size)
        stop_early = False
        for epoch in range(update_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, minibatch_size):
                mb = inds[start : start + minibatch_size]
                
                with autocast('cuda', enabled=(device.type == 'cuda')):
                    _, newlogprob, entropy, newvalue = model.get_action_and_value(
                        flat_grid[mb], flat_vec[mb], flat_actions[mb]
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
                    if approx_kl > target_kl:
                        stop_early = True
                        break
            if stop_early:
                break

        # Logging
        if update <= 10 or update % 10 == 0:
            dt = time.time() - start_time
            fps = (update * num_envs * num_steps) / dt
            avg_ret = np.mean(ep_returns[-100:]) if ep_returns else 0
            log(f">>> [PPO-Turbo] Upd {update} | Step {global_step}/{total_timesteps} ({global_step/total_timesteps:.1%}) | FPS {fps:.0f} | Ret {avg_ret:.2f}")

        # Periodic Saving & Pool Update
        if global_step >= 500_000 and global_step % 500_000 < num_envs * num_steps:
             cpu_state = {k: v.detach().cpu() for k, v in original_model.state_dict().items()}
             sp_manager.add_model(cpu_state, name=f"ppo_step_{global_step}")

    # Final Save
    final_path = Path(checkpoint_path).with_suffix(".final.pth")
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
    args = p.parse_args()

    # V18.0: Turbo Hyperparams
    num_snakes = 1 if args.single else 4
    num_envs = 128 if args.single else 64 
    base_lr = 2.0e-4 if args.single else 1.5e-4
    ckpt = "agent/checkpoints/ppo_best.pth" if args.single else "agent/checkpoints/ppo_battle_best.pth"

    train_ppo(
        num_envs=num_envs,
        num_snakes=num_snakes,
        total_timesteps=int(args.steps),
        load_path=args.load,
        checkpoint_path=ckpt,
        pool_dir="agent/pool/ppo",
        self_play_prob=0.3 if num_snakes > 1 else 0.0,
        lr=base_lr,
    )
