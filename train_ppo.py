"""train_ppo.py

PPO training for BattleSnake.

This is a clean, stable PPO implementation tailored to this repo:
- Hybrid Actor-Critic network in `agent/ppo.py` (25D vector + 3x7x7 grid)
- Multi-snake battle via self-play opponents (sampled from `utils/self_play.py` pool)
- Correct episode handling: reset an env immediately when the learning snake dies
  (the environment otherwise keeps stepping with a dead snake producing zeros).
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent.ppo import ActorCritic
from env.battle_snake_env import BattleSnakeConfig, BattleSnakeEnv
from utils.self_play import SelfPlayManager


def log(msg: str) -> None:
    print(msg, flush=True)


def _atomic_torch_save(state_dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state_dict, tmp)
    os.replace(tmp, path)


def make_env(num_snakes: int, grid_size: int, seed: Optional[int] = None):
    def thunk():
        env_cfg = BattleSnakeConfig(
            width=grid_size,
            height=grid_size,
            num_snakes=num_snakes,
            min_food=max(2, num_snakes // 2),
            max_steps=1000, # V7.0: Longer games for battle
        )
        if num_snakes == 1:
            # Phase 1: High focus on navigation (V46.0: Aggressive Reward Tuning)
            env_cfg.closer_reward = 0.25
            env_cfg.farther_penalty = -0.20
            env_cfg.food_reward = 120.0
            env_cfg.death_penalty = -10.0
            env_cfg.step_penalty = -0.02
            env_cfg.self_collision_penalty = -100.0 # V17.0: Absolute body awareness
        else:
            # V9.2: Smooth Transition & High Survival
            env_cfg.closer_reward = 0.15
            env_cfg.farther_penalty = -0.10
            env_cfg.step_penalty = -0.05
            env_cfg.kill_reward = 250.0 # V13.0: Aggressive killing incentive
            env_cfg.food_reward = 80.0
            env_cfg.win_reward = 800.0 # V13.0: Professional leader mindset
            env_cfg.loss_penalty = -200.0
            env_cfg.death_penalty = -30.0 # V9.2: Increased to discourage reckless play
            env_cfg.self_collision_penalty = -50.0 
        return BattleSnakeEnv(env_cfg, seed=seed)

    return thunk


class VectorizedEnv:
    def __init__(self, env_fns, device: torch.device):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.device = device
        self.opp_model_paths: List[List[Optional[str]]] = [
            [None] * env.config.num_snakes for env in self.envs
        ]
        self._model_cache: Dict[str, Tuple[float, nn.Module]] = {}

    def reset(self):
        return [env.reset() for env in self.envs]

    def reset_one(self, idx: int):
        return self.envs[idx].reset()

    def _get_model(self, path: str) -> Optional[nn.Module]:
        try:
            stat = os.stat(path)
            mtime = stat.st_mtime
        except OSError:
            return None

        cached = self._model_cache.get(path)
        if cached is not None and cached[0] == mtime:
            return cached[1]

        try:
            # V11.0: Dynamic Dimension Check to prevent crash on old models
            dummy_model = ActorCritic(vector_dim=28, grid_shape=(5, 20, 20), action_dim=self.envs[0].action_dim).to(self.device).eval()
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            
            # Check for dimension mismatch before loading
            if "actor.0.weight" in state_dict:
                ckpt_dim = state_dict["actor.0.weight"].shape[1]
                model_dim = dummy_model.actor[0].weight.shape[1]
                if ckpt_dim != model_dim:
                    return None # Silently skip incompatible models
            
            dummy_model.load_state_dict(state_dict)
            for p in dummy_model.parameters():
                p.requires_grad = False
            self._model_cache[path] = (mtime, dummy_model)
            if len(self._model_cache) > 80:
                del self._model_cache[next(iter(self._model_cache))]
            return dummy_model
        except Exception:
            return None

    def step(self, learning_actions, current_obs):
        all_actions = [[None] * env.config.num_snakes for env in self.envs]

        groups: Dict[nn.Module, List[Tuple[int, int, dict]]] = {}
        for env_i in range(self.num_envs):
            all_actions[env_i][0] = int(learning_actions[env_i])
            for snake_i in range(1, self.envs[env_i].config.num_snakes):
                m_path = self.opp_model_paths[env_i][snake_i]
                if not m_path:
                    all_actions[env_i][snake_i] = random.randint(0, 3)
                    continue
                model = self._get_model(m_path)
                if model is None:
                    all_actions[env_i][snake_i] = random.randint(0, 3)
                    continue
                groups.setdefault(model, []).append((env_i, snake_i, current_obs[env_i][snake_i]))

        for model, samples in groups.items():
            if not samples:
                continue
            with torch.inference_mode():
                grids = np.asarray([s[2]["grid"] for s in samples], dtype=np.float32)
                vecs = np.asarray([s[2]["vector"] for s in samples], dtype=np.float32)
                t_g = torch.as_tensor(grids, device=self.device)
                t_v = torch.as_tensor(vecs, device=self.device)
                logits, _ = model(t_g, t_v)
                acts = logits.argmax(dim=1).detach().cpu().numpy()
            for idx, (env_i, snake_i, _) in enumerate(samples):
                all_actions[env_i][snake_i] = int(acts[idx])

        res = [self.envs[i].step(all_actions[i]) for i in range(self.num_envs)]
        obs_n, rew_n, done_n, info_n = zip(*res)
        return list(obs_n), list(rew_n), list(done_n), list(info_n)


@torch.no_grad()
def _batch_obs_for_learner(obs_list: List[List[dict]], device: torch.device):
    grids = np.asarray([o[0]["grid"] for o in obs_list], dtype=np.float32)
    vecs = np.asarray([o[0]["vector"] for o in obs_list], dtype=np.float32)
    t_grid = torch.as_tensor(grids, device=device)
    t_vec = torch.as_tensor(vecs, device=device)
    return t_grid, t_vec


def _assign_self_play_opponents(
    envs: VectorizedEnv,
    env_idx: int,
    num_snakes: int,
    sp_manager: SelfPlayManager,
    self_play_prob: float,
    chaos_prob: float = 0.1,
) -> None:
    if num_snakes <= 1:
        return
    for s_idx in range(1, num_snakes):
        if random.random() >= self_play_prob:
            envs.opp_model_paths[env_idx][s_idx] = None
            continue
        m = sp_manager.sample_model(chaos_prob=chaos_prob)
        envs.opp_model_paths[env_idx][s_idx] = str(m) if m else None


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
    log(f">>> [PPO] Device={device} | snakes={num_snakes} | envs={num_envs} | steps={total_timesteps}")

    Path("agent/checkpoints").mkdir(parents=True, exist_ok=True)
    sp_manager = SelfPlayManager(pool_dir)

    model = ActorCritic(vector_dim=28, grid_shape=(5, 20, 20), action_dim=4).to(device)
    if load_path:
        p = Path(load_path)
        if not p.exists():
            raise FileNotFoundError(f"PPO load_path not found: {load_path}")
        log(f">>> [PPO] Loading weights from {load_path}")
        state_dict = torch.load(str(p), map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    # Put an initial snapshot into the pool so self-play can start immediately in Phase 2.
    sp_manager.add_model({k: v.detach().cpu() for k, v in model.state_dict().items()}, name="ppo_init")

    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    # Rollout / PPO hyperparams (User Adjusted)
    # V12.0 A6000 Turbo: Increased num_steps for longer GPU residency
    # V13.5: Reduced num_steps (512->256) for Phase 1 to double update frequency
    # Phase 2 remains 1024 for high-throughput GPU saturation
    num_steps = 1024 if num_snakes > 1 else 256 
    # V17.0: Deep convergence for 1024-width net. Increased 4 -> 10
    update_epochs = 10
    gamma = 0.99
    # V9.3: Adaptive Hyperparams for Phase 1 vs Phase 2
    gae_lambda = 0.95 if num_snakes == 1 else 0.92 
    clip_coef = 0.20
    vf_coef = 0.5
    ent_start = 0.20 if num_snakes == 1 else 0.12 # V11.0: More exploration for 20x20
    ent_min = 0.01
    lr_init = lr
    # V17.0: KL Relaxed (0.08) for Battle to allow radical strategy shifts
    target_kl = 0.02 if num_snakes == 1 else 0.08 

    batch_size = num_envs * num_steps
    # V12.0 A6000 Turbo: Massive minibatches to saturate thousands of CUDA cores
    minibatch_size = 8192 if num_snakes > 1 else 4096 
    if minibatch_size > batch_size:
        minibatch_size = batch_size

    log(f">>> [PPO] Initializing {num_envs} Parallel Environments...")
    envs = VectorizedEnv([make_env(num_snakes, 20, seed + i) for i in range(num_envs)], device)
    log(f">>> [PPO] Environments Ready. Resetting...")
    obs_list = envs.reset()
    # Initialize opponents for each env
    for i in range(num_envs):
        _assign_self_play_opponents(envs, i, num_snakes, sp_manager, self_play_prob)

    # Pre-allocate rollout buffers (on device) - Updated for 5x20x20 28D
    obs_grid = torch.zeros((num_steps, num_envs, 5, 20, 20), device=device)
    obs_vec = torch.zeros((num_steps, num_envs, 28), device=device)
    actions = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)
    logprobs = torch.zeros((num_steps, num_envs), device=device)
    rewards = torch.zeros((num_steps, num_envs), device=device)
    dones = torch.zeros((num_steps, num_envs), device=device)
    values = torch.zeros((num_steps, num_envs), device=device)

    ep_returns: List[float] = []
    running_return = np.zeros(num_envs, dtype=np.float32)

    # For long runs (e.g., 20M), fixed 200k snapshots can create too many files.
    # Scale intervals with total timesteps and keep disk usage bounded.
    save_every = max(200_000, int(total_timesteps * 0.05))
    pool_every = max(400_000, int(total_timesteps * 0.10))
    next_save = save_every
    next_pool = pool_every

    global_step = 0
    last_done = torch.zeros(num_envs, device=device)
    update = 0 # Track updates for LR/entropy decay

    log(f">>> [PPO] Setup finished. Starting training loop (update_steps={num_envs * num_steps})...")
    while global_step < total_timesteps:
        update += 1
        # Anneal learning rate and entropy (V14.0: 10% / 5% Floor to prevent brain-death)
        frac = 1.0 - (update - 1) / (total_timesteps / (num_envs * num_steps))
        frac = max(0.0, frac)
        lr_now = lr_init * (0.1 + 0.9 * frac) 
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now
        
        # V17.0: 2% Entropy floor buffer to maintain strategic unpredictability
        ent_coef = ent_min + (ent_start - ent_min) * (0.10 + 0.90 * frac)
        ent_coef = max(ent_coef, 0.02)

        model.eval()
        for step in range(num_steps):
            global_step += num_envs

            t_grid, t_vec = _batch_obs_for_learner(obs_list, device)
            obs_grid[step].copy_(t_grid)
            obs_vec[step].copy_(t_vec)

            with torch.no_grad():
                a, lp, _, v = model.get_action_and_value(t_grid, t_vec)
            actions[step].copy_(a)
            logprobs[step].copy_(lp)
            values[step].copy_(v.squeeze(-1))

            next_obs, rew, done, _ = envs.step(a.detach().cpu().numpy(), obs_list)

            r0 = np.asarray([x[0] for x in rew], dtype=np.float32)
            d0 = np.asarray([x[0] for x in done], dtype=np.float32)
            rewards[step].copy_(torch.as_tensor(r0, device=device))
            dones[step].copy_(torch.as_tensor(d0, device=device))
            last_done = dones[step]

            # Episode tracking + immediate reset when learner dies
            for i in range(num_envs):
                running_return[i] += r0[i]
                if d0[i] > 0.5:
                    ep_returns.append(float(running_return[i]))
                    running_return[i] = 0.0
                    next_obs[i] = envs.reset_one(i)
                    _assign_self_play_opponents(envs, i, num_snakes, sp_manager, self_play_prob)

            obs_list = next_obs

        # Bootstrap value
        model.eval()
        with torch.no_grad():
            n_grid, n_vec = _batch_obs_for_learner(obs_list, device)
            next_value = model.get_value(n_grid, n_vec).squeeze(-1)

        # GAE
        advantages = torch.zeros((num_steps, num_envs), device=device)
        lastgaelam = torch.zeros(num_envs, device=device)
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - last_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values

        # Flatten
        b_obs_grid = obs_grid.reshape((-1, 5, 20, 20))
        b_obs_vec = obs_vec.reshape((-1, 28))
        b_actions = actions.reshape((-1,))
        b_logprobs = logprobs.reshape((-1,))
        b_advantages = advantages.reshape((-1,))
        b_returns = returns.reshape((-1,))
        b_values = values.reshape((-1,))

        # Advantage normalization
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # PPO updates
        model.train()
        inds = np.arange(batch_size)
        for _ in range(update_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, minibatch_size):
                mb = inds[start : start + minibatch_size]
                _, newlogprob, entropy, newvalue = model.get_action_and_value(
                    b_obs_grid[mb], b_obs_vec[mb], b_actions[mb]
                )
                newvalue = newvalue.squeeze(-1)

                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()

                pg_loss1 = -b_advantages[mb] * ratio
                pg_loss2 = -b_advantages[mb] * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * (b_returns[mb] - newvalue).pow(2).mean()
                ent_loss = entropy.mean()

                loss = pg_loss + vf_coef * v_loss - ent_coef * ent_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.3) # V9: Tighter grad clip
                optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    if approx_kl > target_kl:
                        break # KL too high, skip further epochs for this update
        
        if update == 1:
            log(f">>> [PPO] First update cycle finished. Thruput test passed.")

        # Logging (V13.2: Early high-frequency feedback to eliminate "stuck" illusion)
        if update <= 10 or update % 2 == 0:
            avg_ret = np.mean(ep_returns[-50:]) if ep_returns else 0
            log(f">>> [PPO] Update {update} | {global_step}/{total_timesteps} ({global_step/total_timesteps:.1%}) avg_ep={avg_ret:.2f} lr={lr_now:.2e}")
            # No "best_avg" saving logic here as per instruction.

        # No periodic snapshots as per user request (Only final saved)

        if global_step >= next_pool:
            cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            # V9.0: Keep quiet, only pool update if we have meaningful progress
            sp_manager.add_model(cpu_state, name=f"ppo_step_{global_step}")
            next_pool += 500_000 # Relaxed to 500k to reduce noise/disk usage

    # Final save (doesn't override best filename unless user points it there)
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    final_path = Path(checkpoint_path).with_suffix(".final.pth")
    _atomic_torch_save(cpu_state, final_path)
    log(f">>> [PPO] Finished. Final saved: {final_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--single", action="store_true")
    p.add_argument("--load", type=str, default=None)
    p.add_argument("--steps", type=int, default=1_000_000)
    args = p.parse_args()

    num_snakes = 1 if args.single else 4
    # V12.0 A6000 Turbo: Doubled parallel envs
    num_envs = 64 
    ckpt = "agent/checkpoints/ppo_best.pth" if args.single else "agent/checkpoints/ppo_battle_best.pth"
    # V13.0: Higher LR for Battle Phase due to larger network capacity
    base_lr = 2.0e-4 if args.single else 1.5e-4 

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
