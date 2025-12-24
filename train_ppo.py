"""
Unified Multi-Agent PPO Training V3.3.
V4.2: Fixed stdout buffering & Added progress percentage.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import time
import random
import sys

from env.battle_snake_env import BattleSnakeEnv, BattleSnakeConfig
from agent.ppo import ActorCritic, PPOAgent
from utils.self_play import SelfPlayManager

def log(msg):
    print(msg, flush=True)

def make_env(num_snakes, grid_size, seed=None):
    def thunk():
        env_cfg = BattleSnakeConfig(
            width=grid_size, height=grid_size, 
            num_snakes=num_snakes,
            min_food=max(2, num_snakes//2), 
            max_steps=500
        )
        if num_snakes == 1:
            env_cfg.closer_reward, env_cfg.food_reward, env_cfg.death_penalty = 0.35, 12.0, -10.0
        else:
            env_cfg.closer_reward, env_cfg.kill_reward, env_cfg.death_penalty = 0.1, 30.0, -25.0
        return BattleSnakeEnv(env_cfg, seed=seed)
    return thunk

class VectorizedEnv:
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.history_agents: List[List[Optional[PPOAgent]]] = [[None]*env.config.num_snakes for env in self.envs]
    def reset(self): return [env.reset() for env in self.envs]
    def step(self, learning_actions_list, current_obs):
        res = []
        for i, env in enumerate(self.envs):
            env_acts = []
            for j in range(env.config.num_snakes):
                if self.history_agents[i][j]: env_acts.append(self.history_agents[i][j].act(current_obs[i][j]))
                else: env_acts.append(learning_actions_list[i][j])
            res.append(env.step(env_acts))
        obs_n, rew_n, done_n, info_n = zip(*res)
        return list(obs_n), list(rew_n), list(done_n), list(info_n)

def train_ppo(num_envs=8, num_snakes=4, total_timesteps=2_000_000, 
              load_path=None, checkpoint_path="agent/checkpoints/ppo_battle_best.pth",
              self_play_prob=0.3, lr=2.5e-4):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f">>> [Heartbeat] Using device: {device}")
    Path("agent/checkpoints").mkdir(parents=True, exist_ok=True)
    sp_manager = SelfPlayManager("agent/pool/ppo")
    
    envs = VectorizedEnv([make_env(num_snakes, 20, i) for i in range(num_envs)])
    agent = ActorCritic(vector_dim=24).to(device)
    
    if load_path and Path(load_path).exists():
        log(f">>> Loading PPO weights from {load_path}...")
        agent.load_state_dict(torch.load(load_path, map_location=device, weights_only=True))
        
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    num_steps, batch_size = 128, num_envs * 128
    
    log(f">>> Starting V4.2 PPO Training: {num_envs} Envs | {num_snakes} Snakes.")
    obs_list = envs.reset() 
    global_step, ep_rewards = 0, []
    current_rewards = np.zeros(num_envs)

    while global_step < total_timesteps:
        # LR Decay
        frac = 1.0 - (global_step / total_timesteps)
        for g in optimizer.param_groups: g['lr'] = frac * lr

        b_grid, b_vector, b_acts, b_logprobs, b_rewards, b_dones, b_values = [], [], [], [], [], [], []
        
        for _ in range(num_steps):
            global_step += num_envs
            flat_grid, flat_vec = [o[0]['grid'] for o in obs_list], [o[0]['vector'] for o in obs_list]
            t_grid = torch.tensor(np.array(flat_grid), dtype=torch.float32).to(device)
            t_vec = torch.tensor(np.array(flat_vec), dtype=torch.float32).to(device)
            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(t_grid, t_vec)
                
            multi_actions = [[int(action[e].item())] + [0]*(num_snakes-1) for e in range(num_envs)]
            next_obs, rewards, dones, _ = envs.step(multi_actions, obs_list)
            
            for i in range(num_envs):
                current_rewards[i] += rewards[i][0]
                if all(dones[i]):
                    ep_rewards.append(current_rewards[i])
                    current_rewards[i] = 0
                    next_obs[i] = envs.envs[i].reset()
                    # Self-Play Shuffle
                    if num_snakes > 1 and random.random() < self_play_prob:
                         for s_idx in range(1, num_snakes):
                             m_p = sp_manager.sample_model()
                             if m_p: envs.history_agents[i][s_idx] = PPOAgent(input_dim=24, model_path=str(m_p))

            b_grid.append(t_grid); b_vector.append(t_vec); b_acts.append(action)
            b_logprobs.append(logprob); b_values.append(value.flatten())
            b_rewards.append(torch.tensor([r[0] for r in rewards], device=device))
            b_dones.append(torch.tensor([d[0] for d in dones], device=device).float())
            obs_list = next_obs

        # Advantage (GAE)
        with torch.no_grad():
            f_g, f_v = [o[0]['grid'] for o in obs_list], [o[0]['vector'] for o in obs_list]
            next_v = agent.get_value(torch.tensor(np.array(f_g), dtype=torch.float32).to(device), torch.tensor(np.array(f_v), dtype=torch.float32).to(device)).reshape(1, -1)
            
        bg, bv, ba, blp, bval = torch.stack(b_grid).view(-1,3,7,7), torch.stack(b_vector).view(-1,24), torch.stack(b_acts).view(-1), torch.stack(b_logprobs).view(-1), torch.stack(b_values)
        br, bd = torch.stack(b_rewards), torch.stack(b_dones)
        
        advantages = torch.zeros_like(br).to(device)
        lastgaelam, gamma, gae_lambda = 0, 0.99, 0.95
        for t in reversed(range(num_steps)):
            nextnonterminal = 1.0 - bd[t] if t == num_steps - 1 else 1.0 - bd[t+1]
            nextvalues = next_v.view(-1) if t == num_steps - 1 else bval[t+1]
            delta = br[t] + gamma * nextvalues * nextnonterminal - bval[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        returns, b_adv = (advantages + bval).reshape(-1), advantages.reshape(-1)
        
        # Optimization
        inds = np.arange(batch_size)
        for _ in range(4):
            np.random.shuffle(inds)
            for s in range(0, batch_size, 64):
                mb = inds[s:s+64]
                _, nlp, ent, nv = agent.get_action_and_value(bg[mb], bv[mb], ba[mb])
                ratio = (nlp - blp[mb]).exp()
                mb_a = (b_adv[mb] - b_adv[mb].mean())/(b_adv[mb].std() + 1e-8)
                pg_loss = torch.max(-mb_a * ratio, -mb_a * torch.clamp(ratio, 0.8, 1.2)).mean()
                v_loss = 0.5 * ((nv.view(-1) - returns[mb])**2).mean()
                optimizer.zero_grad(); (pg_loss - 0.03 * ent.mean() + 0.5 * v_loss).backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5); optimizer.step()
                
        # Logging
        if global_step % 10240 < num_envs:
            avg_rew = np.mean(ep_rewards[-50:]) if ep_rewards else 0
            perc = (global_step/total_timesteps)*100
            log(f">>> [Progress {perc:.1f}%] Step: {global_step} | Ep_Rew: {avg_rew:.2f} | Info: LR={frac*lr:.2e}")
            if global_step % 204800 < num_envs:
                torch.save(agent.state_dict(), checkpoint_path)
                sp_manager.add_model(agent.state_dict(), f"ppo_v4_step_{global_step}")

    torch.save(agent.state_dict(), checkpoint_path)
    log(f"PPO V4.2 Training Finished. Model: {checkpoint_path}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--single", action="store_true"); p.add_argument("--load", type=str); p.add_argument("--steps", type=int, default=1000000)
    args = p.parse_args(); num = 1 if args.single else 4
    ckpt = "agent/checkpoints/ppo_best.pth" if args.single else "agent/checkpoints/ppo_battle_best.pth"
    train_ppo(num_snakes=num, total_timesteps=args.steps, load_path=args.load, checkpoint_path=ckpt)
