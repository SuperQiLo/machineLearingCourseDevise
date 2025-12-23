"""
Unified Multi-Agent PPO Training V3 with Self-Play.
Hybrid CNN+MLP for competitive play.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import time
import random

from env.battle_snake_env import BattleSnakeEnv, BattleSnakeConfig
from agent.ppo import ActorCritic, PPOAgent
from utils.self_play import SelfPlayManager

def make_env(num_snakes, grid_size, seed=None):
    def thunk():
        env = BattleSnakeEnv(BattleSnakeConfig(
            width=grid_size, height=grid_size, 
            num_snakes=num_snakes,
            min_food=max(2, num_snakes//2), 
            max_steps=500
        ), seed=seed)
        return env
    return thunk

class VectorizedEnv:
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.history_agents: List[List[Optional[PPOAgent]]] = [[None]*env.config.num_snakes for env in self.envs]
    
    def reset(self):
        return [env.reset() for env in self.envs]
    
    def step(self, learning_actions_list: List[List[int]], current_obs: List[List[Dict]]):
        """
        Mix learning actions with history agent actions.
        learning_actions_list[env_idx] contains actions for the LEARNING agents in that env.
        However, for PPO simplicity, we usually treat ALL agents as learning or handle masking.
        In this implementation, P0 is ALWAYS the learning agent, others might be history.
        """
        all_results = []
        for i, env in enumerate(self.envs):
            env_actions = []
            for j in range(env.config.num_snakes):
                if self.history_agents[i][j]:
                    # History agent takes control
                    env_actions.append(self.history_agents[i][j].act(current_obs[i][j]))
                else:
                    # Learning agent (P0 typically)
                    # Note: We assume learning_actions_list provides enough actions
                    env_actions.append(learning_actions_list[i][j])
            
            all_results.append(env.step(env_actions))
            
        obs_n, rew_n, done_n, info_n = zip(*all_results)
        return list(obs_n), list(rew_n), list(done_n), list(info_n)

def train_ppo(num_envs=8, num_snakes=4, total_timesteps=2_000_000, 
              load_path=None, checkpoint_path="agent/checkpoints/ppo_battle_best.pth",
              self_play_prob=0.3):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path("agent/checkpoints").mkdir(parents=True, exist_ok=True)
    sp_manager = SelfPlayManager("agent/pool/ppo")
    
    envs = VectorizedEnv([make_env(num_snakes, 20, i) for i in range(num_envs)])
    agent = ActorCritic(vector_dim=24).to(device)
    
    if load_path and Path(load_path).exists():
        print(f">>> Loading PPO weights from {load_path}...")
        agent.load_state_dict(torch.load(load_path, map_location=device, weights_only=True))
        
    optimizer = optim.Adam(agent.parameters(), lr=2.5e-4, eps=1e-5)
    
    num_steps = 128 
    # We only train on P0 (the learning agent) for each environment
    batch_size = num_envs * num_steps 
    
    print(f"Starting V3 PPO Training: {num_envs} Envs x {num_snakes} Snakes (Self-Play Friendly).")
    
    obs_list = envs.reset() # List[List[Dict]]
    
    global_step = 0
    best_rew = -float('inf')
    ep_rewards = []
    current_rewards = np.zeros(num_envs)

    while global_step < total_timesteps:
        b_grid, b_vector, b_acts, b_logprobs, b_rewards, b_dones, b_values = [], [], [], [], [], [], []
        
        # 1. Collection
        for _ in range(num_steps):
            global_step += num_envs
            
            # Predict actions MUST be done for ALL snakes in ALL envs
            # For data collection, we only push data for snakes NOT controlled by history agents
            # Specifically, P0 is our target learner.
            
            flat_grid = []
            flat_vec = []
            for env_idx in range(num_envs):
                # We always collect for P0
                flat_grid.append(obs_list[env_idx][0]['grid'])
                flat_vec.append(obs_list[env_idx][0]['vector'])
            
            t_grid = torch.tensor(np.array(flat_grid), dtype=torch.float32).to(device)
            t_vec = torch.tensor(np.array(flat_vec), dtype=torch.float32).to(device)
            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(t_grid, t_vec)
                
            b_grid.append(t_grid)
            b_vector.append(t_vec)
            b_acts.append(action)
            b_logprobs.append(logprob)
            b_values.append(value.flatten())
            
            # Prepare full actions list for step (P0 is learning, others are dummy/history)
            multi_actions = []
            for e_idx in range(num_envs):
                env_actions = [0] * num_snakes
                env_actions[0] = int(action[e_idx].item()) # P0 action
                # Others filled by VectorizedEnv during step via history_agents
                multi_actions.append(env_actions)
                
            next_obs_list, rewards, dones, _ = envs.step(multi_actions, obs_list)
            
            for i in range(num_envs):
                current_rewards[i] += rewards[i][0] # Focus on P0
                if all(dones[i]):
                    ep_rewards.append(current_rewards[i])
                    current_rewards[i] = 0
                    next_obs_list[i] = envs.envs[i].reset()
                    
                    # Self-Play Shuffle
                    if num_snakes > 1 and random.random() < self_play_prob:
                         for s_idx in range(1, num_snakes):
                             model_p = sp_manager.sample_model()
                             if model_p:
                                 envs.history_agents[i][s_idx] = PPOAgent(input_dim=24, model_path=str(model_p))
                             else:
                                 envs.history_agents[i][s_idx] = None
                    else:
                         envs.history_agents[i] = [None] * num_snakes

            b_rewards.append(torch.tensor([r[0] for r in rewards], device=device))
            b_dones.append(torch.tensor([d[0] for d in dones], device=device).float())
            obs_list = next_obs_list

        # 2. Advantage Estimation (GAE)
        with torch.no_grad():
            final_flat_grid = [obs[0]['grid'] for obs in obs_list]
            final_flat_vec = [obs[0]['vector'] for obs in obs_list]
            t_final_grid = torch.tensor(np.array(final_flat_grid), dtype=torch.float32).to(device)
            t_final_vec = torch.tensor(np.array(final_flat_vec), dtype=torch.float32).to(device)
            next_value = agent.get_value(t_final_grid, t_final_vec).reshape(1, -1)
            
        b_grid = torch.stack(b_grid).view(-1, 3, 7, 7)
        b_vector = torch.stack(b_vector).view(-1, 24)
        b_acts = torch.stack(b_acts).view(-1)
        b_logprobs = torch.stack(b_logprobs).view(-1)
        b_values = torch.stack(b_values)
        _rewards = torch.stack(b_rewards)
        _dones = torch.stack(b_dones)
        
        advantages = torch.zeros_like(_rewards).to(device)
        lastgaelam = 0
        gamma, gae_lambda = 0.99, 0.95
        
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - _dones[t]
                nextvalues = next_value.view(-1)
            else:
                nextnonterminal = 1.0 - _dones[t+1]
                nextvalues = b_values[t+1]
                
            delta = _rewards[t] + gamma * nextvalues * nextnonterminal - b_values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            
        returns = (advantages + b_values).reshape(-1)
        b_advantages = advantages.reshape(-1)
        
        # 3. Optimization
        inds = np.arange(batch_size)
        for _ in range(4): # epochs
            np.random.shuffle(inds)
            for start in range(0, batch_size, 32):
                end = start + 32
                mb_inds = inds[start:end]
                
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_grid[mb_inds], b_vector[mb_inds], b_acts[mb_inds])
                ratio = (newlogprob - b_logprobs[mb_inds]).exp()
                
                mb_adv = b_advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                
                pg_loss = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 0.8, 1.2)).mean()
                v_loss = 0.5 * ((newvalue.view(-1) - returns[mb_inds])**2).mean()
                
                loss = pg_loss - 0.01 * entropy.mean() + 0.5 * v_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
                
        # Logging & Pool Update
        if len(ep_rewards) > 0 and global_step % 10240 == 0:
            avg_ep_rew = np.mean(ep_rewards[-20:])
            print(f"Step {global_step} | Ep_Rew_Mean: {avg_ep_rew:.3f} | Total_Ep: {len(ep_rewards)}")
            
            if global_step % 102400 == 0:
                torch.save(agent.state_dict(), checkpoint_path)
                # Add to pool
                sp_manager.add_model(agent.state_dict(), f"ppo_step_{global_step}_{int(time.time())}")
                print(f"Model added to pool. Pool size: {len(sp_manager.history_models)}")

    torch.save(agent.state_dict(), checkpoint_path)
    print(f"PPO V3 Training Finished. Model: {checkpoint_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--steps", type=int, default=1_000_000)
    args = parser.parse_args()
    
    num = 1 if args.single else 4
    ckpt = "agent/checkpoints/ppo_best.pth" if args.single else "agent/checkpoints/ppo_battle_best.pth"
    train_ppo(num_snakes=num, total_timesteps=args.steps, load_path=args.load, checkpoint_path=ckpt)
