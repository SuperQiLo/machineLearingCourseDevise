
import numpy as np
import sys
import os

# 确保能导入 env 模块
sys.path.append(os.path.abspath(os.curdir))

from env.battle_snake_env import BattleSnakeEnv, BattleSnakeConfig, Action

def test_logic():
    print("--- 开始逻辑测试 ---")
    config = BattleSnakeConfig(width=20, height=20, num_snakes=2)
    env = BattleSnakeEnv(config)
    obs = env.reset()
    
    # 为测试 DASH 创造条件（初始长度为 3，需 > 3 才能触发）
    env.snakes[0].append(env.snakes[0][-1]) 
    
    # 1. 验证视野维度
    print(f"验证视野维度: {obs[0]['grid'].shape}") # 预期应为 (5, 20, 20)
    assert obs[0]['grid'].shape == (5, 20, 20), "视野维度不正确！"
    
    # 2. 验证持续冲刺逻辑
    print("触发 P0 冲刺...")
    initial_len = len(env.snakes[0])
    # 模拟 P0 执行冲刺动作，P1 原地不动（STRAIGHT）
    obs, rewards, dones, info = env.step([Action.DASH, Action.STRAIGHT])
    
    current_len = len(env.snakes[0])
    print(f"P0 触发冲刺后长度: {current_len} (初始: {initial_len})")
    assert current_len == initial_len - 1, "冲刺应消耗一个长度！"
    assert env.dash_durations[0] == config.dash_duration_steps - 1, "冲刺持续时间计数错误！"
    
    # 3. 验证连发限制
    obs, rewards, dones, info = env.step([Action.DASH, Action.STRAIGHT])
    # 冲刺期间再次动作不应重叠或报错，且 duration 继续减少
    print(f"冲刺中再次尝试 DASH，剩余持续时间: {env.dash_durations[0]}")
    
    # 4. 验证全灭判定
    print("模拟蛇全灭...")
    env.dead = [True, True]
    obs, rewards, dones, info = env.step([Action.STRAIGHT, Action.STRAIGHT])
    print(f"全灭后 dones: {dones}")
    assert all(dones), "全灭后游戏应结束！"
    
    print("--- 逻辑测试通过！ ---")

if __name__ == "__main__":
    try:
        test_logic()
    except Exception as e:
        print(f"测试失败: {e}")
        sys.exit(1)
