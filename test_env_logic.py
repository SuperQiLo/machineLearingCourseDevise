
import numpy as np
import sys
import os

# 确保能导入 env 模块
sys.path.append(os.path.abspath(os.curdir))

from env.battle_snake_env import BattleSnakeEnv, BattleSnakeConfig, Action

_HAS_GYMNASIUM = True
try:
    from env.gymnasium_wrapper import make_gymnasium_env
except Exception:
    _HAS_GYMNASIUM = False


def test_gymnasium_wrapper_full_obs_single_snake():
    if not _HAS_GYMNASIUM:
        print("[SKIP] gymnasium 未安装，跳过 gymnasium_wrapper 测试")
        return
    env = make_gymnasium_env(
        num_snakes=1,
        grid_size=20,
        return_full_obs=True,
        use_score_delta_reward=True,
        score_reward_coef=0.001,
        env_reward_coef=0.2,
        dash_effect_window=6,
    )
    obs, info = env.reset()
    assert "full_obs_grids" in info and "full_obs_vecs" in info, "单蛇 reset 必须返回 full_obs_*"
    assert info["full_obs_grids"].shape == (1, 5, 20, 20), "full_obs_grids 形状应为 (1,5,20,20)"
    assert info["full_obs_vecs"].shape[0] == 1, "full_obs_vecs 第一维应为 1"

    obs, reward, terminated, truncated, info2 = env.step(0)
    assert "full_obs_grids" in info2 and "full_obs_vecs" in info2, "单蛇 step 必须返回 full_obs_*"
    assert info2["full_obs_grids"].shape == (1, 5, 20, 20)
    assert info2["full_obs_vecs"].shape[0] == 1
    assert "delta_score0" in info2 and "env_reward0" in info2, "wrapper step 应返回 delta_score0/env_reward0"


def test_gymnasium_wrapper_full_obs_multi_snake():
    if not _HAS_GYMNASIUM:
        print("[SKIP] gymnasium 未安装，跳过 gymnasium_wrapper 测试")
        return
    env = make_gymnasium_env(
        num_snakes=2,
        grid_size=20,
        return_full_obs=True,
        use_score_delta_reward=True,
        score_reward_coef=0.001,
        env_reward_coef=0.2,
        dash_effect_window=6,
    )
    obs, info = env.reset()
    assert "full_obs_grids" in info and "full_obs_vecs" in info
    assert info["full_obs_grids"].shape == (2, 5, 20, 20)
    assert info["full_obs_vecs"].shape[0] == 2

def test_logic():
    print("--- 开始逻辑测试 ---")
    config = BattleSnakeConfig(width=20, height=20, num_snakes=2)
    env = BattleSnakeEnv(config)
    obs = env.reset()
    
    # 为测试 DASH 创造条件（初始长度为 3，需 > 3 才能触发）
    env.snakes[0].append(env.snakes[0][-1]) 

    # 重要：避免 Dash 当步“刚好吃到食物”抵消长度消耗，导致测试随机失败。
    # 将食物强制放在远处（不会被当前一步/冲刺路径吃到）。
    env.foods = [(0, 0), (0, 1)]
    
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

    # 3.5 验证死亡掉落：整条身体全部变食物
    print("验证死亡掉落：整条身体全部变食物...")
    env = BattleSnakeEnv(config)
    env.reset()
    env.foods = []
    env.dead[0] = False
    env.snakes[0] = [(5, 5), (5, 6), (5, 7), (5, 8)]
    env._handle_death(0)
    assert all(seg in env.foods for seg in [(5, 5), (5, 6), (5, 7), (5, 8)]), "死亡后整条身体应全部转化为食物"
    
    # 4. 验证全灭判定
    print("模拟蛇全灭...")
    env.dead = [True, True]
    obs, rewards, dones, info = env.step([Action.STRAIGHT, Action.STRAIGHT])
    print(f"全灭后 dones: {dones}")
    assert all(dones), "全灭后游戏应结束！"

    # 5. 验证只剩一条蛇时立刻结束（避免最后赢家继续移动撞死）
    print("模拟只剩一条蛇存活...")
    env = BattleSnakeEnv(config)
    env.reset()
    env.dead[1] = True
    obs, rewards, dones, info = env.step([Action.STRAIGHT, Action.STRAIGHT])
    print(f"只剩一条蛇存活 dones: {dones}, rewards: {rewards}")
    assert all(dones), "只剩一条蛇存活时游戏应结束！"
    assert rewards[0] > rewards[1], "赢家奖励应大于失败者惩罚！"

    # 6. 验证“全员死亡”特例：赢家必须来自结束前上一时刻的存活蛇
    print("模拟全员死亡且存在早死长蛇...")
    config3 = BattleSnakeConfig(width=20, height=20, num_snakes=3)
    env = BattleSnakeEnv(config3)
    env.reset()
    # P0 早死但身体很长（不应成为赢家）
    env.dead[0] = True
    env.snakes[0] = [(10, 10)] * 30
    # P1/P2 存活，下一步直接撞墙同归于尽
    env.dead[1] = False
    env.dead[2] = False
    env.snakes[1] = [(0, 0), (0, 1), (0, 2)]
    env.snakes[2] = [(19, 0), (19, 1), (19, 2)]
    from env.battle_snake_env import Direction
    env.directions[1] = Direction.UP
    env.directions[2] = Direction.UP
    obs, rewards, dones, info = env.step([Action.STRAIGHT, Action.STRAIGHT, Action.STRAIGHT])
    assert all(dones), "全员死亡后游戏应结束！"
    assert bool(info.get("winner_all_dead", False)) is True, "全员死亡应标记 winner_all_dead"
    assert info.get("winner_idx") in (1, 2), "MVP(最高分/平分按存活与长度)应来自结束前仍存活的蛇（P1/P2）"
    
    print("--- 逻辑测试通过！ ---")

if __name__ == "__main__":
    try:
        test_gymnasium_wrapper_full_obs_single_snake()
        test_gymnasium_wrapper_full_obs_multi_snake()
        test_logic()
    except Exception as e:
        print(f"测试失败: {e}")
        sys.exit(1)
