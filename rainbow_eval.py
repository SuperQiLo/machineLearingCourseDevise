"""本地对战/演示用的 Rainbow 评估脚本。"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from agent.config import RainbowConfig
from agent.rainbow_agent import RainbowAgent
from env.multi_snake_env import MultiSnakeEnv
from network.renderer import BattleRenderer, GameClientProtocol


@dataclass
class EvalConfig:
    """封装本地评估所需的关键参数。"""

    model_path: Path
    grid_size: int
    num_snakes: int
    num_food: int
    tick_rate: float
    human_slot: Optional[int]
    ai_slots: Sequence[int]
    epsilon: float
    device: str


def parse_args() -> EvalConfig:
    """解析命令行参数并返回 `EvalConfig`。"""

    parser = argparse.ArgumentParser(description="Rainbow 智能体本地评估")
    parser.add_argument("--model", type=Path, default=Path("agent/checkpoints/rainbow_snake_latest.pth"), help="模型路径，默认使用最新检查点")
    parser.add_argument("--grid-size", type=int, default=20, help="棋盘边长 (默认 20)")
    parser.add_argument("--num-snakes", type=int, default=2, help="场上蛇数量，至少 2 条")
    parser.add_argument("--num-food", type=int, default=8, help="食物数量，默认 8")
    parser.add_argument("--tick", type=float, default=0.12, help="环境步进间隔（秒），默认 0.12s")
    parser.add_argument("--human-slot", type=int, default=0, help="手动控制的槽位编号，设为负数或配合 --spectator 切换为纯观战")
    parser.add_argument("--ai-slot", dest="ai_slots", type=int, action="append", help="指定 AI 占用的槽位，可多次填写；缺省使用槽位 1")
    parser.add_argument("--epsilon", type=float, default=0.05, help="评估时的 epsilon 值，默认 0.05")
    parser.add_argument("--device", default="auto", help="推理设备，auto/CPU/CUDA:n")
    parser.add_argument("--spectator", action="store_true", help="以观战身份启动渲染，不占用蛇槽位")
    parser.add_argument("--no-ai", action="store_true", help="禁用 AI 控制，仅保留手动或观战")

    args = parser.parse_args()
    if args.num_snakes < 2:
        parser.error("num-snakes 必须 >= 2")

    human_slot = None if args.spectator or args.human_slot < 0 else args.human_slot
    raw_ai_slots = [] if args.no_ai else (args.ai_slots or [1])
    ai_slots = tuple(sorted({slot for slot in raw_ai_slots if slot is not None and slot >= 0}))

    if human_slot is not None and not (0 <= human_slot < args.num_snakes):
        parser.error("human-slot 超出蛇数量范围")
    for slot in ai_slots:
        if slot >= args.num_snakes:
            parser.error("AI 槽位编号必须小于 num-snakes")
        if slot == human_slot:
            parser.error("AI 槽位不能与 human-slot 相同")
    if human_slot is None and not ai_slots:
        parser.error("至少需要一个控制实体：手动或 AI")

    return EvalConfig(
        model_path=args.model,
        grid_size=args.grid_size,
        num_snakes=args.num_snakes,
        num_food=max(1, args.num_food),
        tick_rate=max(0.02, args.tick),
        human_slot=human_slot,
        ai_slots=ai_slots,
        epsilon=max(0.0, args.epsilon),
        device=args.device,
    )


class LocalGameAdapter(GameClientProtocol):
    """将本地环境包装成 Renderer 可消费的客户端协议。"""

    def __init__(self, env: MultiSnakeEnv, config: EvalConfig, agent: Optional[RainbowAgent]) -> None:
        self.env = env
        self.cfg = config
        self.agent = agent if config.ai_slots else None
        self.human_slot = config.human_slot
        self.ai_slots = tuple(slot for slot in config.ai_slots if slot != self.human_slot)

        self.pending_human_action = 0
        self.last_obs: List = self.env.reset()
        self.state_cache = self._build_state([0.0] * self.env.num_snakes, {})
        self.done = False
        self._running = False
        self._lock = threading.Lock()

    def get_state(self, room_id: str) -> Optional[Dict]:
        with self._lock:
            return dict(self.state_cache)

    def send_action(self, action: int) -> None:
        if self.human_slot is None:
            return
        self.pending_human_action = int(action)

    def run_loop(self) -> None:
        self._running = True
        while self._running and not self.done:
            self.step_once()
            time.sleep(self.cfg.tick_rate)

    def stop(self) -> None:
        self._running = False

    def step_once(self) -> None:
        if self.done:
            return
        actions = [0] * self.env.num_snakes

        if self.human_slot is not None and self.human_slot < self.env.num_snakes:
            if self.env.snakes[self.human_slot]["alive"]:
                actions[self.human_slot] = self.pending_human_action
        self.pending_human_action = 0

        if self.agent:
            for slot in self.ai_slots:
                if slot >= len(self.last_obs):
                    continue
                if not self.env.snakes[slot]["alive"]:
                    continue
                actions[slot] = self.agent.act(self.last_obs[slot], epsilon=self.cfg.epsilon)

        observations, rewards, dones, info = self.env.step(actions)
        self.last_obs = observations
        if all(dones) or info.get("game_over"):
            self.done = True

        with self._lock:
            self.state_cache = self._build_state(rewards, info)

    def _build_state(self, rewards: Sequence[float], info: Dict) -> Dict:
        snakes_payload = []
        for idx, snake in enumerate(self.env.snakes):
            snakes_payload.append(
                {
                    "slot": idx,
                    "body": list(snake.get("body", [])),
                    "color": self.env.colors[idx % len(self.env.colors)],
                    "score": snake.get("score", 0),
                    "alive": snake.get("alive", False),
                    "direction": snake.get("direction").name if snake.get("direction") else "UP",
                    "owner_name": self._slot_label(idx),
                }
            )

        return {
            "room_id": "LocalEval",
            "mode": "local_eval",
            "snakes": snakes_payload,
            "food": list(self.env.food),
            "steps": info.get("steps", self.env.steps),
            "scores": info.get("scores", [s.get("score", 0) for s in self.env.snakes]),
            "rewards": list(rewards),
            "grid": self.env.width,
            "alive_count": info.get("alive_count", sum(1 for s in self.env.snakes if s.get("alive", False))),
            "game_over": info.get("game_over", False),
        }

    def _slot_label(self, slot: int) -> str:
        if self.human_slot is not None and slot == self.human_slot:
            return "Player"
        if slot in self.ai_slots:
            return "RainbowAI"
        return f"Bot-{slot}"


def build_agent(cfg: EvalConfig) -> Optional[RainbowAgent]:
    if not cfg.ai_slots:
        return None

    agent_cfg = RainbowConfig(grid_size=cfg.grid_size, num_snakes=cfg.num_snakes, device=cfg.device)
    agent = RainbowAgent(agent_cfg)
    if cfg.model_path.exists():
        print(f"加载模型: {cfg.model_path}")
        agent.load(cfg.model_path)
    else:
        print(f"未找到模型 {cfg.model_path}，将使用随机初始化权重。")
    return agent


def eval_game() -> None:
    cfg = parse_args()
    print("=== 本地评估配置 ===")
    print(f"Grid: {cfg.grid_size}x{cfg.grid_size} | Snakes: {cfg.num_snakes} | Food: {cfg.num_food}")
    print(f"Human slot: {cfg.human_slot if cfg.human_slot is not None else 'Spectator'} | AI slots: {list(cfg.ai_slots)}")
    print(f"Tick: {cfg.tick_rate:.3f}s | Epsilon: {cfg.epsilon}")

    env = MultiSnakeEnv(width=cfg.grid_size, height=cfg.grid_size, num_snakes=cfg.num_snakes, num_food=cfg.num_food)
    agent = build_agent(cfg)
    adapter = LocalGameAdapter(env, cfg, agent)

    loop_thread = threading.Thread(target=adapter.run_loop, daemon=True)
    loop_thread.start()

    role = "human" if cfg.human_slot is not None else "spectator"
    renderer = BattleRenderer(client=adapter, room_id="LocalEval", slot=cfg.human_slot, role=role)

    try:
        renderer.run()
    finally:
        adapter.stop()
        loop_thread.join(timeout=1.0)
        print("评估结束。")


if __name__ == "__main__":
    eval_game()
