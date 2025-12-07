"""Standalone local arena module for offline practice."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pygame
import torch

from agent.ppo import PPOAgent
from agent.utils import resolve_device
from env.multi_snake_env import Direction, MultiSnakeEnv
from network.constants import COLOR_MAP, COLOR_SEQUENCE
from network.utils import direction_to_relative, normalize_port


class LocalArena:
    def __init__(
        self,
        grid_size: int = 30,
        num_snakes: int = 4,
        model_path: Optional[str] = None,
        human_player: bool = True,
        target_fps: int = 30,
        sim_interval: float = 0.12,
        countdown_duration: float = 3.0,
        device: str = "auto",
    ) -> None:
        """初始化本地对练场景，配置地图大小、蛇数量、模型、帧率与节奏。"""
        self.grid_size = grid_size
        self.num_snakes = num_snakes
        self.model_path = model_path
        self.human_player = human_player
        self.target_fps = target_fps
        self.sim_interval = sim_interval
        self.countdown_duration = countdown_duration
        self.device_preference = device

        self.env = MultiSnakeEnv(width=grid_size, height=grid_size, num_snakes=num_snakes)
        self.ai_agent: Optional[PPOAgent] = None
        if model_path:
            self._load_agent(model_path)

        self.controlled_slot = 0
        self.desired_direction: Optional[Direction] = None
        self.await_restart = False
        self.pending_restart = False
        self.latest_scores: Optional[List[int]] = None

        self.in_countdown = False
        self.countdown_end_time: Optional[float] = None
        self.last_step_time = 0.0

        self.clock = pygame.time.Clock()
        self.screen: Optional[pygame.Surface] = None
        self.font: Optional[pygame.font.Font] = None
        self.small_font: Optional[pygame.font.Font] = None
        self.running = False

    def _load_agent(self, model_path: str) -> None:
        """尝试加载外部模型，用于除人工蛇外的 AI 控制。"""
        try:
            device = resolve_device(self.device_preference)
            agent = PPOAgent(grid_size=self.grid_size, device=device)
            state_dict = torch.load(Path(model_path), map_location="cpu")
            agent.policy.load_state_dict(state_dict)
            agent.policy.eval()
            self.ai_agent = agent
            print(f"[LocalArena] 模型已加载: {model_path} | 设备: {device}")
        except Exception as exc:  # noqa: BLE001
            print(f"[LocalArena] 模型加载失败: {exc}")
            self.ai_agent = None

    def run(self) -> None:
        """启动 Pygame 循环，处理输入、渲染以及回合重开。"""
        pygame.init()
        self.font = pygame.font.SysFont("simhei", 24)
        self.small_font = pygame.font.SysFont("simhei", 18)
        width = self.grid_size * 30 + 300
        height = self.grid_size * 30 + 80
        self.screen = pygame.display.set_mode((max(width, 900), max(height, 600)))
        pygame.display.set_caption("本地对练模式")

        observations = self.env.reset()
        self._sync_control_direction()
        info_state = {"steps": 0, "alive_count": self.num_snakes, "scores": [0 for _ in range(self.num_snakes)]}
        self._start_countdown()
        self.running = True

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    self._handle_keydown(event.key)

            if not self.running:
                break

            if self.pending_restart:
                observations = self.env.reset()
                info_state = {"steps": 0, "alive_count": self.num_snakes, "scores": [0 for _ in range(self.num_snakes)]}
                self.pending_restart = False
                self.await_restart = False
                self.latest_scores = None
                self._sync_control_direction()
                self._start_countdown()
                continue

            current_time = time.perf_counter()

            if self.await_restart:
                self._render_local(info_state)
                self.clock.tick(self.target_fps)
                continue

            if self.in_countdown:
                remaining = max(0.0, (self.countdown_end_time or current_time) - current_time)
                if remaining <= 0:
                    self.in_countdown = False
                    self.last_step_time = current_time
                else:
                    seconds_left = max(1, math.ceil(remaining))
                    self._render_local(info_state, countdown_seconds=seconds_left)
                    self.clock.tick(self.target_fps)
                    continue

            if current_time - self.last_step_time < self.sim_interval:
                self._render_local(info_state)
                self.clock.tick(self.target_fps)
                continue

            self.last_step_time = current_time

            actions: List[int] = []
            for idx in range(self.num_snakes):
                if idx == self.controlled_slot and self.human_player:
                    actions.append(self._human_action(idx))
                else:
                    actions.append(self._ai_or_heuristic(observations[idx], idx))

            observations, _, dones, info_state = self.env.step(actions)
            self._render_local(info_state)

            if info_state.get("game_over") and all(dones):
                self.await_restart = True
                self.latest_scores = info_state.get("scores", [])

            self.clock.tick(self.target_fps)

        pygame.quit()

    def _ai_or_heuristic(self, observation: np.ndarray, slot: int) -> int:
        """根据是否加载模型决定使用 PPO 推理或启发式动作。"""
        if self.ai_agent:
            return self.ai_agent.predict(observation)
        return self._heuristic_action(slot)

    def _human_action(self, slot: int) -> int:
        """根据当前/期望方向，将玩家方向键转为相对动作编号。"""
        snake = self.env.snakes[slot]
        if not snake["alive"]:
            return 0
        current_dir = snake["direction"]
        desired = self.desired_direction or current_dir
        if (current_dir.value + 2) % 4 == desired.value:
            desired = current_dir
        return direction_to_relative(current_dir, desired)

    def _heuristic_action(self, slot: int) -> int:
        """简单的食物趋近策略：尽量沿最短曼哈顿路径靠近最近食物。"""
        snake = self.env.snakes[slot]
        if not snake["alive"] or not self.env.food:
            return 0
        head_x, head_y = snake["body"][0]
        target = min(self.env.food, key=lambda pos: abs(pos[0] - head_x) + abs(pos[1] - head_y))
        dx = target[0] - head_x
        dy = target[1] - head_y
        desired_dir = snake["direction"]
        if abs(dx) > abs(dy):
            desired_dir = Direction.RIGHT if dx > 0 else Direction.LEFT
        elif dy != 0:
            desired_dir = Direction.DOWN if dy > 0 else Direction.UP
        return direction_to_relative(snake["direction"], desired_dir)

    def _render_local(self, info: Dict, countdown_seconds: Optional[int] = None) -> None:
        """绘制网格、蛇、HUD、倒计时与回合结束提示。"""
        if not self.screen or not self.font or not self.small_font:
            return
        grid_surface = pygame.Surface((self.grid_size * 30, self.grid_size * 30))
        grid_surface.fill((15, 17, 24))
        for x in range(0, grid_surface.get_width(), 30):
            pygame.draw.line(grid_surface, (40, 40, 45), (x, 0), (x, grid_surface.get_height()))
        for y in range(0, grid_surface.get_height(), 30):
            pygame.draw.line(grid_surface, (40, 40, 45), (0, y), (grid_surface.get_width(), y))

        for fx, fy in self.env.food:
            rect = pygame.Rect(fx * 30, fy * 30, 30, 30)
            pygame.draw.rect(grid_surface, (255, 110, 110), rect)

        color_palette = [COLOR_MAP[name] for name in COLOR_SEQUENCE]
        for idx, snake in enumerate(self.env.snakes):
            if not snake["alive"]:
                continue
            color = color_palette[idx % len(color_palette)]
            for seg, (bx, by) in enumerate(snake["body"]):
                rect = pygame.Rect(bx * 30, by * 30, 30, 30)
                pygame.draw.rect(grid_surface, color, rect)
                pygame.draw.rect(grid_surface, (0, 0, 0), rect, 1)
                if seg == 0:
                    pygame.draw.rect(grid_surface, (255, 255, 255), rect, 2)
                if idx == self.controlled_slot and self.human_player:
                    glow = rect.inflate(6, 6)
                    pygame.draw.rect(grid_surface, (255, 255, 200), glow, 2, border_radius=4)

        self.screen.fill((12, 14, 20))
        self.screen.blit(grid_surface, (40, 40))

        status_text = self.font.render(
            f"步数: {info.get('steps', 0)} | 存活: {info.get('alive_count', 0)}",
            True,
            (230, 230, 230),
        )
        self.screen.blit(status_text, (40, grid_surface.get_height() + 60))

        control_color_name = COLOR_SEQUENCE[self.controlled_slot % len(COLOR_SEQUENCE)]
        control_hint = self.small_font.render(
            f"你控制: 蛇 {self.controlled_slot} ({control_color_name}) | 方向键=控制 | ESC=退出",
            True,
            (210, 230, 240),
        )
        self.screen.blit(control_hint, (40, grid_surface.get_height() + 90))

        if countdown_seconds:
            self._render_countdown_overlay(countdown_seconds, control_color_name)

        if self.await_restart:
            self._render_restart_overlay()

        pygame.display.flip()

    def _handle_keydown(self, key: int) -> None:
        """处理键盘输入：方向键控制、ESC 退出、回合结束后的确认。"""
        if key == pygame.K_ESCAPE:
            self.running = False
            return

        if self.await_restart:
            if key in (pygame.K_RETURN, pygame.K_SPACE):
                self.pending_restart = True
            elif key in (pygame.K_q, pygame.K_BACKSPACE):
                self.running = False
            return

        if not self.human_player:
            return

        direction_map = {
            pygame.K_UP: Direction.UP,
            pygame.K_DOWN: Direction.DOWN,
            pygame.K_LEFT: Direction.LEFT,
            pygame.K_RIGHT: Direction.RIGHT,
        }
        desired = direction_map.get(key)
        if desired is not None:
            self.desired_direction = desired

    def _sync_control_direction(self) -> None:
        """同步玩家控制蛇的默认朝向，避免一开始相对动作错误。"""
        if not self.env.snakes:
            self.desired_direction = Direction.RIGHT
            return
        self.desired_direction = self.env.snakes[self.controlled_slot]["direction"]

    def _render_restart_overlay(self) -> None:
        """在回合结束时渲染提示面板，展示得分并引导下一步操作。"""
        if not self.screen or not self.font or not self.small_font:
            return
        width, height = self.screen.get_size()
        panel = pygame.Surface((width, 200), pygame.SRCALPHA)
        panel.fill((10, 10, 10, 220))
        self.screen.blit(panel, (0, height // 2 - 100))

        title = self.font.render("本局结束", True, (255, 230, 190))
        self.screen.blit(title, (width // 2 - title.get_width() // 2, height // 2 - 80))
        instructions = self.small_font.render("按 Enter/Space 继续，按 Q 或 ESC 退出", True, (230, 230, 230))
        self.screen.blit(instructions, (width // 2 - instructions.get_width() // 2, height // 2 - 40))

        if self.latest_scores:
            for idx, score in enumerate(self.latest_scores):
                text = self.small_font.render(f"蛇 {idx}: {score}", True, (220, 220, 240))
                self.screen.blit(text, (width // 2 - 80, height // 2 - 10 + idx * 20))

    def _render_countdown_overlay(self, seconds: int, color_name: str) -> None:
        """在正式开局前绘制倒计时与身份提示。"""
        if not self.screen or not self.font or not self.small_font:
            return
        width, height = self.screen.get_size()
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.screen.blit(overlay, (0, 0))

        number_text = self.font.render(str(seconds), True, (255, 240, 180))
        self.screen.blit(number_text, (width // 2 - number_text.get_width() // 2, height // 2 - 70))

        hint = self.small_font.render(
            f"准备: 你将控制蛇 {self.controlled_slot} ({color_name})",
            True,
            (235, 235, 235),
        )
        self.screen.blit(hint, (width // 2 - hint.get_width() // 2, height // 2))

    def _start_countdown(self) -> None:
        """重置倒计时状态，保证每轮开局有缓冲时间。"""
        self.in_countdown = True
        now = time.perf_counter()
        self.countdown_end_time = now + self.countdown_duration
        self.last_step_time = now


def start_local_arena(
    grid_size: int = 30,
    num_snakes: int = 4,
    human_player: bool = True,
    model_path: Optional[str] = None,
    target_fps: int = 30,
    sim_interval: float = 0.12,
    countdown_duration: float = 3.0,
    device: str = "auto",
) -> None:
    """封装入口函数，创建 LocalArena 后直接运行。"""
    LocalArena(
        grid_size=grid_size,
        num_snakes=num_snakes,
        human_player=human_player,
        model_path=model_path,
        target_fps=target_fps,
        sim_interval=sim_interval,
        countdown_duration=countdown_duration,
        device=device,
    ).run()
