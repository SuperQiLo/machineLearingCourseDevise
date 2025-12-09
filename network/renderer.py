"""基于 Pygame 的网格渲染器，负责显示游戏状态。"""

from __future__ import annotations

import time
from typing import Dict, Optional, Protocol, Any

import pygame

from env.multi_snake_env import Direction
COLOR_MAP = {
    "green": (0, 200, 0),
    "blue": (50, 120, 255),
    "red": (255, 70, 70),
    "yellow": (255, 200, 0),
    "purple": (180, 70, 255),
    "orange": (255, 150, 60),
    "teal": (60, 200, 200),
    "pink": (255, 105, 180),
    "lime": (180, 255, 80),
    "cyan": (80, 220, 255),
    "magenta": (255, 80, 200),
    "silver": (200, 200, 210),
    "gold": (255, 215, 0),
}
ROLE_NAMES = {"human": "手操", "ai": "AI", "spectator": "观战"}
from network.utils import direction_to_relative


class GameClientProtocol(Protocol):
    """渲染器所需的客户端接口协议。"""
    
    def get_state(self, room_id: str) -> Optional[Dict]:
        ...
    
    def send_action(self, action: int) -> None:
        ...


class BattleRenderer:
    """基于 Pygame 的网格渲染器，仅负责显示状态与采集按键。"""

    def __init__(self, client: GameClientProtocol, room_id: str, slot: Optional[int], *, role: str = "human") -> None:
        self.client = client
        self.room_id = room_id
        self.slot = slot
        self.role = role  # human/ai/spectator
        self.running = True
        self.grid = 30
        self.cell = 24
        self.current_direction = Direction.RIGHT
        self.countdown_started_at: Optional[float] = None
        self.countdown_done = False
        self.server_countdown_active = False
        self.game_over = False
        self.mvp_text: Optional[str] = None

    def run(self) -> None:
        pygame.init()
        pygame.display.set_caption("多蛇对战渲染窗口")
        font = pygame.font.SysFont("simhei", 20)
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((960, 720))

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    else:
                        self._handle_key(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.game_over:
                        self.running = False

            state = self.client.get_state(self.room_id)
            screen.fill((15, 16, 24))
            if state:
                self._render_state(screen, state, font)
            else:
                text = font.render("等待状态更新……", True, (240, 240, 240))
                screen.blit(text, (40, 40))

            pygame.display.flip()
            clock.tick(30)

        pygame.quit()

    def _handle_key(self, key: int) -> None:
        if self.game_over:
            self.running = False
            return
        if not self.countdown_done:
            return
        if self.role != "human":
            return
        if self.slot is None:
            return
        mapping = {
            pygame.K_UP: Direction.UP,
            pygame.K_DOWN: Direction.DOWN,
            pygame.K_LEFT: Direction.LEFT,
            pygame.K_RIGHT: Direction.RIGHT,
        }
        if key not in mapping:
            return
        desired = mapping[key]
        if (self.current_direction.value + 2) % 4 == desired.value:
            return
        # 转换绝对方向为相对动作 (0=直行, 1=左转, 2=右转)
        action = direction_to_relative(self.current_direction, desired)
        self.client.send_action(action)
        
        # 客户端本地预测更新方向（以便连续按键），实际方向以服务器为准
        # 但为了手感，这里通常做乐观更新
        delta = {0: 0, 1: -1, 2: 1}.get(action, 0)
        self.current_direction = Direction((self.current_direction.value + delta) % 4)

    def _render_state(self, screen: pygame.Surface, state: Dict, font: pygame.font.Font) -> None:
        grid = int(state.get("grid", 30))
        self.grid = grid
        self.cell = max(12, min(32, 640 // grid))
        surface = pygame.Surface((grid * self.cell, grid * self.cell))
        surface.fill((22, 24, 32))

        for x in range(grid):
            pygame.draw.line(surface, (40, 40, 50), (x * self.cell, 0), (x * self.cell, grid * self.cell))
        for y in range(grid):
            pygame.draw.line(surface, (40, 40, 50), (0, y * self.cell), (grid * self.cell, y * self.cell))

        for fx, fy in state.get("food", []):
            rect = pygame.Rect(fx * self.cell, fy * self.cell, self.cell, self.cell)
            pygame.draw.rect(surface, (230, 120, 120), rect)

        snakes = state.get("snakes", [])

        def snake_label(idx: int) -> str:
            if idx < len(snakes):
                entry = snakes[idx]
                name = entry.get("owner_name")
                if name:
                    return name
            return f"玩家 {idx}"
        for snake in snakes:
            if not snake.get("alive", True):
                continue
            color = COLOR_MAP.get(snake.get("color", "green"), (0, 200, 0))
            for idx, (sx, sy) in enumerate(snake.get("body", [])):
                rect = pygame.Rect(sx * self.cell, sy * self.cell, self.cell, self.cell)
                pygame.draw.rect(surface, color, rect)
                pygame.draw.rect(surface, (0, 0, 0), rect, 1)
                if idx == 0:
                    pygame.draw.rect(surface, (255, 255, 255), rect, 2)

        screen.blit(surface, (40, 40))
        info_panel = pygame.Surface((300, grid * self.cell))
        info_panel.fill((18, 20, 28))
        screen.blit(info_panel, (surface.get_width() + 80, 40))

        lines = [
            f"房间号: {self.room_id}",
            f"步数: {state.get('steps', 0)}",
        ]
        for idx, text in enumerate(lines):
            label = font.render(text, True, (230, 230, 235))
            screen.blit(label, (surface.get_width() + 90, 60 + idx * 28))

        scores = state.get("scores") or []
        score_y = 120
        for idx, snake in enumerate(snakes):
            owner_id = snake.get("owner_id")
            owner_name = snake.get("owner_name")
            if not owner_id and not owner_name:
                continue
            role_text = snake.get("role")
            role_label = ROLE_NAMES.get(role_text, role_text) if role_text else None
            display_name = owner_name or owner_id or f"玩家 {idx}"
            if role_label:
                display_name = f"{display_name}-{role_label}"
            score_val = scores[idx] if idx < len(scores) else snake.get("score", 0)
            text = font.render(f"{display_name}: {score_val:.1f}", True, (200, 200, 210))
            screen.blit(text, (surface.get_width() + 90, score_y))
            score_y += 26

        if self.slot is not None and self.slot < len(snakes):
            player_snake = snakes[self.slot]
            if player_snake.get("alive", True):
                dir_name = player_snake.get("direction")
                if dir_name:
                    try:
                        self.current_direction = Direction[dir_name]
                    except KeyError:
                        pass

        def draw_countdown(remain: float) -> None:
            overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 140))
            screen.blit(overlay, (40, 40))
            txt = "开始倒计时" if remain > 0.1 else "开始!"
            disp = f"{int(remain) + 1}" if remain > 0.1 else "GO"
            label = font.render(txt, True, (255, 230, 170))
            screen.blit(label, (surface.get_width() // 2 - 40, surface.get_height() // 2 - 60))
            big_font = pygame.font.SysFont("simhei", 48)
            g_label = big_font.render(disp, True, (255, 255, 255))
            screen.blit(g_label, (surface.get_width() // 2 - 20, surface.get_height() // 2 - 10))

        now = time.time()
        countdown_needed = True
        countdown_value = state.get("countdown")
        countdown_in_progress = False

        if countdown_value is not None:
            try:
                remain = max(0.0, float(countdown_value))
            except (TypeError, ValueError):
                remain = 0.0
            countdown_in_progress = remain > 0.05
            self.server_countdown_active = countdown_in_progress
            self.countdown_done = not countdown_in_progress
            if countdown_in_progress:
                self.countdown_started_at = None
                self.game_over = False
                self.mvp_text = None
                draw_countdown(remain)
        elif self.server_countdown_active:
            # Server-driven countdown just finished; mark as done immediately.
            self.server_countdown_active = False
            self.countdown_started_at = None
            self.countdown_done = True
        elif countdown_needed and not self.countdown_done:
            if self.countdown_started_at is None:
                self.countdown_started_at = now
            remain = max(0.0, 3.0 - (now - self.countdown_started_at))
            countdown_in_progress = remain > 0.05
            if countdown_in_progress:
                self.game_over = False
                self.mvp_text = None
                draw_countdown(remain)
            if not countdown_in_progress:
                self.countdown_done = True
        else:
            if not countdown_needed:
                self.countdown_done = True

        game_over = False if countdown_in_progress else (bool(state.get("game_over")) or state.get("alive_count", len(snakes)) <= 1)
        self.game_over = game_over
        if game_over:
            if self.mvp_text is None:
                scores = state.get("scores") or []
                if not scores and snakes:
                    scores = [s.get("score", 0) for s in snakes]
                if scores:
                    best = max(range(len(scores)), key=lambda i: scores[i])
                    label_text = snake_label(best)
                    self.mvp_text = f"MVP: {label_text} 分数 {scores[best]}"
                else:
                    self.mvp_text = "对局结束"
            overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            screen.blit(overlay, (0, 0))
            big_font = pygame.font.SysFont("simhei", 40)
            msg = big_font.render(self.mvp_text or "对局结束", True, (255, 230, 170))
            tip = font.render("点击或按任意键返回成员列表", True, (240, 240, 240))
            screen.blit(msg, (screen.get_width() // 2 - 160, screen.get_height() // 2 - 40))
            screen.blit(tip, (screen.get_width() // 2 - 160, screen.get_height() // 2 + 10))
        else:
            self.mvp_text = None
            tips = font.render("ESC 退出渲染窗口", True, (255, 230, 170))
            screen.blit(tips, (40, surface.get_height() + 60))
