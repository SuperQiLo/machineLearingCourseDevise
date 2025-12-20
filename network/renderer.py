"""基于 Pygame 的网格渲染器，负责显示游戏状态。"""

from __future__ import annotations

import time
from typing import Dict, Optional, Protocol, Any

import numpy as np
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
    """基于 Pygame 的网格渲染器，支持状态插值实现丝滑移动，并提供精致的霓虹视觉效果。"""

    def __init__(self, client: GameClientProtocol, room_id: str, slot: Optional[int], *, role: str = "human") -> None:
        self.client = client
        self.room_id = room_id
        self.slot = slot
        self.role = role  # human/ai/spectator
        self.running = True
        self.grid = 30
        self.cell = 24
        self.current_direction = Direction.RIGHT
        
        # 插值相关状态
        self.prev_state: Optional[Dict] = None
        self.curr_state: Optional[Dict] = None
        self.last_state_time = 0.0
        self.state_interval = 0.12  # 预期状态更新间隔
        
        self.countdown_done = False
        self.server_countdown_active = False
        self.game_over = False
        self.mvp_text: Optional[str] = None

        # 霓虹配色配置
        self.snake_colors = [
            (0, 255, 159),   # Neon Green
            (255, 0, 85),    # Neon Red/Pink
            (0, 243, 255),   # Neon Cyan
            (255, 204, 0),   # Neon Yellow
            (189, 0, 255),   # Neon Purple
            (255, 153, 0),   # Neon Orange
        ]

    def run(self) -> None:
        pygame.init()
        pygame.display.set_caption("Antigravity Snake - Precision Edition")
        
        # 尝试加载更美观的字体
        try:
            font = pygame.font.SysFont("Segoe UI, Roboto, sans-serif", 20)
            big_font = pygame.font.SysFont("Segoe UI, Roboto, sans-serif", 40, bold=True)
        except:
            font = pygame.font.SysFont("simhei", 20)
            big_font = pygame.font.SysFont("simhei", 40)

        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((960, 720))

        while self.running:
            now = time.time()
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

            # 获取最新状态
            new_state = self.client.get_state(self.room_id)
            if new_state:
                # 简单判断状态是否更新（基于步数）
                curr_steps = self.curr_state.get("steps", -1) if self.curr_state else -1
                new_steps = new_state.get("steps", 0)
                
                curr_countdown = self.curr_state.get("countdown", -1.0) if self.curr_state else -1.0
                new_countdown = new_state.get("countdown", -1.0)
                
                if new_steps != curr_steps or new_countdown != curr_countdown:
                    self.prev_state = self.curr_state
                    self.curr_state = new_state
                    
                    # 同步方向，防止控制反置
                    if self.slot is not None and self.curr_state:
                        for s in self.curr_state.get("snakes", []):
                            if s["slot"] == self.slot:
                                server_dir_name = s.get("direction")
                                if server_dir_name:
                                    self.current_direction = Direction[server_dir_name]
                                break
                    if self.prev_state:
                        # 动态估算更新间隔
                        self.state_interval = now - self.last_state_time
                    self.last_state_time = now
                elif not self.curr_state:
                    self.curr_state = new_state
                    self.last_state_time = now

            screen.fill((10, 11, 18)) # 更深的底色
            
            if self.curr_state:
                # 计算插值进度 alpha (0.0 到 1.0)
                # 我们延迟显示一帧，以便在 prev 和 curr 之间平滑移动
                if self.prev_state and not self.game_over and self.countdown_done:
                    alpha = min(1.0, (now - self.last_state_time) / max(0.01, self.state_interval))
                else:
                    alpha = 1.0
                
                self._render_scene(screen, self.prev_state or self.curr_state, self.curr_state, alpha, font, big_font)
            else:
                text = font.render("Initializing Neural Links...", True, (80, 100, 120))
                screen.blit(text, (40, 40))

            pygame.display.flip()
            clock.tick(60) # 60 FPS 提供极致丝滑

        pygame.quit()

    def _handle_key(self, key: int) -> None:
        if self.game_over:
            self.running = False
            return
        if not self.countdown_done:
            return
        if self.role != "human" or self.slot is None:
            return
            
        mapping = {
            pygame.K_UP: Direction.UP, pygame.K_DOWN: Direction.DOWN,
            pygame.K_LEFT: Direction.LEFT, pygame.K_RIGHT: Direction.RIGHT,
            pygame.K_w: Direction.UP, pygame.K_s: Direction.DOWN,
            pygame.K_a: Direction.LEFT, pygame.K_d: Direction.RIGHT,
        }
        if key not in mapping:
            return
        desired = mapping[key]
        if (self.current_direction.value + 2) % 4 == desired.value:
            return
            
        action = direction_to_relative(self.current_direction, desired)
        self.client.send_action(action)
        
        # 乐观更新方向，提升响应手感
        delta = {1: -1, 2: 1}.get(action, 0)
        self.current_direction = Direction((self.current_direction.value + delta) % 4)

    def _render_scene(self, screen: pygame.Surface, prev: Dict, curr: Dict, alpha: float, font: pygame.font.Font, big_font: pygame.font.Font) -> None:
        grid = int(curr.get("grid", 30))
        self.grid = grid
        self.cell = max(12, min(32, 640 // grid))
        game_area_size = grid * self.cell
        
        # 创建主绘图表面
        surface = pygame.Surface((game_area_size, game_area_size), pygame.SRCALPHA)
        
        # 1. 绘制背景与扫视线
        bg_color = (13, 15, 26)
        surface.fill(bg_color)
        
        # 绘制精细网格
        grid_color = (25, 30, 45)
        for i in range(grid + 1):
            pos = i * self.cell
            pygame.draw.line(surface, grid_color, (pos, 0), (pos, game_area_size), 1)
            pygame.draw.line(surface, grid_color, (0, pos), (game_area_size, pos), 1)
            
        # 2. 绘制食物 (带呼吸效果)
        food_color = (255, 0, 85)
        pulse = (np.sin(time.time() * 5) + 1) * 0.5 # 0 ~ 1
        for fx, fy in curr.get("food", []):
            center = ((fx + 0.5) * self.cell, (fy + 0.5) * self.cell)
            radius = self.cell * (0.35 + 0.05 * pulse)
            
            # 辉光
            glow_radius = radius * 2.5
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*food_color, int(40 + 20 * pulse)), (glow_radius, glow_radius), glow_radius)
            surface.blit(glow_surf, (center[0] - glow_radius, center[1] - glow_radius))
            
            # 核心
            pygame.draw.circle(surface, (255, 255, 255), center, radius)
            pygame.draw.circle(surface, food_color, center, radius, width=2)

        # 3. 绘制蛇 (插值 + 连续平滑渲染)
        prev_snakes = {s['slot']: s for s in prev.get("snakes", [])}
        curr_snakes = {s['slot']: s for s in curr.get("snakes", [])}
        
        for slot, curr_snake in curr_snakes.items():
            if not curr_snake.get("alive", False): continue
            
            base_color = self.snake_colors[slot % len(self.snake_colors)]
            prev_snake = prev_snakes.get(slot)
            
            curr_body = curr_snake.get("body", [])
            prev_body = prev_snake.get("body", []) if prev_snake else curr_body
            
            if not curr_body: continue
            
            # 计算插值后的身体各点
            interp_body = []
            max_len = max(len(curr_body), len(prev_body))
            for i in range(max_len):
                c_pt = curr_body[min(i, len(curr_body)-1)]
                p_pt = prev_body[min(i, len(prev_body)-1)]
                # 插值坐标
                ix = p_pt[0] + (c_pt[0] - p_pt[0]) * alpha
                iy = p_pt[1] + (c_pt[1] - p_pt[1]) * alpha
                interp_body.append((ix, iy))

            # 绘制身体连接
            for i in range(len(interp_body) - 1):
                p1 = interp_body[i]
                p2 = interp_body[i+1]
                
                # 绘制连接线段（圆角矩形效果）
                start_px = ((p1[0] + 0.5) * self.cell, (p1[1] + 0.5) * self.cell)
                end_px = ((p2[0] + 0.5) * self.cell, (p2[1] + 0.5) * self.cell)
                
                # 身体颜色渐变（越往后越暗）
                shade = max(0.4, 1.0 - (i / len(interp_body)) * 0.6)
                color = tuple(int(c * shade) for c in base_color)
                
                width = self.cell * 0.82
                pygame.draw.line(surface, color, start_px, end_px, int(width))
                pygame.draw.circle(surface, color, start_px, width // 2)
                pygame.draw.circle(surface, color, end_px, width // 2)

            # 绘制头部
            head = interp_body[0]
            head_px = ((head[0] + 0.5) * self.cell, (head[1] + 0.5) * self.cell)
            
            # 头部发光
            hg_radius = self.cell * 1.2
            hg_surf = pygame.Surface((hg_radius*2, hg_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(hg_surf, (*base_color, 100), (hg_radius, hg_radius), hg_radius)
            surface.blit(hg_surf, (head_px[0] - hg_radius, head_px[1] - hg_radius))
            
            # 头部实体 (白色高亮边缘)
            pygame.draw.circle(surface, (255, 255, 255), head_px, self.cell * 0.45)
            pygame.draw.circle(surface, base_color, head_px, self.cell * 0.45, width=3)
            
            # 绘制眼睛
            dir_enum = curr_snake.get("direction", "RIGHT")
            eye_offsets = {
                "UP": [(-0.15, -0.15), (0.15, -0.15)],
                "DOWN": [(-0.15, 0.15), (0.15, 0.15)],
                "LEFT": [(-0.15, -0.15), (-0.15, 0.15)],
                "RIGHT": [(0.15, -0.15), (0.15, 0.15)],
            }.get(dir_enum, [(0, 0)])
            
            for ox, oy in eye_offsets:
                eye_pos = (head_px[0] + ox * self.cell, head_px[1] + oy * self.cell)
                pygame.draw.circle(surface, (10, 15, 25), eye_pos, self.cell * 0.1)

        # 4. 最终合成与侧边栏
        screen.blit(surface, (40, 40))
        
        # 装饰性边框
        pygame.draw.rect(screen, (0, 243, 255), (38, 38, game_area_size + 4, game_area_size + 4), 2, border_radius=4)

        # 侧边栏
        self._render_ui(screen, curr, game_area_size + 80, 40, font, big_font)

        # 倒计时与游戏结束处理
        self._handle_overlays(screen, curr, game_area_size, font, big_font)

    def _render_ui(self, screen: pygame.Surface, state: Dict, x: int, y: int, font: pygame.font.Font, big_font: pygame.font.Font) -> None:
        panel = pygame.Surface((280, 640), pygame.SRCALPHA)
        pygame.draw.rect(panel, (20, 22, 35, 180), panel.get_rect(), border_radius=10)
        screen.blit(panel, (x, y))
        
        title = big_font.render("TERMINAL", True, (0, 243, 255))
        screen.blit(title, (x + 20, y + 20))
        
        stats = [
            f"ROOM: {self.room_id}",
            f"STEP: {state.get('steps', 0)}",
            f"ALIVE: {state.get('alive_count', 0)}",
        ]
        for i, text in enumerate(stats):
            lbl = font.render(text, True, (150, 160, 180))
            screen.blit(lbl, (x + 20, y + 80 + i * 30))
            
        # 计分板
        y_offset = y + 200
        pygame.draw.line(screen, (40, 45, 60), (x + 20, y_offset), (x + 260, y_offset))
        y_offset += 20
        
        snakes = state.get("snakes", [])
        scores = state.get("scores") or [s.get("score", 0) for s in snakes]
        
        for idx, snake in enumerate(snakes):
            color = self.snake_colors[idx % len(self.snake_colors)]
            name = snake.get("owner_name", f"Snake-{idx}")
            score = scores[idx] if idx < len(scores) else 0
            
            # 状态指示灯
            status_color = color if snake.get("alive") else (60, 60, 70)
            pygame.draw.circle(screen, status_color, (x + 30, y_offset + 12), 6)
            
            text = font.render(f"{name}: {score:.1f}", True, (220, 230, 240) if snake.get("alive") else (100, 110, 120))
            screen.blit(text, (x + 50, y_offset))
            y_offset += 35

    def _handle_overlays(self, screen: pygame.Surface, state: Dict, size: int, font: pygame.font.Font, big_font: pygame.font.Font) -> None:
        countdown_val = state.get("countdown")
        self.countdown_done = countdown_val is None or float(countdown_val) <= 0
        
        if not self.countdown_done:
            overlay = pygame.Surface((size, size), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            screen.blit(overlay, (40, 40))
            
            cd_text = str(int(float(countdown_val)) + 1) if float(countdown_val) > 0.1 else "READY"
            msg = big_font.render(cd_text, True, (0, 255, 159))
            rect = msg.get_rect(center=(40 + size // 2, 40 + size // 2))
            screen.blit(msg, rect)

        self.game_over = state.get("game_over", False)
        if self.game_over:
            overlay = pygame.Surface((960, 720), pygame.SRCALPHA)
            overlay.fill((10, 12, 20, 220))
            screen.blit(overlay, (0, 0))
            
            scores = state.get("scores") or [s.get("score", 0) for s in state.get("snakes", [])]
            mvp_idx = np.argmax(scores) if scores else 0
            mvp_name = state.get("snakes", [])[mvp_idx].get("owner_name", "None") if state.get("snakes") else "Unknown"
            
            finish_msg = big_font.render("MISSION ACCOMPLISHED", True, (255, 204, 0))
            mvp_msg = font.render(f"WINNER: {mvp_name} | SCORE: {scores[mvp_idx] if scores else 0}", True, (255, 255, 255))
            tip_msg = font.render("Press any key to return", True, (150, 160, 180))
            
            screen.blit(finish_msg, (480 - finish_msg.get_width() // 2, 300))
            screen.blit(mvp_msg, (480 - mvp_msg.get_width() // 2, 360))
            screen.blit(tip_msg, (480 - tip_msg.get_width() // 2, 450))
