"""局域网客户端，支持联机大厅、AI 推理与本地训练/对练。"""

from __future__ import annotations

import json
import socket
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pygame
import torch

from agent.ppo import PPOAgent
from agent.train import TrainConfig, train
from env.multi_snake_env import Direction, build_observation_from_snapshot
from network.constants import COLOR_MAP, COLOR_SEQUENCE
from network.local_arena import start_local_arena
from network.utils import direction_to_relative


@dataclass
class PlayerInfo:
    player_id: int
    name: str
    mode: str
    ready: bool
    slot: Optional[int]


@dataclass
class UIButton:
    rect: pygame.Rect
    label: str
    action: Callable[[], None]
    enabled: bool = True

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        base_color = (75, 115, 210) if self.enabled else (60, 60, 75)
        hover_color = (95, 145, 240)
        mouse_pos = pygame.mouse.get_pos()
        fill = hover_color if self.enabled and self.rect.collidepoint(mouse_pos) else base_color
        pygame.draw.rect(surface, fill, self.rect, border_radius=10)
        pygame.draw.rect(surface, (20, 20, 35), self.rect, width=2, border_radius=10)
        text_color = (255, 255, 255) if self.enabled else (150, 150, 150)
        text = font.render(self.label, True, text_color)
        surface.blit(text, text.get_rect(center=self.rect.center))


_dir_to_relative = direction_to_relative


class GameClient:
    """负责大厅渲染、玩家模式切换、AI 推理与输入。"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        name: str | None = None,
        ai_model_hint: str | None = None,
    ) -> None:
        """保存连接参数并初始化 UI、状态机、锁等运行期变量。"""
        self.host = host
        self.port = port
        self.name = (name or "Player").strip() or "Player"
        self.ai_model_hint = ai_model_hint.strip() if ai_model_hint else None
        self.ai_agent: Optional[PPOAgent] = None
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.cell_size = 30
        self.grid_width = 30
        self.grid_height = 30
        self.grid_pixel_width = self.grid_width * self.cell_size
        self.grid_pixel_height = self.grid_height * self.cell_size

        self.running = False
        self.phase = "lobby"
        self.player_id: Optional[int] = None
        self.host_id: Optional[int] = None
        self.player_mode = "player"
        self.ready = False
        self.slot: Optional[int] = None
        self.my_color_name: Optional[str] = None
        self.current_direction: Direction = Direction.RIGHT

        self.lobby_players: List[PlayerInfo] = []
        self.state_snapshot: Dict | None = None
        self.last_game_over: Dict | None = None
        self.tip_message: str = ""
        self.tip_timer: int = 0
        self.await_restart_choice = False
        self.countdown_seconds = 0
        self.assignment_hint: Optional[str] = None
        self.assignment_hint_timer = 0

        self.screen: Optional[pygame.Surface] = None
        self.font: Optional[pygame.font.Font] = None
        self.small_font: Optional[pygame.font.Font] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.current_buttons: List[UIButton] = []
        self.restart_buttons: List[UIButton] = []

        self.lock = threading.Lock()

    # ------------------------------------------------------------------
    # 网络通信
    # ------------------------------------------------------------------
    def connect(self) -> bool:
        """尝试连接服务器，成功返回 True，失败打印错误。"""
        try:
            self.client_socket.connect((self.host, self.port))
            print(f"Connected to {self.host}:{self.port}")
            return True
        except OSError as exc:
            print(f"Connection failed: {exc}")
            return False

    def send_json(self, payload: Dict) -> None:
        """向服务器发送一行 JSON，失败时会结束运行循环。"""
        try:
            self.client_socket.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        except OSError:
            self.running = False

    def receive_data(self) -> None:
        """后台线程循环接收服务器消息并解析为 JSON。"""
        buffer = ""
        while self.running:
            try:
                data = self.client_socket.recv(4096).decode("utf-8")
                if not data:
                    break
                buffer += data
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._handle_message(msg)
            except OSError:
                break
        print("Disconnected from server")
        self.running = False

    def _handle_message(self, msg: Dict) -> None:
        """根据消息类型更新大厅、游戏状态或提示。"""
        mtype = msg.get("type")
        if mtype == "welcome":
            self.player_id = msg.get("player_id")
            self.host_id = msg.get("host_id")
            self.send_json({"type": "join", "name": self.name, "mode": self.player_mode})
        elif mtype == "lobby":
            players = [
                PlayerInfo(
                    player_id=entry.get("id"),
                    name=entry.get("name", "Player"),
                    mode=entry.get("mode", "player"),
                    ready=entry.get("ready", False),
                    slot=entry.get("slot"),
                )
                for entry in msg.get("players", [])
            ]
            with self.lock:
                self.lobby_players = players
                self.phase = msg.get("phase", self.phase)
                self.host_id = msg.get("host_id", self.host_id)
                lobby_grid = msg.get("grid_size")
                if lobby_grid:
                    self._update_grid_dimensions(int(lobby_grid), int(lobby_grid))
                for entry in players:
                    if entry.player_id == self.player_id:
                        self.player_mode = entry.mode
                        self.ready = entry.ready
                        self.slot = entry.slot
        elif mtype == "start":
            self.slot = msg.get("slot")
            self.player_mode = msg.get("mode", self.player_mode)
            color = msg.get("color")
            if color:
                self.my_color_name = color
            if self.slot is not None:
                hint_color = color or self._current_color_name()
                self.assignment_hint = f"本局你将控制蛇 {self.slot}（颜色 {hint_color}）"
                self.assignment_hint_timer = 240
        elif mtype == "state":
            with self.lock:
                width = int(msg.get("width", self.grid_width))
                height = int(msg.get("height", self.grid_height))
                self._update_grid_dimensions(width, height)
                phase_value = msg.get("phase", "game")
                self.phase = phase_value
                self.state_snapshot = msg
                if self.slot is not None:
                    snakes = msg.get("snakes", [])
                    if 0 <= self.slot < len(snakes):
                        snake = snakes[self.slot]
                        dir_name = snake.get("direction")
                        if dir_name:
                            try:
                                self.current_direction = Direction[dir_name]
                            except KeyError:
                                pass
                        self.my_color_name = snake.get("color", self.my_color_name)
                self.await_restart_choice = False
                self.restart_buttons = []
                if phase_value != "countdown":
                    self.countdown_seconds = 0
            self._maybe_drive_ai(msg)
        elif mtype == "game_over":
            with self.lock:
                self.last_game_over = msg
                self.phase = "lobby"
                self.await_restart_choice = True
                self.ready = False
                self.restart_buttons = []
                self.countdown_seconds = 0
        elif mtype == "tip":
            self.tip_message = msg.get("message", "")
            self.tip_timer = 180
        elif mtype == "countdown":
            seconds = int(msg.get("seconds", 0))
            with self.lock:
                self.countdown_seconds = max(0, seconds)
                self.phase = msg.get("phase", self.phase)

    # ------------------------------------------------------------------
    # 主循环及输入处理
    # ------------------------------------------------------------------
    def run(self) -> None:
        """启动 Pygame 主循环，处理事件、渲染和退出。"""
        if not self.connect():
            return

        self.running = True
        threading.Thread(target=self.receive_data, daemon=True).start()

        pygame.init()
        self.font = self._load_font()
        self.small_font = self._load_font(18)
        self._resize_window()
        pygame.display.set_caption("多蛇博弈 - 联机大厅")
        self.clock = pygame.time.Clock()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    else:
                        self._handle_key(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)

            if not self.screen:
                continue

            self.screen.fill((18, 20, 28))
            with self.lock:
                phase = self.phase
                state = self.state_snapshot.copy() if self.state_snapshot else None
                lobby_players = list(self.lobby_players)
                last_game_over = self.last_game_over
                tip_message = self.tip_message
                tip_timer = self.tip_timer
                countdown_seconds = self.countdown_seconds if self.phase == "countdown" else 0
                assignment_hint = self.assignment_hint
                assignment_hint_timer = self.assignment_hint_timer
                if self.tip_timer > 0:
                    self.tip_timer -= 1
                if self.assignment_hint_timer > 0:
                    self.assignment_hint_timer -= 1
                else:
                    self.assignment_hint = None

            if phase == "lobby":
                self._render_lobby(lobby_players)
            else:
                self._render_game(
                    state,
                    countdown_seconds=countdown_seconds,
                    assignment_hint=assignment_hint if assignment_hint_timer > 0 else None,
                )
                if last_game_over:
                    self._render_game_over(last_game_over)

            if tip_message and tip_timer > 0:
                self._render_tip(tip_message)

            if self.await_restart_choice:
                self._render_restart_overlay()

            pygame.display.flip()
            if self.clock:
                self.clock.tick(30)

        self.client_socket.close()
        pygame.quit()
        return

    def _handle_key(self, key: int) -> None:
        """根据当前阶段处理快捷键：大厅换模式，游戏中控制方向。"""
        if self.phase == "lobby":
            if key in (pygame.K_TAB, pygame.K_F1):
                self._set_mode("player")
            elif key in (pygame.K_a, pygame.K_F2):
                self._set_mode("ai")
            elif key in (pygame.K_o, pygame.K_F3):
                self._set_mode("observer")
            elif key == pygame.K_SPACE:
                self._toggle_ready()
        else:
            if self.slot is None or self.player_mode != "player":
                return
            if key in (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT):
                self._handle_direction_key(key)

    def _handle_click(self, pos: tuple[int, int]) -> None:
        """响应鼠标左键，触发当前激活按钮或重开按钮。"""
        targets = self.restart_buttons if self.await_restart_choice else self.current_buttons
        for button in targets:
            if button.enabled and button.rect.collidepoint(pos):
                button.action()
                break

    def _handle_direction_key(self, key: int) -> None:
        """将绝对方向键转换为相对动作并发送到服务器。"""
        key_map = {
            pygame.K_UP: Direction.UP,
            pygame.K_DOWN: Direction.DOWN,
            pygame.K_LEFT: Direction.LEFT,
            pygame.K_RIGHT: Direction.RIGHT,
        }
        desired = key_map.get(key)
        if desired is None:
            return
        current = self.current_direction or desired
        if (current.value + 2) % 4 == desired.value:
            return  # 禁止直接调头
        relative = _dir_to_relative(current, desired)
        self.send_json({"type": "action", "value": relative})
        delta = {0: 0, 1: -1, 2: 1}.get(relative, 0)
        self.current_direction = Direction((current.value + delta) % 4)

    def _set_mode(self, mode: str) -> None:
        """切换玩家模式，同时重置准备状态和 AI 模型缓存。"""
        if self.player_mode == mode:
            return
        self.player_mode = mode
        self.ready = False
        self.ai_agent = None  # 重新加载模型
        self.send_json({"type": "mode", "mode": mode})

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------
    def _render_lobby(self, players: List[PlayerInfo]) -> None:
        """绘制大厅 UI：玩家列表、提示文本与操作按钮。"""
        if not self.screen or not self.font or not self.small_font:
            return
        screen = self.screen
        font = self.font
        small_font = self.small_font
        self.current_buttons = []

        panel = pygame.Surface((screen.get_width() - 80, screen.get_height() - 120), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 85))
        screen.blit(panel, (40, 60))

        title = font.render("多人联机大厅", True, (255, 255, 255))
        screen.blit(title, (60, 30))

        instructions = [
            "F1/TAB: 人工模式",
            "F2/A: AI 模式",
            "F3/O: 观察模式",
            "SPACE: 准备/取消",
            "ENTER: 房主开始 (≥2 且全员准备)",
            "ESC: 退出客户端",
        ]
        for idx, text in enumerate(instructions):
            surface = small_font.render(text, True, (200, 200, 210))
            screen.blit(surface, (60, 80 + idx * 22))

        header = small_font.render("玩家 | 模式 | 状态 | 槽位", True, (180, 180, 190))
        screen.blit(header, (60, 230))

        row_y = 260
        row_height = 30
        for p in players:
            bg_color = (60, 70, 90) if p.player_id == self.player_id else (40, 45, 58)
            row_surface = pygame.Surface((screen.get_width() - 120, row_height))
            row_surface.fill(bg_color)
            screen.blit(row_surface, (60, row_y))

            label = f"{'[房主]' if p.player_id == self.host_id else '      '}  {p.player_id:02d}  {p.name:<12}  {self._mode_label(p.mode):<4}  {'准备' if p.ready else '待命'}  槽位:{p.slot if p.slot is not None else '-'}"
            color = (120, 220, 150) if p.ready else (220, 220, 220)
            text_surface = small_font.render(label, True, color)
            screen.blit(text_surface, (70, row_y + 6))
            row_y += row_height + 6

        self._render_color_badge(screen, small_font)
        self._build_lobby_buttons(screen)
        for button in self.current_buttons:
            button.draw(screen, small_font)

    def _render_color_badge(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        """展示本方蛇颜色与当前模式/模型信息。"""
        color_name = self._current_color_name()
        color_rgb = COLOR_MAP.get(color_name, (180, 180, 180))
        badge_rect = pygame.Rect(60, screen.get_height() - 90, 42, 42)
        pygame.draw.rect(screen, color_rgb, badge_rect, border_radius=6)
        pygame.draw.rect(screen, (240, 240, 240), badge_rect, width=2, border_radius=6)
        label = font.render(f"我的蛇颜色: {color_name}", True, (230, 230, 230))
        screen.blit(label, (badge_rect.right + 12, badge_rect.y + 8))
        mode_hint = font.render(f"当前模式: {self._mode_label(self.player_mode)} | 模型: {self.ai_model_hint or '本地启发式'}", True, (200, 200, 210))
        screen.blit(mode_hint, (60, screen.get_height() - 45))

    def _build_lobby_buttons(self, screen: pygame.Surface) -> None:
        """根据身份和状态创建模式切换、准备、开局按钮。"""
        base_x = screen.get_width() - 260
        y = screen.get_height() - 220
        btn_w, btn_h = 200, 44
        spacing = 18
        buttons: List[UIButton] = []

        buttons.append(
            UIButton(
                pygame.Rect(base_x, y, btn_w, btn_h),
                f"模式: {self._mode_label(self.player_mode)}",
                self._cycle_mode,
                True,
            )
        )
        y += btn_h + spacing

        buttons.append(
            UIButton(
                pygame.Rect(base_x, y, btn_w, btn_h),
                "取消准备" if self.ready else "准备就绪",
                self._toggle_ready,
                True,
            )
        )
        y += btn_h + spacing

        is_host = self.player_id == self.host_id
        start_enabled = is_host and self.phase == "lobby" and not self.await_restart_choice
        start_label = "开始游戏" if is_host else "等待房主"
        buttons.append(
            UIButton(
                pygame.Rect(base_x, y, btn_w, btn_h),
                start_label,
                self._request_start,
                start_enabled,
            )
        )

        self.current_buttons = buttons

    def _current_color_name(self) -> str:
        """返回当前玩家蛇的颜色名称（未分配则回退为占位文本）。"""
        if self.my_color_name:
            return self.my_color_name
        if self.slot is not None and 0 <= self.slot < len(COLOR_SEQUENCE):
            return COLOR_SEQUENCE[self.slot]
        return "未分配"

    def _cycle_mode(self) -> None:
        """按 player->ai->observer 顺序依次轮换模式。"""
        order = ["player", "ai", "observer"]
        idx = order.index(self.player_mode) if self.player_mode in order else 0
        next_mode = order[(idx + 1) % len(order)]
        self._set_mode(next_mode)

    def _toggle_ready(self) -> None:
        """切换本人的准备状态并同步给服务器。"""
        self.ready = not self.ready
        self.send_json({"type": "ready", "ready": self.ready})

    def _request_start(self) -> None:
        """房主点击“开始游戏”时发送 start_request。"""
        if self.player_id != self.host_id:
            return
        self.send_json({"type": "start_request"})

    def _render_game(
        self,
        state: Optional[Dict],
        *,
        countdown_seconds: int = 0,
        assignment_hint: Optional[str] = None,
    ) -> None:
        """渲染对局中的网格、蛇、统计信息与提示。"""
        if not self.screen or not self.font or not self.small_font:
            return
        screen = self.screen
        font = self.font
        small_font = self.small_font
        self.current_buttons = []

        if not state:
            waiting = font.render("等待服务器状态...", True, (255, 255, 255))
            screen.blit(waiting, (60, 40))
            if assignment_hint:
                self._render_assignment_hint(assignment_hint)
            if countdown_seconds:
                self._render_countdown_overlay(countdown_seconds)
            return

        grid_surface = pygame.Surface((self.grid_pixel_width, self.grid_pixel_height))
        grid_surface.fill((12, 14, 20))
        self._draw_grid(grid_surface)
        self._draw_food(grid_surface, state.get("food", []))
        self._draw_snakes(grid_surface, state.get("snakes", []))
        screen.blit(grid_surface, (60, 40))

        info_panel = pygame.Surface((260, self.grid_pixel_height), pygame.SRCALPHA)
        info_panel.fill((0, 0, 0, 120))
        screen.blit(info_panel, (self.grid_pixel_width + 100, 40))

        stats = [
            f"步数: {state.get('steps', 0)}",
            f"存活: {state.get('alive_count', 0)}",
            f"地图: {self.grid_width}x{self.grid_height}",
        ]
        for idx, line in enumerate(stats):
            surface = small_font.render(line, True, (240, 240, 240))
            screen.blit(surface, (self.grid_pixel_width + 120, 60 + idx * 26))

        scores = state.get("scores", [])
        for idx, score in enumerate(scores):
            text = small_font.render(f"蛇 {idx}: {score}", True, (200, 200, 200))
            screen.blit(text, (self.grid_pixel_width + 120, 160 + idx * 24))

        hint = small_font.render("方向键控制蛇，空格回到大厅面板。", True, (220, 220, 230))
        screen.blit(hint, (60, self.grid_pixel_height + 60))
        self._render_color_badge(screen, small_font)

        if assignment_hint:
            self._render_assignment_hint(assignment_hint)
        if countdown_seconds:
            self._render_countdown_overlay(countdown_seconds)

    def _render_game_over(self, summary: Dict) -> None:
        """在屏幕底部展示上一局的得分概览。"""
        if not self.screen or not self.font:
            return
        overlay = pygame.Surface((self.screen.get_width(), 140), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, self.screen.get_height() - 160))

        title = self.font.render("本局结束", True, (255, 230, 160))
        self.screen.blit(title, (80, self.screen.get_height() - 150))
        for idx, score in enumerate(summary.get("scores", [])):
            text = self.font.render(f"蛇 {idx}: {score}", True, (220, 220, 220))
            self.screen.blit(text, (80, self.screen.get_height() - 110 + idx * 26))

    def _render_assignment_hint(self, message: str) -> None:
        """在屏幕底部展示当前槽位提示。"""
        if not self.screen or not self.small_font:
            return
        panel = pygame.Surface((self.screen.get_width(), 50), pygame.SRCALPHA)
        panel.fill((15, 15, 20, 200))
        self.screen.blit(panel, (0, self.screen.get_height() - 70))
        text = self.small_font.render(message, True, (255, 235, 200))
        self.screen.blit(
            text,
            (self.screen.get_width() // 2 - text.get_width() // 2, self.screen.get_height() - 60),
        )

    def _render_countdown_overlay(self, seconds: int) -> None:
        """在联机对战开始前渲染倒计时。"""
        if not self.screen or not self.font:
            return
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.screen.blit(overlay, (0, 0))
        label = self.font.render(str(seconds), True, (255, 245, 200))
        self.screen.blit(
            label,
            (self.screen.get_width() // 2 - label.get_width() // 2, self.screen.get_height() // 2 - 70),
        )
        sub = self.small_font.render("倒计时结束后自动开局", True, (235, 235, 235))
        self.screen.blit(
            sub,
            (self.screen.get_width() // 2 - sub.get_width() // 2, self.screen.get_height() // 2),
        )

    def _render_restart_overlay(self) -> None:
        """当等待玩家决定下一局时，绘制按钮覆盖层。"""
        if not self.screen or not self.font or not self.small_font:
            return
        width, height = self.screen.get_size()
        panel = pygame.Surface((width, 220), pygame.SRCALPHA)
        panel.fill((10, 10, 10, 210))
        self.screen.blit(panel, (0, height // 2 - 110))

        title = self.font.render("本局结束，要继续挑战吗？", True, (255, 230, 180))
        self.screen.blit(title, (width // 2 - title.get_width() // 2, height // 2 - 80))
        subtitle = self.small_font.render("选择“继续作战”会自动准备下一局，或退出客户端休息。", True, (220, 220, 230))
        self.screen.blit(subtitle, (width // 2 - subtitle.get_width() // 2, height // 2 - 40))

        btn_w, btn_h = 180, 50
        spacing = 30
        start_x = width // 2 - btn_w - spacing // 2
        y = height // 2 + 10
        buttons = [
            UIButton(pygame.Rect(start_x, y, btn_w, btn_h), "继续作战", lambda: self._choose_restart(True), True),
            UIButton(
                pygame.Rect(start_x + btn_w + spacing, y, btn_w, btn_h),
                "休息一下",
                lambda: self._choose_restart(False),
                True,
            ),
            UIButton(
                pygame.Rect(width // 2 - btn_w // 2, y + btn_h + spacing, btn_w, btn_h),
                "退出游戏",
                self._quit_client,
                True,
            ),
        ]
        self.restart_buttons = buttons
        for button in buttons:
            button.draw(self.screen, self.small_font)

    def _choose_restart(self, ready: bool) -> None:
        """处理重开按钮：设置 ready 状态并通知服务器。"""
        self.await_restart_choice = False
        self.restart_buttons = []
        self.ready = ready
        self.send_json({"type": "ready", "ready": ready})

    def _quit_client(self) -> None:
        """退出按钮回调，停止主循环并关闭窗口。"""
        self.await_restart_choice = False
        self.running = False

    def _render_tip(self, message: str) -> None:
        """在顶部显示短暂提示条，如条件不足、切换模式提醒。"""
        if not self.screen or not self.small_font:
            return
        panel = pygame.Surface((self.screen.get_width(), 40), pygame.SRCALPHA)
        panel.fill((20, 20, 20, 180))
        self.screen.blit(panel, (0, 0))
        surface = self.small_font.render(message, True, (255, 200, 120))
        self.screen.blit(surface, (40, 10))

    def _draw_grid(self, surface: pygame.Surface) -> None:
        """绘制棋盘格线，辅助观察位置。"""
        for x in range(0, self.grid_pixel_width, self.cell_size):
            pygame.draw.line(surface, (40, 40, 45), (x, 0), (x, self.grid_pixel_height))
        for y in range(0, self.grid_pixel_height, self.cell_size):
            pygame.draw.line(surface, (40, 40, 45), (0, y), (self.grid_pixel_width, y))

    def _draw_food(self, surface: pygame.Surface, foods: List[List[int]]) -> None:
        """根据 food 列表绘制红色食物块。"""
        for fx, fy in foods:
            rect = pygame.Rect(fx * self.cell_size, fy * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(surface, (250, 120, 120), rect)

    def _draw_snakes(self, surface: pygame.Surface, snakes: List[Dict]) -> None:
        """遍历蛇列表，按颜色绘制身体与蛇头高亮。"""
        for snake in snakes:
            if not snake.get("alive", False):
                continue
            color = COLOR_MAP.get(snake.get("color", "green"), (0, 200, 0))
            for idx, (bx, by) in enumerate(snake.get("body", [])):
                rect = pygame.Rect(bx * self.cell_size, by * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(surface, color, rect)
                pygame.draw.rect(surface, (0, 0, 0), rect, 1)
                if idx == 0:
                    pygame.draw.rect(surface, (255, 255, 255), rect, 2)

    # ------------------------------------------------------------------
    # AI 推理与尺寸管理
    # ------------------------------------------------------------------
    def _update_grid_dimensions(self, width: int, height: int) -> None:
        """当服务器广播的地图尺寸变化时，更新像素尺寸并重建窗口。"""
        width = max(6, width)
        height = max(6, height)
        if width == self.grid_width and height == self.grid_height:
            return
        self.grid_width = width
        self.grid_height = height
        self.grid_pixel_width = self.grid_width * self.cell_size
        self.grid_pixel_height = self.grid_height * self.cell_size
        self.ai_agent = None  # 网格变化后需重建模型
        self._resize_window()

    def _resize_window(self) -> None:
        """根据当前网格像素宽高调节显示窗口大小。"""
        if not pygame.get_init():
            return
        display_width = max(self.grid_pixel_width + 320, 1024)
        display_height = max(self.grid_pixel_height + 120, 640)
        self.screen = pygame.display.set_mode((display_width, display_height))

    def _maybe_drive_ai(self, state: Dict) -> None:
        """在 AI 模式下，根据最新状态预测动作并发送。"""
        if self.phase != "game" or self.player_mode != "ai" or self.slot is None:
            return
        action = self._predict_action(state)
        if action is not None:
            self.send_json({"type": "action", "value": int(action)})

    def _predict_action(self, state: Dict) -> Optional[int]:
        """生成 AI 动作：优先使用加载的模型，否则回退启发式策略。"""
        snakes = state.get("snakes", [])
        if self.slot is None or self.slot >= len(snakes):
            return None

        width = int(state.get("width", self.grid_width))
        height = int(state.get("height", self.grid_height))

        obs = build_observation_from_snapshot(
            width=width,
            height=height,
            snakes=snakes,
            food=state.get("food", []),
            slot=self.slot,
        )

        if self.ai_model_hint:
            if not self._ensure_ai_agent(width):
                return self._heuristic_action(state)
            return self.ai_agent.predict(obs) if self.ai_agent else 0
        return self._heuristic_action(state)

    def _heuristic_action(self, state: Dict) -> int:
        """简单启发式：沿着最近食物方向移动。"""
        snakes = state.get("snakes", [])
        foods = state.get("food", [])
        if self.slot is None or self.slot >= len(snakes) or not foods:
            return 0
        snake = snakes[self.slot]
        if not snake.get("alive", False):
            return 0
        head_x, head_y = snake.get("body", [[0, 0]])[0]
        target = min(foods, key=lambda pos: abs(pos[0] - head_x) + abs(pos[1] - head_y))
        dx = target[0] - head_x
        dy = target[1] - head_y
        dir_name = snake.get("direction", "RIGHT")
        try:
            current_dir = Direction[dir_name]
        except KeyError:
            current_dir = Direction.RIGHT

        desired = current_dir
        if abs(dx) > abs(dy):
            desired = Direction.RIGHT if dx > 0 else Direction.LEFT
        elif dy != 0:
            desired = Direction.DOWN if dy > 0 else Direction.UP
        return _dir_to_relative(current_dir, desired)

    def _ensure_ai_agent(self, grid_size: int) -> bool:
        """确保已成功加载 AI 模型，若缺失或出错则回退。"""
        if not self.ai_model_hint:
            return False
        if self.ai_agent is not None:
            return True
        model_path = Path(self.ai_model_hint)
        if not model_path.exists():
            print(f"[Client] 模型不存在：{model_path}")
            self.ai_model_hint = None
            return False
        try:
            agent = PPOAgent(input_channels=3, grid_size=grid_size, action_dim=3)
            state_dict = torch.load(model_path, map_location="cpu")
            agent.policy.load_state_dict(state_dict)
            agent.policy.eval()
            self.ai_agent = agent
            print(f"[Client] 成功加载模型 {model_path}")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[Client] 加载模型失败: {exc}")
            self.ai_model_hint = None
            return False

    def _load_font(self, size: int = 22) -> pygame.font.Font:
        """加载中文字体，失败时回退到 Arial。"""
        try:
            return pygame.font.SysFont("simhei", size)
        except Exception:  # noqa: BLE001
            return pygame.font.SysFont("arial", size)

    @staticmethod
    def _mode_label(mode: str) -> str:
        """将内部模式标识映射为中文标签。"""
        return {"player": "人工", "ai": "AI", "observer": "观察"}.get(mode, mode)


def start_network_client(host: str = "127.0.0.1", port: int = 5555, name: str = "Player", ai_model: Optional[str] = None) -> None:
    """命令式入口：根据参数实例化并运行 GameClient。"""
    GameClient(host=host, port=port, name=name, ai_model_hint=ai_model).run()


def start_local_training(config: Optional[TrainConfig] = None) -> None:
    """触发 PPO 训练，如未传配置则使用默认参数。"""
    cfg = config or TrainConfig()
    train(cfg)


def launch_client_gui() -> None:
    """启动三选项卡 GUI，统一入口管理训练/对练/联机。"""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    speed_presets = {
        "慢速学习 (0.18s/步)": {"sim_interval": 0.18, "fps": 24},
        "标准流畅 (0.12s/步)": {"sim_interval": 0.12, "fps": 30},
        "快速挑战 (0.08s/步)": {"sim_interval": 0.08, "fps": 45},
    }

    root = tk.Tk()
    root.title("多蛇客户端控制台")
    root.geometry("520x480")
    root.configure(bg="#0f111a")

    style = ttk.Style()
    style.theme_use("default")
    style.configure("TNotebook", background="#0f111a", borderwidth=0)
    style.configure("TNotebook.Tab", padding=[12, 6], font=("Microsoft YaHei", 11))
    style.map("TNotebook.Tab", background=[("selected", "#1f2435")], foreground=[("selected", "#ffffff")])

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=16, pady=16)

    def labeled_entry(parent, text, default=""):
        """创建带标签的输入框，返回可读写的 StringVar。"""
        frame = tk.Frame(parent, bg="#181b2a")
        frame.pack(fill="x", pady=6)
        tk.Label(frame, text=text, fg="white", bg="#181b2a", anchor="w").pack(anchor="w")
        var = tk.StringVar(value=default)
        entry = tk.Entry(frame, textvariable=var, bg="#243046", fg="white", insertbackground="white", relief="flat")
        entry.pack(fill="x", pady=2)
        return var

    # 训练 Tab
    train_tab = tk.Frame(notebook, bg="#181b2a")
    notebook.add(train_tab, text="模型训练")
    tk.Label(train_tab, text="配置 PPO 训练参数，点击开始后将在当前窗口输出日志。", bg="#181b2a", fg="#cfd8ff").pack(pady=8, anchor="w", padx=8)
    train_grid = labeled_entry(train_tab, "网格大小", "30")
    train_snakes = labeled_entry(train_tab, "蛇数量", "4")
    train_eps = labeled_entry(train_tab, "训练轮数", "2000")
    train_device = labeled_entry(train_tab, "训练设备 (auto/cpu/cuda:0)", "auto")

    def launch_training() -> None:
        """读取训练参数并启动 PPO 训练流程。"""
        try:
            cfg = TrainConfig(
                grid_size=int(train_grid.get() or 30),
                num_snakes=int(train_snakes.get() or 4),
                max_episodes=int(train_eps.get() or 2000),
                device=(train_device.get().strip() or "auto"),
            )
        except ValueError:
            messagebox.showerror("错误", "训练参数必须为数字")
            return
        root.destroy()
        start_local_training(cfg)

    tk.Button(train_tab, text="开始训练", command=launch_training, bg="#3a7bd5", fg="white", relief="flat").pack(pady=16, fill="x", padx=8)

    # 本地对练 Tab
    arena_tab = tk.Frame(notebook, bg="#181b2a")
    notebook.add(arena_tab, text="本地对练")
    tk.Label(
        arena_tab,
        text="填写参数后点击“启动本地对练”，进入 Pygame 窗口，用方向键操控指定蛇。",
        bg="#181b2a",
        fg="#cfd8ff",
        wraplength=460,
        justify="left",
    ).pack(pady=8, anchor="w", padx=8)
    arena_grid = labeled_entry(arena_tab, "地图大小", "30")
    arena_snakes = labeled_entry(arena_tab, "蛇数量", "4")
    arena_model = labeled_entry(arena_tab, "AI 模型路径 (可选)")
    arena_device = labeled_entry(arena_tab, "AI 推理设备 (auto/cpu/cuda:0)", "auto")
    human_var = tk.BooleanVar(value=True)
    tk.Checkbutton(
        arena_tab,
        text="人工操控第一条蛇",
        variable=human_var,
        bg="#181b2a",
        fg="white",
        selectcolor="#243046",
        activebackground="#181b2a",
    ).pack(anchor="w", padx=8, pady=6)

    tk.Label(arena_tab, text="速度/流畅度预设", bg="#181b2a", fg="#cfd8ff").pack(anchor="w", padx=8, pady=(4, 0))
    speed_var = tk.StringVar(value="标准流畅 (0.12s/步)")
    speed_menu = tk.OptionMenu(arena_tab, speed_var, *speed_presets.keys())
    speed_menu.configure(bg="#2b3248", fg="white", activebackground="#3a425e", highlightthickness=0)
    speed_menu.pack(anchor="w", padx=8, pady=4)

    def browse_arena_model() -> None:
        """选择本地对练所需的模型文件。"""
        path = filedialog.askopenfilename(title="选择模型", filetypes=[("PyTorch Model", "*.pth"), ("所有文件", "*")])
        if path:
            arena_model.set(path)

    tk.Button(arena_tab, text="浏览模型", command=browse_arena_model, bg="#2b3248", fg="white", relief="flat").pack(pady=4, padx=8, anchor="e")

    def launch_arena() -> None:
        """校验本地对练参数并开启 Pygame 对练窗口。"""
        try:
            grid = int(arena_grid.get() or 30)
            snakes = int(arena_snakes.get() or 4)
        except ValueError:
            messagebox.showerror("错误", "地图与蛇数量必须为整数")
            return
        model = arena_model.get().strip() or None
        preset = speed_presets.get(speed_var.get(), speed_presets["标准流畅 (0.12s/步)"])
        root.destroy()
        start_local_arena(
            grid_size=grid,
            num_snakes=snakes,
            human_player=human_var.get(),
            model_path=model,
            target_fps=int(preset["fps"]),
            sim_interval=float(preset["sim_interval"]),
            device=arena_device.get().strip() or "auto",
        )

    tk.Button(arena_tab, text="启动本地对练", command=launch_arena, bg="#3a7bd5", fg="white", relief="flat").pack(pady=16, fill="x", padx=8)

    # 联机 Tab
    client_tab = tk.Frame(notebook, bg="#181b2a")
    notebook.add(client_tab, text="联机大厅")
    tk.Label(client_tab, text="填写服务器参数后进入大厅，支持人工/AI/观察模式切换。", bg="#181b2a", fg="#cfd8ff").pack(pady=8, anchor="w", padx=8)
    host_var = labeled_entry(client_tab, "服务器 IP", "127.0.0.1")
    port_var = labeled_entry(client_tab, "端口", "5555")
    name_var = labeled_entry(client_tab, "昵称", "Player")
    model_var = labeled_entry(client_tab, "AI 模型路径 (可选)")

    def browse_client_model() -> None:
        """为联机 AI 模式挑选本地模型文件。"""
        path = filedialog.askopenfilename(title="选择模型", filetypes=[("PyTorch Model", "*.pth"), ("所有文件", "*")])
        if path:
            model_var.set(path)

    tk.Button(client_tab, text="浏览模型", command=browse_client_model, bg="#2b3248", fg="white", relief="flat").pack(pady=4, padx=8, anchor="e")

    def launch_client() -> None:
        """读取服务器参数并启动 Pygame 联机客户端。"""
        try:
            port = int(port_var.get() or 5555)
        except ValueError:
            messagebox.showerror("错误", "端口必须是数字")
            return
        ai_path = model_var.get().strip() or None
        root.destroy()
        start_network_client(host=host_var.get() or "127.0.0.1", port=port, name=name_var.get() or "Player", ai_model=ai_path)

    tk.Button(client_tab, text="启动联机客户端", command=launch_client, bg="#3a7bd5", fg="white", relief="flat").pack(pady=16, fill="x", padx=8)

    tk.Button(root, text="退出控制台", command=root.destroy, bg="#2b3248", fg="white", relief="flat").pack(pady=4, fill="x", padx=16)
    root.mainloop()


if __name__ == "__main__":
    launch_client_gui()
