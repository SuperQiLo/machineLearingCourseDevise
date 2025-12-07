"""局域网服务器，提供大厅与 authoritative 游戏逻辑。"""

from __future__ import annotations

import json
import math
import socket
import threading
import time
from typing import Dict, List, Optional

from env.multi_snake_env import Direction, MultiSnakeEnv


class GameServer:
    """Server-Authoritative 多蛇服务器，包含大厅、准备和观战。"""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5555,
        max_players: int = 8,
        grid_size: int = 30,
        num_food: Optional[int] = None,
        tick_rate: float = 0.15,
        countdown_duration: float = 3.0,
    ) -> None:
        """初始化服务器网络 socket、大厅状态与环境配置。"""
        self.host = host
        self.port = port
        self.max_players = max_players
        self.grid_size = grid_size
        self.food_count = num_food if num_food is not None else max(4, grid_size // 4)
        self.tick_rate = tick_rate
        self.countdown_duration = countdown_duration
        self.countdown_end_time: Optional[float] = None

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(32)

        self.env: Optional[MultiSnakeEnv] = None

        self.running = False
        self.phase = "lobby"
        self.last_obs = None
        self.clients: Dict[int, Dict] = {}
        self.client_actions: Dict[int, int] = {}
        self.lobby_info: Dict[int, Dict] = {}
        self.slot_assignments: List[Optional[int]] = []
        self.control_map: Dict[int, int] = {}
        self.next_player_id = 0
        self.host_id: Optional[int] = None
        self.lock = threading.Lock()

    # ------------------------------------------------------------------
    # 启动与主循环
    # ------------------------------------------------------------------
    def start(self) -> None:
        """启动监听线程和游戏主循环，响应 Ctrl+C 安全关闭。"""
        print(f"Server started on {self.host}:{self.port}")
        self.running = True

        threading.Thread(target=self.accept_clients, daemon=True).start()
        threading.Thread(target=self.game_loop, daemon=True).start()

        try:
            while self.running:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Shutting down server...")
            self.running = False

    def game_loop(self) -> None:
        """权威模拟循环：收集动作、推进环境、广播状态。"""
        while self.running:
            if self.phase == "countdown":
                self._tick_countdown()
                continue
            if self.phase != "game" or self.env is None:
                time.sleep(0.1)
                continue

            start_time = time.time()
            actions = self._gather_actions()
            obs, rewards, dones, info = self.env.step(actions)
            self.last_obs = obs

            self._broadcast_state(info)

            if info["game_over"]:
                self._finish_game(info)

            elapsed = time.time() - start_time
            time.sleep(max(0, self.tick_rate - elapsed))

    # ------------------------------------------------------------------
    # 客户端接入与消息处理
    # ------------------------------------------------------------------
    def accept_clients(self) -> None:
        """阻塞接受新连接，为每个玩家分配 ID 并启动监听线程。"""
        while self.running:
            client_sock, addr = self.server_socket.accept()
            with self.lock:
                player_id = self.next_player_id
                self.next_player_id += 1
                self.clients[player_id] = {"socket": client_sock, "addr": addr}
                self.client_actions[player_id] = 0
                if self.host_id is None:
                    self.host_id = player_id

            self._send_raw(client_sock, {
                "type": "welcome",
                "player_id": player_id,
                "max_slots": self.max_players,
                "host_id": self.host_id,
            })
            print(f"Player {player_id} connected from {addr}")

            threading.Thread(
                target=self.handle_client,
                args=(client_sock, player_id),
                daemon=True,
            ).start()

    def handle_client(self, client_socket: socket.socket, player_id: int) -> None:
        """按行读取客户端消息，解析 JSON 并交给路由函数。"""
        buffer = ""
        while self.running:
            try:
                data = client_socket.recv(2048).decode("utf-8")
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
                    self._route_message(player_id, msg)
            except OSError:
                break

        print(f"Player {player_id} disconnected")
        client_socket.close()
        self._handle_disconnect(player_id)

    def _route_message(self, player_id: int, msg: Dict) -> None:
        """根据消息类型分派到加入/模式/准备/动作等处理函数。"""
        msg_type = msg.get("type", "action")
        if msg_type == "join":
            self._handle_join(player_id, msg)
        elif msg_type == "mode":
            self._handle_mode(player_id, msg)
        elif msg_type == "ready":
            self._handle_ready(player_id, bool(msg.get("ready", False)))
        elif msg_type == "start_request":
            self._try_start_game(player_id)
        elif msg_type == "action":
            self._handle_action(player_id, int(msg.get("value", 0)))

    # ------------------------------------------------------------------
    # 大厅逻辑
    # ------------------------------------------------------------------
    def _handle_join(self, player_id: int, msg: Dict) -> None:
        """记录玩家昵称、模式等信息，并刷新大厅广播。"""
        name = (msg.get("name") or f"Player {player_id}").strip()[:16]
        mode = msg.get("mode", "player")
        ai_model = msg.get("ai_model")
        with self.lock:
            entry = self.lobby_info.get(player_id, {})
            entry.update({
                "name": name or f"Player {player_id}",
                "mode": mode,
                "ready": False,
                "slot": None,
                "ai_model": ai_model,
            })
            self.lobby_info[player_id] = entry
        self._broadcast_lobby()

    def _handle_mode(self, player_id: int, payload: Dict) -> None:
        """切换玩家模式 (人工/AI/观察)，同时清空其准备状态。"""
        mode = payload.get("mode", "player")
        ai_model = payload.get("ai_model")
        with self.lock:
            entry = self.lobby_info.get(player_id)
            if not entry:
                return
            entry["mode"] = mode
            entry["ready"] = False
            entry["ai_model"] = ai_model

        self._broadcast_lobby()

    def _handle_ready(self, player_id: int, ready: bool) -> None:
        """更新玩家准备状态并广播大厅。"""
        with self.lock:
            entry = self.lobby_info.get(player_id)
            if not entry:
                return
            entry["ready"] = bool(ready)
        self._broadcast_lobby()

    def _try_start_game(self, requester_id: int) -> None:
        """校验是否满足房主、人数、准备条件，若满足则创建环境并开局。"""
        with self.lock:
            if requester_id != self.host_id:
                self._send_tip(requester_id, "只有房主才能开始游戏。")
                return

            participants = [
                pid for pid, data in self.lobby_info.items()
                if data.get("mode") in {"player", "ai"}
            ]
            if len(participants) < 2:
                self._send_tip(requester_id, "至少需要两名参赛者才能开始。")
                return

            if not all(self.lobby_info[pid].get("ready") for pid in participants):
                self._send_tip(requester_id, "所有参赛者准备后才能开始。")
                return

            ordered = sorted(participants)[: self.max_players]
            num_snakes = len(ordered)
            if num_snakes < 2:
                self._send_tip(requester_id, "至少保留两名参赛者。")
                return

            self.env = MultiSnakeEnv(
                width=self.grid_size,
                height=self.grid_size,
                num_snakes=num_snakes,
                num_food=self.food_count,
            )
            self.slot_assignments = [None] * num_snakes
            self.control_map = {}
            for slot, pid in enumerate(ordered):
                self.slot_assignments[slot] = pid
                self.control_map[pid] = slot
                self.lobby_info[pid]["slot"] = slot
            self.phase = "countdown"
            self.last_obs = self.env.reset()
            self.countdown_end_time = time.time() + self.countdown_duration

        self._broadcast_lobby()
        self._notify_start()
        if self.env:
            self._broadcast_state(
                {
                    "steps": 0,
                    "alive_count": self.env.num_snakes,
                    "scores": [0] * self.env.num_snakes,
                    "game_over": False,
                }
            )
            self._broadcast_countdown(int(self.countdown_duration))

    def _notify_start(self) -> None:
        """向所有客户端发送 start 消息，告知槽位与模式。"""
        for pid, client in list(self.clients.items()):
            slot = self.control_map.get(pid)
            mode = self.lobby_info.get(pid, {}).get("mode", "observer")
            color = None
            if slot is not None and self.env:
                color = self.env.colors[slot % len(self.env.colors)]
            payload = {"type": "start", "slot": slot, "mode": mode, "color": color}
            self._send_raw(client["socket"], payload)
            if slot is not None and color:
                self._send_tip(pid, f"你将控制槽位 {slot}，颜色 {color}")

    def _broadcast_lobby(self) -> None:
        """将当前大厅玩家列表及基本信息广播给所有客户端。"""
        players = []
        with self.lock:
            for pid, data in self.lobby_info.items():
                players.append(
                    {
                        "id": pid,
                        "name": data.get("name", f"Player {pid}"),
                        "mode": data.get("mode", "player"),
                        "ready": data.get("ready", False),
                        "slot": data.get("slot"),
                    }
                )
        payload = {
            "type": "lobby",
            "players": players,
            "phase": self.phase,
            "max_slots": self.max_players,
            "host_id": self.host_id,
            "grid_size": self.grid_size,
        }
        self._broadcast(payload)

    # ------------------------------------------------------------------
    # 游戏循环辅助
    # ------------------------------------------------------------------
    def _handle_action(self, player_id: int, action: int) -> None:
        """记录玩家最新的相对动作，等待下一帧环境读取。"""
        if self.phase != "game":
            return
        if player_id not in self.control_map:
            return
        self.client_actions[player_id] = action

    def _gather_actions(self) -> List[int]:
        """按照槽位顺序收集动作，没有玩家控制的槽位默认直行。"""
        if not self.slot_assignments:
            return []

        actions: List[int] = []
        with self.lock:
            for slot in range(len(self.slot_assignments)):
                player_id = self.slot_assignments[slot] if slot < len(self.slot_assignments) else None
                if player_id is None:
                    actions.append(0)
                    continue

                action = self.client_actions.get(player_id, 0)
                self.client_actions[player_id] = 0
                actions.append(action)
        return actions

    def _broadcast_state(self, info: Dict) -> None:
        """根据环境当前状态组装 payload，推送给所有客户端。"""
        if not self.env:
            return

        payload = {
            "type": "state",
            "phase": self.phase,
            "snakes": [
                {
                    "id": i,
                    "body": snake["body"],
                    "alive": snake["alive"],
                    "color": self.env.colors[i % len(self.env.colors)],
                    "score": snake["score"],
                    "direction": snake["direction"].name,
                }
                for i, snake in enumerate(self.env.snakes)
            ],
            "food": list(self.env.food),
            "scores": info.get("scores", [s["score"] for s in self.env.snakes]),
            "steps": info.get("steps", self.env.steps),
            "alive_count": info.get("alive_count", sum(1 for s in self.env.snakes if s["alive"])),
            "width": self.env.width,
            "height": self.env.height,
        }
        self._broadcast(payload)

    def _broadcast_countdown(self, seconds: int) -> None:
        """向所有客户端广播倒计时剩余秒数。"""
        self._broadcast({"type": "countdown", "phase": "countdown", "seconds": max(0, seconds)})

    def _tick_countdown(self) -> None:
        """在 countdown 阶段周期性广播剩余时间，时间到后切换为 game phase。"""
        if self.countdown_end_time is None:
            time.sleep(0.1)
            return
        remaining = max(0.0, self.countdown_end_time - time.time())
        if remaining <= 0:
            self.phase = "game"
            self.countdown_end_time = None
            return
        seconds = max(1, math.ceil(remaining))
        self._broadcast_countdown(seconds)
        time.sleep(0.2)

    def _finish_game(self, info: Dict) -> None:
        """广播 game_over，总结得分并重置大厅状态。"""
        summary = {
            "type": "game_over",
            "scores": info.get("scores", [s["score"] for s in self.env.snakes]),
            "steps": info.get("steps", self.env.steps),
        }
        self._broadcast(summary)

        with self.lock:
            self.phase = "lobby"
            self.slot_assignments = []
            self.control_map.clear()
            for entry in self.lobby_info.values():
                entry["slot"] = None
                entry["ready"] = False
            self.env = None
            self.last_obs = None
            self.countdown_end_time = None

        self._broadcast_lobby()

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    def _broadcast(self, payload: Dict) -> None:
        """向所有在线客户端发送统一 payload，自动清理断线玩家。"""
        message = (json.dumps(payload) + "\n").encode("utf-8")
        dead_clients = []
        for pid, client in self.clients.items():
            try:
                client["socket"].sendall(message)
            except OSError:
                dead_clients.append(pid)
        for pid in dead_clients:
            self._handle_disconnect(pid)

    def _send_raw(self, sock: socket.socket, payload: Dict) -> None:
        """对单个 socket 发送 JSON 文本，异常时静默忽略。"""
        try:
            sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        except OSError:
            pass

    def _handle_disconnect(self, player_id: int) -> None:
        """处理玩家断线：清理 socket、槽位、重新选择房主。"""
        with self.lock:
            client = self.clients.pop(player_id, None)
            self.client_actions.pop(player_id, None)
            entry = self.lobby_info.pop(player_id, None)
            if player_id in self.control_map:
                slot = self.control_map.pop(player_id)
                if slot < len(self.slot_assignments):
                    self.slot_assignments[slot] = None
            if player_id == self.host_id:
                self.host_id = self._select_new_host()
        if entry:
            self._broadcast_lobby()
        if client:
            try:
                client["socket"].close()
            except OSError:
                pass

    def _send_tip(self, player_id: int, message: str) -> None:
        """给指定玩家发送提示消息，例如开局条件不足。"""
        client = self.clients.get(player_id)
        if not client:
            return
        self._send_raw(client["socket"], {"type": "tip", "message": message})

    def _select_new_host(self) -> Optional[int]:
        """当房主离开时，挑选 ID 最小的玩家作为新的房主。"""
        if not self.lobby_info:
            return None
        return sorted(self.lobby_info.keys())[0]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Snake authoritative server")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址 (默认 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5555, help="端口 (默认 5555)")
    parser.add_argument("--max-players", type=int, default=8, help="最大玩家数")
    parser.add_argument("--grid-size", type=int, default=30, help="地图网格尺寸")
    parser.add_argument("--food-count", type=int, default=None, help="食物数量 (默认自动按网格计算)")
    parser.add_argument("--tick-rate", type=float, default=0.12, help="环境推进间隔秒数 (越小越快)")
    parser.add_argument("--countdown", type=float, default=3.0, help="开局倒计时秒数")
    args = parser.parse_args()

    GameServer(
        host=args.host,
        port=args.port,
        max_players=args.max_players,
        grid_size=args.grid_size,
        num_food=args.food_count,
        tick_rate=args.tick_rate,
        countdown_duration=args.countdown,
    ).start()
