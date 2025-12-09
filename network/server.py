"""多蛇云对战服务器：只负责环境演化与状态广播，策略/训练由客户端承担。"""

from __future__ import annotations

import asyncio
import json
import secrets
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from env.config import EnvConfig
from env.multi_snake_env import Direction, MultiSnakeEnv
from network.utils import direction_to_relative

EVENT_REWARD_FOOD = 20.0
EVENT_REWARD_KILL = 100.0
EVENT_REWARD_DEATH = 0.0


@dataclass
class RoomConfig:
    grid_size: int = 30
    num_snakes: int = 4
    num_food: int = 6
    max_steps: int = 1500
    tick_rate: float = 0.12

    @classmethod
    def from_request(cls, raw: dict) -> "RoomConfig":
        def _int(name: str, default: int, minimum: int = 1) -> int:
            try:
                value = int(raw.get(name, default))
            except Exception:
                value = default
            return max(minimum, value)

        grid_size = _int("grid_size", 30, minimum=4)
        requested_snakes = _int("num_snakes", 4, minimum=1)
        num_food = _int("num_food", 6, minimum=1)
        max_steps = _int("max_steps", 1500, minimum=100)
        tick_rate: float = float(raw.get("tick_rate", 0.12))
        return cls(
            grid_size=grid_size,
            num_snakes=requested_snakes,
            num_food=num_food,
            max_steps=max_steps,
            tick_rate=tick_rate,
        )

    def to_public_dict(self) -> dict:
        payload = {
            "grid_size": self.grid_size,
            "num_snakes": self.num_snakes,
            "num_food": self.num_food,
            "max_steps": self.max_steps,
            "tick_rate": self.tick_rate,
        }
        return payload



    def build_env_config(self) -> EnvConfig:
        return EnvConfig(
            width=self.grid_size,
            height=self.grid_size,
            num_snakes=self.num_snakes,
            num_food=self.num_food,
            max_steps=self.max_steps,
        )


@dataclass
class ClientSession:
    client_id: str
    name: str
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    room_id: Optional[str] = None
    pending_action: int = 0


class BaseRoom:
    """公共房间基类，负责成员管理与广播。"""

    def __init__(self, server: "CloudGameServer", owner_id: str, config: RoomConfig) -> None:
        self.server = server
        self.room_id = secrets.token_hex(4)
        self.owner_id = owner_id
        self.config = config
        self.members: List[str] = [owner_id]
        self.active = True

    def summary(self) -> dict:
        owner_name = "未知"
        owner_session = self.server.clients.get(self.owner_id)
        if owner_session:
            owner_name = owner_session.name
        return {
            "room_id": self.room_id,
            "owner": self.owner_id,
            "owner_name": owner_name,
            "members": len(self.members),
            "joinable": True,
            "config": self.config.to_public_dict(),
        }

    async def add_member(self, client_id: str) -> None:
        if client_id not in self.members:
            self.members.append(client_id)

    async def before_remove_member(self, client_id: str) -> Optional[dict]:
        return {}

    async def after_remove_member(
        self, client_id: str, *, owner_changed: bool, room_closed: bool, context: Optional[dict] = None
    ) -> None:
        return

    async def remove_member(self, client_id: str) -> None:
        context = await self.before_remove_member(client_id)
        owner_changed = False
        if client_id in self.members:
            self.members.remove(client_id)
            if client_id == self.owner_id:
                if self.members:
                    self.owner_id = self.members[0]
                    owner_changed = True
                else:
                    self.owner_id = None
        room_closed = False
        if not self.members:
            await self.close()
            room_closed = True
        elif owner_changed:
            await self.server.broadcast_rooms()
        if room_closed:
            return
        await self.after_remove_member(client_id, owner_changed=owner_changed, room_closed=room_closed, context=context)

    async def close(self) -> None:
        self.active = False
        for member in list(self.members):
            session = self.server.clients.get(member)
            if session:
                session.room_id = None
                await self.server.send(session, {"type": "room_closed", "room_id": self.room_id})
        self.members.clear()
        self.server.rooms.pop(self.room_id, None)
        await self.server.broadcast_rooms()

    async def broadcast(self, payload: dict) -> None:
        for client_id in list(self.members):
            session = self.server.clients.get(client_id)
            if session:
                await self.server.send(session, payload)



class BattleRoom(BaseRoom):
    """统一处理云对战房间。"""

    def __init__(self, server: "CloudGameServer", owner_id: str, config: RoomConfig) -> None:
        super().__init__(server, owner_id, config)
        self.grid_size = config.grid_size
        self.num_snakes = config.num_snakes
        self.tick_rate = config.tick_rate
        self.config = config
        self.env = MultiSnakeEnv(config=config.build_env_config())
        self.observations = self.env.reset()
        self.loop_task: Optional[asyncio.Task] = None
        self.slots: List[Optional[str]] = [None] * self.num_snakes
        if self.slots:
            self.slots[0] = owner_id
        self.in_progress = False
        self.countdown_until: Optional[float] = None
        self.pending_reset = False
        self.active_slots: List[int] = []
        self.score_board: List[float] = [0.0 for _ in range(self.num_snakes)]
        self.member_states: Dict[str, dict] = {
            owner_id: {"role": "human", "ready": False},
        }

    def _assign_slot(self, client_id: str) -> Optional[int]:
        if client_id in self.slots:
            return self.slots.index(client_id)
        for idx, owner in enumerate(self.slots):
            if owner is None:
                self.slots[idx] = client_id
                return idx
        return None

    async def start(self) -> None:
        if self.in_progress:
            return
        if not self.can_start():
            # 不满足开始条件直接返回，由调用方提示
            return
        self._prepare_new_game()
        loop = asyncio.get_event_loop()
        self.in_progress = True
        self.countdown_until = loop.time() + 3.0
        await self.broadcast_room_state()
        await self.broadcast({"type": "log", "message": "倒计时开始，3 秒后开局"})
        await self.broadcast({"type": "start", "room_id": self.room_id, "countdown": 3.0})
        if self.loop_task:
            return

        async def _loop() -> None:
            while self.active:
                if not self.in_progress:
                    await asyncio.sleep(0.2)
                    continue

                now = asyncio.get_event_loop().time()
                if self.countdown_until is not None and now < self.countdown_until:
                    await self._broadcast_state(countdown_remaining=self.countdown_until - now)
                    await asyncio.sleep(self.tick_rate)
                    continue
                if self.countdown_until is not None and now >= self.countdown_until:
                    self.countdown_until = None

                actions = self._gather_actions()
                self.observations, _, dones, info = self.env.step(actions)
                self._apply_events(info.get("events"))
                await self._broadcast_state(info=info)
                if info["game_over"]:
                    self.pending_reset = True
                    self.in_progress = False
                    self.countdown_until = None
                    await self.broadcast({"type": "log", "message": "对局结束，等待房主再次开始"})
                    await self._announce_mvp()
                    await self.broadcast_room_state()
                    await asyncio.sleep(0.5)
                    continue

                await asyncio.sleep(self.tick_rate)

        self.loop_task = asyncio.create_task(_loop())

    def _gather_actions(self) -> List[int]:
        actions: List[int] = []
        for slot in range(self.num_snakes):
            owner_id = self.slots[slot]
            if owner_id is None:
                actions.append(0)
                continue
            session = self.server.clients.get(owner_id)
            if session is None:
                actions.append(0)
                continue
            action = session.pending_action
            session.pending_action = 0
            actions.append(action)
        return actions

    def _prepare_new_game(self) -> None:
        self.observations = self.env.reset()
        self.pending_reset = False
        self.active_slots = []
        self.score_board = [0.0 for _ in range(self.num_snakes)]
        ready_snapshot = {
            client_id: state.get("ready", False)
            for client_id, state in self.member_states.items()
        }

        for idx, snake in enumerate(self.env.snakes):
            owner_id = self.slots[idx]
            state = self.member_states.get(owner_id) if owner_id else None
            if owner_id and state and self._role_requires_slot(state["role"]):
                ready_or_owner = ready_snapshot.get(owner_id, False) or owner_id == self.owner_id
            else:
                ready_or_owner = False

            if ready_or_owner:
                self.active_slots.append(idx)
                snake["alive"] = True
            else:
                snake["alive"] = False
                snake["body"] = []
        if len(self.active_slots) < 2:
            raise RuntimeError("Not enough active snakes to start match")
        # Hard reset member readiness (except owner) after starting so they must ready up again
        for client_id, state in self.member_states.items():
            if client_id == self.owner_id:
                continue
            if self._role_requires_slot(state["role"]):
                state["ready"] = False

    async def _broadcast_state(self, *, info: Optional[dict] = None, countdown_remaining: Optional[float] = None) -> None:
        info = info or {
            "steps": self.env.steps,
            "scores": [s["score"] for s in self.env.snakes],
            "game_over": False,
            "alive_count": sum(1 for s in self.env.snakes if s["alive"]),
        }
        snakes_payload = []
        for idx, snake in enumerate(self.env.snakes):
            owner_id = self.slots[idx] if idx < len(self.slots) else None
            owner_session = self.server.clients.get(owner_id) if owner_id else None
            owner_state = self.member_states.get(owner_id) if owner_id else None
            snakes_payload.append(
                {
                    "slot": idx,
                    "body": snake["body"],
                    "alive": snake["alive"],
                    "color": self.get_slot_color(idx, client_id=owner_id) if idx < len(self.slots) else self.env.colors[idx % len(self.env.colors)],
                    "score": snake["score"],
                    "direction": snake["direction"].name,
                    "owner_id": owner_id,
                    "owner_name": owner_session.name if owner_session else None,
                    "role": owner_state.get("role") if owner_state else None,
                }
            )

        payload = {
            "type": "state",
            "room_id": self.room_id,
            "snakes": snakes_payload,
            "food": list(self.env.food),
            "steps": info.get("steps", self.env.steps),
            "scores": list(self.score_board),
            "grid": self.grid_size,
            "alive_count": info.get("alive_count", sum(1 for s in self.env.snakes if s["alive"])),
            "game_over": info.get("game_over", False),
        }
        if countdown_remaining is not None:
            payload["countdown"] = max(0.0, countdown_remaining)
        await self.broadcast(payload)

    def _apply_events(self, events: Optional[Sequence[Dict]]) -> None:
        if not events:
            return
        for idx, event in enumerate(events):
            if idx >= len(self.score_board):
                break
            if event.get("ate_food"):
                self.score_board[idx] += EVENT_REWARD_FOOD
            kills = int(event.get("kills", 0) or 0)
            if kills:
                self.score_board[idx] += EVENT_REWARD_KILL * kills
            if event.get("died"):
                self.score_board[idx] += EVENT_REWARD_DEATH

    def _forfeit_slot(self, slot: int) -> bool:
        if slot < 0 or slot >= len(self.env.snakes):
            return False
        snake = self.env.snakes[slot]
        if not snake.get("alive", False):
            return False
        snake["alive"] = False
        snake["body"] = []
        snake["steps_alive"] = snake.get("steps_alive", 0)
        if slot in self.active_slots:
            self.active_slots.remove(slot)
        if slot < len(self.score_board):
            self.score_board[slot] += EVENT_REWARD_DEATH
        return True

    async def _announce_mvp(self) -> None:
        if not self.score_board:
            return
        best_idx = max(range(len(self.score_board)), key=lambda i: self.score_board[i])
        best_score = self.score_board[best_idx]
        owner_id = self.slots[best_idx]
        owner_session = self.server.clients.get(owner_id) if owner_id else None
        owner_name = owner_session.name if owner_session else f"槽位 {best_idx}"
        await self.broadcast(
            {
                "type": "log",
                "message": f"本局 MVP: {owner_name} 得分 {best_score:.1f}",
            }
        )

    def get_slot_color(self, slot: int, *, client_id: Optional[str] = None) -> str:
        if client_id is not None and client_id == self.owner_id:
            return "gold"
        palette = self.env.colors or ["green"]
        return palette[slot % len(palette)]


    async def add_member(self, client_id: str) -> Optional[int]:
        await super().add_member(client_id)
        self.member_states.setdefault(client_id, {"role": "human", "ready": False})
        slot = self._assign_slot(client_id)
        if slot is not None:
            return slot
        # 槽位已满，转为观战
        self.member_states[client_id]["role"] = "spectator"
        return None

    async def before_remove_member(self, client_id: str) -> dict:
        departed_slots: List[int] = []
        for slot in range(self.num_snakes):
            if self.slots[slot] == client_id:
                self.slots[slot] = None
                departed_slots.append(slot)

        session = self.server.clients.get(client_id)
        display_name = session.name if session else client_id

        forfeited = False
        if self.in_progress and departed_slots:
            for slot in departed_slots:
                forfeited |= self._forfeit_slot(slot)

        self.member_states.pop(client_id, None)
        return {
            "departed_slots": departed_slots,
            "display_name": display_name,
            "forfeited": forfeited,
        }

    async def after_remove_member(
        self, client_id: str, *, owner_changed: bool, room_closed: bool, context: Optional[dict] = None
    ) -> None:
        if room_closed:
            return
        context = context or {}
        forfeited = bool(context.get("forfeited"))
        display_name = context.get("display_name", client_id)
        if forfeited:
            alive_now = sum(1 for snake in self.env.snakes if snake["alive"])
            await self.broadcast(
                {
                    "type": "log",
                    "message": f"玩家 {display_name} 退出/断线，判定阵亡",
                }
            )
            await self._broadcast_state(
                info={
                    "steps": self.env.steps,
                    "scores": [snake["score"] for snake in self.env.snakes],
                    "alive_count": alive_now,
                    "game_over": False,
                }
            )

        await self.broadcast_room_state()

    def get_player_slot(self, client_id: str) -> Optional[int]:
        for idx, owner in enumerate(self.slots):
            if owner == client_id:
                return idx
        return None

    def _role_requires_slot(self, role: str) -> bool:
        return role in {"human", "ai"}

    async def set_member_role(self, client_id: str, role: str) -> bool:
        role = role.lower()
        if role not in {"human", "ai", "spectator"}:
            return False
        state = self.member_states.get(client_id)
        if not state:
            return False
        forfeited = False
        previous_slot = self.get_player_slot(client_id)
        if self._role_requires_slot(role):
            slot = self.get_player_slot(client_id)
            if slot is None:
                slot = self._assign_slot(client_id)
            if slot is None:
                return False
        else:
            # spectator: 释放槽位
            if previous_slot is not None:
                self.slots[previous_slot] = None
                if self.in_progress:
                    forfeited = self._forfeit_slot(previous_slot)
        state["role"] = role
        state["ready"] = False
        await self.broadcast_room_state()
        if forfeited:
            await self._broadcast_state(
                info={
                    "steps": self.env.steps,
                    "scores": [snake["score"] for snake in self.env.snakes],
                    "alive_count": sum(1 for snake in self.env.snakes if snake["alive"]),
                    "game_over": False,
                }
            )
        return True

    async def set_member_ready(self, client_id: str, ready: bool) -> bool:
        state = self.member_states.get(client_id)
        if not state:
            return False
        if not self._role_requires_slot(state["role"]):
            return False
        state["ready"] = bool(ready)
        await self.broadcast_room_state()
        return True

    def can_start(self) -> bool:
        playable = [(cid, state) for cid, state in self.member_states.items() if self._role_requires_slot(state["role"])]
        if len(playable) < 2:
            return False
        others_ready = all(state["ready"] for cid, state in playable if cid != self.owner_id)
        return others_ready

    async def broadcast_room_state(self) -> None:
        payload = {
            "type": "room_state",
            "room_id": self.room_id,
            "in_progress": self.in_progress,
            "can_start": self.can_start(),
            "owner_id": self.owner_id,
            "members": [],
        }
        for client_id in self.members:
            session = self.server.clients.get(client_id)
            if not session:
                continue
            state = self.member_states.get(client_id, {"role": "spectator", "ready": False})
            slot = self.get_player_slot(client_id)
            payload["members"].append(
                {
                    "client_id": client_id,
                    "name": session.name,
                    "role": state["role"],
                    "ready": state["ready"],
                    "slot": slot,
                    "color": self.get_slot_color(slot, client_id=client_id) if slot is not None else None,
                    "is_owner": client_id == self.owner_id,
                }
            )
        await self.broadcast(payload)


class CloudGameServer:
    """云游戏服务器：只管理客户端与环境，不再托管模型。"""

    def __init__(self, host: str = "0.0.0.0", port: int = 5555) -> None:
        self.host = host
        self.port = port
        self.clients: Dict[str, ClientSession] = {}
        self.rooms: Dict[str, BaseRoom] = {}
        self.server: Optional[asyncio.AbstractServer] = None

    async def start(self) -> None:
        self.server = await asyncio.start_server(self._handle_client, self.host, self.port)
        print(f"[CloudServer] Listening on {self.host}:{self.port}")
        async with self.server:
            await self.server.serve_forever()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        client_id = secrets.token_hex(3)
        session = ClientSession(client_id=client_id, name=f"访客{client_id}", reader=reader, writer=writer)
        self.clients[client_id] = session
        await self.send(session, {"type": "welcome", "client_id": client_id})
        await self.broadcast_rooms()

        try:
            while True:
                raw = await reader.readline()
                if not raw:
                    break
                try:
                    data = json.loads(raw.decode())
                except json.JSONDecodeError:
                    continue
                try:
                    await self._route_message(session, data)
                except Exception as exc:  # noqa: BLE001
                    await self.send(session, {"type": "error", "message": f"服务器处理消息失败: {exc}"})
                    print(f"[CloudServer] route error: {exc}")
        finally:
            await self._disconnect(client_id)

    async def _route_message(self, session: ClientSession, payload: dict) -> None:
        mtype = payload.get("type")
        if mtype == "hello":
            session.name = payload.get("name", session.name)[:20]
            await self.send(session, {"type": "ack", "message": "已更新昵称"})
        elif mtype == "list_rooms":
            await self.send(session, {"type": "rooms", "rooms": [room.summary() for room in self.rooms.values()]})
        elif mtype == "create_room":
            await self._create_room(session, payload)
        elif mtype == "join_room":
            await self._join_room(session, payload.get("room_id"))
        elif mtype == "leave_room":
            await self._leave_room(session)
        elif mtype == "action":
            session.pending_action = int(payload.get("value", 0))
        elif mtype == "set_role":
            await self._set_role(session, payload.get("role"))
        elif mtype == "set_ready":
            await self._set_ready(session, bool(payload.get("ready", False)))
        elif mtype == "start_game":
            await self.send(session, {"type": "log", "message": "收到客户端 start_game 消息"})
            await self._start_game(session)
        else:
            await self.send(session, {"type": "error", "message": f"未知消息类型: {mtype}"})

    async def _create_room(self, session: ClientSession, payload: dict) -> None:
        if session.room_id:
            await self.send(session, {"type": "error", "message": "请先离开当前房间后再创建新房间"})
            return
        raw_config = payload.get("config", {}) or {}
        try:
            room_config = RoomConfig.from_request(raw_config)
        except ValueError as exc:
            await self.send(session, {"type": "error", "message": str(exc)})
            return

        room = BattleRoom(self, session.client_id, room_config)
        self.rooms[room.room_id] = room
        session.room_id = room.room_id
        slot = room.get_player_slot(session.client_id)
        await self.send(
            session,
            {
                "type": "room_joined",
                "room_id": room.room_id,
                "slot": slot,
            },
        )
        await self.broadcast_rooms()
        await room.broadcast_room_state()

    async def _join_room(self, session: ClientSession, room_id: Optional[str]) -> None:
        if session.room_id:
            await self.send(session, {"type": "error", "message": "请先离开已加入的房间"})
            return
        if not room_id or room_id not in self.rooms:
            await self.send(session, {"type": "error", "message": "房间不存在"})
            return
        room = self.rooms[room_id]
        assigned_slot = await room.add_member(session.client_id)
        session.room_id = room.room_id
        slot = room.get_player_slot(session.client_id)
        await self.send(
            session,
            {
                "type": "room_joined",
                "room_id": room.room_id,
                "slot": slot if slot is not None else assigned_slot,
            },
        )
        await self.broadcast_rooms()
        await room.broadcast_room_state()

    async def _leave_room(self, session: ClientSession) -> None:
        if not session.room_id or session.room_id not in self.rooms:
            return
        room = self.rooms[session.room_id]
        await room.remove_member(session.client_id)
        session.room_id = None
        await self.broadcast_rooms()
        if isinstance(room, BattleRoom):
            await room.broadcast_room_state()

    async def _set_role(self, session: ClientSession, role: Optional[str]) -> None:
        room = self._room_for_session(session)
        if not room:
            await self.send(session, {"type": "error", "message": "请先加入房间"})
            return
        if not role:
            await self.send(session, {"type": "error", "message": "角色不能为空"})
            return
        ok = await room.set_member_role(session.client_id, role)
        if not ok:
            await self.send(session, {"type": "error", "message": "切换角色失败，可能没有空余槽位"})

    async def _set_ready(self, session: ClientSession, ready: bool) -> None:
        room = self._room_for_session(session)
        if not room:
            await self.send(session, {"type": "error", "message": "请先加入房间"})
            return
        ok = await room.set_member_ready(session.client_id, ready)
        if not ok:
            await self.send(session, {"type": "error", "message": "当前模式无需准备或尚未分配槽位"})

    async def _start_game(self, session: ClientSession) -> None:
        room = self._room_for_session(session)
        if not room:
            await self.send(session, {"type": "error", "message": "请先加入房间"})
            return
        if room.owner_id != session.client_id:
            await self.send(session, {"type": "error", "message": "只有房主可以开始游戏"})
            return
        await self.send(session, {"type": "log", "message": "收到开始请求，校验条件中…"})
        playable = [(cid, state) for cid, state in room.member_states.items() if room._role_requires_slot(state.get("role", ""))]
        ready_non_owner = [cid for cid, state in playable if cid != room.owner_id and state.get("ready")]
        if not room.can_start():
            detail = f"可参战 {len(playable)}，非房主已准备 {len(ready_non_owner)}"
            await self.send(session, {"type": "error", "message": "仍有玩家未准备或没有可参战玩家"})
            await room.broadcast({"type": "log", "message": f"开始失败：仍有玩家未准备或人数不足 ({detail})"})
            await room.broadcast_room_state()
            return
        if room.in_progress:
            await self.send(session, {"type": "error", "message": "对局已在进行中"})
            return
        await room.broadcast({"type": "log", "message": f"房主已开始对局，准备倒计时… (可参战 {len(playable)}, 非房主已准备 {len(ready_non_owner)})"})
        await room.start()

    def _room_for_session(self, session: ClientSession) -> Optional[BattleRoom]:
        if not session.room_id or session.room_id not in self.rooms:
            return None
        room = self.rooms[session.room_id]
        if not isinstance(room, BattleRoom):
            return None
        return room

    async def _disconnect(self, client_id: str) -> None:
        session = self.clients.pop(client_id, None)
        if not session:
            return
        if session.room_id and session.room_id in self.rooms:
            room = self.rooms[session.room_id]
            await room.remove_member(client_id)
            if isinstance(room, BattleRoom):
                await room.broadcast_room_state()
        try:
            session.writer.close()
            await session.writer.wait_closed()
        except Exception:  # noqa: BLE001
            pass
        await self.broadcast_rooms()

    async def broadcast_rooms(self) -> None:
        payload = {"type": "rooms", "rooms": [room.summary() for room in self.rooms.values()]}
        for session in self.clients.values():
            await self.send(session, payload)

    async def send(self, session: ClientSession, payload: dict) -> None:
        message = json.dumps(payload, ensure_ascii=False) + "\n"
        session.writer.write(message.encode("utf-8"))
        await session.writer.drain()


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="云游戏多蛇服务器")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    server = CloudGameServer(host=args.host, port=args.port)
    await server.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("服务器已主动关闭。")
