"""self_play.py

自博弈（Self-Play）历史模型池管理。

用途
- 训练过程中把“历史较强模型”存到池中，供后续训练采样为对手。
- 通过“历史池 + 少量随机对手(chaos)”提升对抗多样性，降低过拟合。

关键行为
- 池中只保留最近 N 个模型（默认 10，可用环境变量 `SELF_PLAY_POOL_SIZE` 覆盖）。
- `sample_model()` 带 `chaos_prob`：一定概率返回 None，表示用随机对手，增强探索多样性。
"""

import os
import random
import torch
from pathlib import Path
from typing import List, Optional

class SelfPlayManager:
    def __init__(self, pool_dir: str, max_pool_size: int = 10):
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        # 允许通过环境变量覆盖池容量（更大的池 = 更多样的对手）
        # 示例（PowerShell）：$env:SELF_PLAY_POOL_SIZE = "50"
        env_max = os.getenv("SELF_PLAY_POOL_SIZE")
        self.max_pool_size = int(env_max) if env_max else max_pool_size
        self.history_models: List[Path] = []
        self._refresh_pool()

    def _refresh_pool(self):
        """刷新池：按修改时间排序，只保留最近 max_pool_size 个，其余删除。"""
        self.history_models = sorted(list(self.pool_dir.glob("*.pth")), key=os.path.getmtime)
        # Keep only latest N
        if len(self.history_models) > self.max_pool_size:
            for old_model in self.history_models[:-self.max_pool_size]:
                old_model.unlink()
            self.history_models = self.history_models[-self.max_pool_size:]

    def add_model(self, state_dict, name: str):
        """向池中写入一个模型。

        Args:
            state_dict: 模型权重（通常是 CPU tensor 的 dict）
            name: 文件名（不带后缀），最终保存为 `<name>.pth`
        """
        path = self.pool_dir / f"{name}.pth"
        torch.save(state_dict, path)
        self._refresh_pool()

    def sample_model(self, chaos_prob: float = 0.1) -> Optional[Path]:
        """采样一个历史模型路径。

        Returns:
            - Path：选中的模型文件路径（对手将加载该模型）
            - None：表示“混沌/随机对手”（增强多样性，避免只打历史池）
        """
        if not self.history_models or random.random() < chaos_prob:
            return None
        return random.choice(self.history_models)
