"""
Self-Play Management for Snake AI.
Handles storage and sampling of historical best models.
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
        self.max_pool_size = max_pool_size
        self.history_models: List[Path] = []
        self._refresh_pool()

    def _refresh_pool(self):
        self.history_models = sorted(list(self.pool_dir.glob("*.pth")), key=os.path.getmtime)
        # Keep only latest N
        if len(self.history_models) > self.max_pool_size:
            for old_model in self.history_models[:-self.max_pool_size]:
                old_model.unlink()
            self.history_models = self.history_models[-self.max_pool_size:]

    def add_model(self, state_dict, name: str):
        path = self.pool_dir / f"{name}.pth"
        torch.save(state_dict, path)
        self._refresh_pool()

    def sample_model(self) -> Optional[Path]:
        if not self.history_models:
            return None
        return random.choice(self.history_models)
