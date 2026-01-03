"""agent/__init__.py

【中文说明】
Agent 注册与工厂：自动发现 `agent/` 目录下以 `Agent` 结尾的类，并注册到 `AGENTS`。

- `AGENTS`：`{算法名: AgentClass}`，例如 `dqn -> DQNAgent`。
- `AGENT_ALIASES`：训练脚本里常见的别名映射（如 `per`、`dueling`）。
- `get_agent()`：统一创建入口，供 GUI/网络客户端动态加载模型。
"""

import importlib
import pkgutil
import inspect
from pathlib import Path

# Registry mapping: "algo_name" -> AgentClass
# E.g. "dqn" -> DQNAgent
AGENTS = {}

def _discover_agents():
    """自动发现并注册 Agent 类。

    约定：
    1) 文件在 `agent/` 目录下（排除 `__init__.py`）
    2) 类名以 `Agent` 结尾（例如 `DQNAgent`）
    3) 注册 key 使用类名前缀的小写（例如 `DQNAgent -> dqn`）
    """
    current_dir = Path(__file__).parent
    
    # Iterate over all .py files in agent/
    for module_info in pkgutil.iter_modules([str(current_dir)]):
        if module_info.name == "__init__": continue
        
        try:
            # Dynamically import module (e.g. agent.dqn)
            module = importlib.import_module(f".{module_info.name}", package=__name__)
            
            # Scan for classes
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Filter: Must be defined in this module (not imported) and end with 'Agent'
                if obj.__module__ == module.__name__ and name.endswith("Agent"):
                    # Register
                    # Strategy 1: Use module name as key (e.g. 'dqn')
                    # Strategy 2: Use class name prefix (e.g. 'DQN')
                    # Let's use the distinct key.
                    
                    key = name.replace("Agent", "").lower() # DQNAgent -> dqn
                    AGENTS[key] = obj
                    
        except ImportError as e:
            print(f"Warning: Failed to import agent module {module_info.name}: {e}")

# Run discovery on import
_discover_agents()

# V42.0: Add aliases for convenience (matches train_dqn_variants.py naming)
AGENT_ALIASES = {
    "dueling": "duelingdqn",
    "per": "perdqn",
}
for alias, target in AGENT_ALIASES.items():
    if target in AGENTS and alias not in AGENTS:
        AGENTS[alias] = AGENTS[target]

def get_agent(name: str, input_dim: int, model_path: str = None, **kwargs):
    """创建 agent 实例（工厂方法）。

    参数：
    - `name`：算法名（如 `dqn`/`per`/`dueling`）或类名（如 `DQNAgent`）。
    - `input_dim`：vector 维度（本项目默认 28）。
    - `model_path`：可选模型权重路径（`.pth`）。
    """
    key = name.lower().replace("agent", "") # normalize 'DQNAgent' -> 'dqn'
    
    # Try exact match first (case sensitive? no we stored keys)
    # Our keys are 'dqn' and 'DQNAgent' (from above logic)
    
    # Simple lookup
    if name in AGENTS:
        return AGENTS[name](input_dim=input_dim, model_path=model_path, **kwargs)
        
    # Normalized lookup
    if key in AGENTS:
        return AGENTS[key](input_dim=input_dim, model_path=model_path, **kwargs)
        
    raise ValueError(f"Agent '{name}' not found. Available: {list(AGENTS.keys())}")
