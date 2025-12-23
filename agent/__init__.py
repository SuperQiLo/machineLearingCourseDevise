import importlib
import pkgutil
import inspect
from pathlib import Path

# Registry mapping: "algo_name" -> AgentClass
# E.g. "dqn" -> DQNAgent
AGENTS = {}

def _discover_agents():
    """
    Automatically search for Agent classes in the current package.
    Convention:
    1. File must be in agent/
    2. Class name must end with 'Agent' (e.g. DQNAgent)
    3. We register it as 'dqn' (derived from class name or filename)
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
                    # Also register full class name for flexibility
                    AGENTS[name] = obj
                    
        except ImportError as e:
            print(f"Warning: Failed to import agent module {module_info.name}: {e}")

# Run discovery on import
_discover_agents()

def get_agent(name: str, input_dim: int, model_path: str = None, **kwargs):
    """
    Factory to create agent instance.
    Args:
        name: Algorithm name (e.g. 'dqn') or Class Name (e.g. 'DQNAgent')
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
