from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

class BaseAgent(ABC):
    """Abstract base class for all Reinforcement Learning agents."""

    @abstractmethod
    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Determines the action to take given the current state.

        Args:
           state: The current state observation (e.g., CHW numpy array).
           epsilon: Epsilon value for epsilon-greedy exploration (if applicable).

        Returns:
            int: The action index (0: Straight, 1: Left, 2: Right).
        """
        raise NotImplementedError

    @abstractmethod
    def train_step(self, batch: Any, weights: Optional[np.ndarray] = None) -> Tuple[float, Optional[np.ndarray]]:
        """Performs a single training step using a batch of experiences.

        Args:
            batch: A batch of transitions (states, actions, rewards, next_states, dones).
            weights: Optional importance sampling weights for the batch.

        Returns:
            Tuple[float, Optional[np.ndarray]]: A tuple containing the loss value and TD-errors (if applicable).
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """Saves the agent's model to the specified path.

        Args:
            path: filesystem path to save the model.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """Loads the agent's model from the specified path.

        Args:
            path: filesystem path to load the model from.
        """
        raise NotImplementedError
