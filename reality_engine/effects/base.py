from abc import ABC, abstractmethod
import numpy as np
from reality_engine.core.state import AppState

class BaseEffect(ABC):
    @abstractmethod
    def reset(self) -> None:
        """Reset internal effect state when disabled."""
        pass

    @abstractmethod
    def update(self, state: AppState, dt: float) -> None:
        """Update physics, particle systems, animations based on state & gestures."""
        pass

    @abstractmethod
    def render(self, frame: np.ndarray, state: AppState) -> np.ndarray:
        """Apply the effect to the frame/layer and return composite."""
        pass