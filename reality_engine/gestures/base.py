from abc import ABC, abstractmethod
from reality_engine.core.state import HandData, AppState

class BaseGestureDetector(ABC):
    @abstractmethod
    def detect(self, hand: HandData, state: AppState) -> bool:
        """Return True if the gesture is currently detected for the given hand."""
        pass
