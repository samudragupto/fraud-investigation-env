"""Abstract base grader interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any
from src.models import InvestigationState


class BaseGrader(ABC):
    @abstractmethod
    def grade(
        self,
        state: InvestigationState,
        scenario: Dict[str, Any],
    ) -> float:
        """Return a score between 0.0 and 1.0."""
        pass