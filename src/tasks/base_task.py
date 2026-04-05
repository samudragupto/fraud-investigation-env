"""Abstract base task interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseTask(ABC):
    @abstractmethod
    def get_id(self) -> str:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_difficulty(self) -> str:
        pass

    @abstractmethod
    def get_description(self) -> str:
        pass

    @abstractmethod
    def get_max_steps(self) -> int:
        pass

    @abstractmethod
    def generate_scenario(self) -> Dict[str, Any]:
        pass