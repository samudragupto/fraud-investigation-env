"""Medium task: Multi-account pattern detection."""

from typing import Dict, Any
from src.tasks.base_task import BaseTask
from src.data.scenarios import generate_medium_scenario


class TaskMedium(BaseTask):
    def get_id(self) -> str:
        return "multi_account_pattern_detection"

    def get_name(self) -> str:
        return "Multi-Account Transaction Pattern Detection"

    def get_difficulty(self) -> str:
        return "medium"

    def get_description(self) -> str:
        return (
            "Identify coordinated fraud patterns across 3 accounts "
            "with 8-12 flagged transactions. Correctly classify "
            "and link related fraudulent transactions."
        )

    def get_max_steps(self) -> int:
        return 20

    def generate_scenario(self) -> Dict[str, Any]:
        return generate_medium_scenario()