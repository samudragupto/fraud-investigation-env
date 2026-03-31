"""Hard task: Fraud ring detection."""

from typing import Dict, Any
from src.tasks.base_task import BaseTask
from src.data.scenarios import generate_hard_scenario


class TaskHard(BaseTask):
    def get_id(self) -> str:
        return "fraud_ring_detection"

    def get_name(self) -> str:
        return "Complex Fraud Ring Detection"

    def get_difficulty(self) -> str:
        return "hard"

    def get_description(self) -> str:
        return (
            "Uncover a sophisticated fraud ring across 6+ accounts. "
            "Map the ring structure, separate red herrings from "
            "actual fraud, and produce a complete investigation "
            "report."
        )

    def get_max_steps(self) -> int:
        return 35

    def generate_scenario(self) -> Dict[str, Any]:
        return generate_hard_scenario()