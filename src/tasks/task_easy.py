"""Easy task: Single transaction classification."""

from typing import Dict, Any
from src.tasks.base_task import BaseTask
from src.data.scenarios import generate_easy_scenario


class TaskEasy(BaseTask):
    def get_id(self) -> str:
        return "single_transaction_classification"

    def get_name(self) -> str:
        return "Single Transaction Anomaly Classification"

    def get_difficulty(self) -> str:
        return "easy"

    def get_description(self) -> str:
        return (
            "Classify a single flagged transaction as legitimate, "
            "suspicious, or fraudulent using account history "
            "and transaction metadata."
        )

    def get_max_steps(self) -> int:
        return 10

    def generate_scenario(self) -> Dict[str, Any]:
        return generate_easy_scenario()