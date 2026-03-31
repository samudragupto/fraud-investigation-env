"""Grader for easy task: classification accuracy."""

from typing import Dict, Any
from src.graders.base_grader import BaseGrader
from src.models import InvestigationState


class GraderEasy(BaseGrader):
    def grade(
        self,
        state: InvestigationState,
        scenario: Dict[str, Any],
    ) -> float:
        ground_truth = scenario.get("ground_truth", {})
        key_evidence = scenario.get("key_evidence", [])
        score = 0.0

        correct_classification = False
        for classification in state.classifications:
            txn_id = classification.transaction_id
            if txn_id in ground_truth:
                if classification.label == ground_truth[txn_id]:
                    correct_classification = True
                    break

        if correct_classification:
            score += 0.5

        evidence_sources = {
            e.source for e in state.gathered_evidence
        }
        relevant_sources = {
            "account_history", "merchant_profile",
            "geolocation", "device_fingerprint",
        }
        cited = len(evidence_sources & relevant_sources)
        if cited >= 2:
            score += 0.3
        elif cited >= 1:
            score += 0.15

        if state.current_step <= 6:
            score += 0.2
        elif state.current_step <= 8:
            score += 0.1

        return round(min(max(score, 0.0), 1.0), 4)