"""Grader for medium task: pattern F1 + linking."""

from typing import Dict, Any
from src.graders.base_grader import BaseGrader
from src.models import InvestigationState


class GraderMedium(BaseGrader):
    def grade(
        self,
        state: InvestigationState,
        scenario: Dict[str, Any],
    ) -> float:
        ground_truth = scenario.get("ground_truth", {})
        flagged = scenario.get("flagged_transactions", [])
        linked_truth = scenario.get("linked_accounts_truth", {})
        score = 0.0

        if not flagged:
            return 0.0001

        total_flagged = len(flagged)
        per_txn = 1.0 / max(total_flagged, 1)

        classified_map = {
            c.transaction_id: c.label
            for c in state.classifications
        }

        for txn in flagged:
            txn_id = txn.transaction_id
            predicted = classified_map.get(txn_id)
            actual = ground_truth.get(txn_id, "legitimate")
            if predicted == actual:
                score += per_txn * 0.5

        true_fraud_ids = {
            tid for tid, label in ground_truth.items()
            if label == "fraudulent"
        }
        predicted_fraud_ids = {
            tid for tid, label in classified_map.items()
            if label == "fraudulent"
        }

        tp = len(true_fraud_ids & predicted_fraud_ids)
        fp = len(predicted_fraud_ids - true_fraud_ids)
        fn = len(true_fraud_ids - predicted_fraud_ids)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        score += f1 * 0.3

        if linked_truth:
            correct_links = 0
            total_links = len(linked_truth)
            for acc_id in state.flagged_accounts:
                if acc_id in linked_truth:
                    correct_links += 1
            link_score = correct_links / max(total_links, 1)
            score += link_score * 0.2

        score = min(max(score, 0.0001), 0.9999)
        return round(score, 4)