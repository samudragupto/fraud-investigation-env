"""Grader for hard task: ring Jaccard + report."""

from typing import Dict, Any, Set
from src.graders.base_grader import BaseGrader
from src.models import InvestigationState


class GraderHard(BaseGrader):
    def _jaccard(self, set_a: Set, set_b: Set) -> float:
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union

    def grade(
        self,
        state: InvestigationState,
        scenario: Dict[str, Any],
    ) -> float:
        ground_truth = scenario.get("ground_truth", {})
        ring_members = set(scenario.get("ring_members", []))
        key_evidence = scenario.get("key_evidence", [])
        score = 0.0

        predicted_ring = set(state.flagged_accounts)
        ring_jaccard = self._jaccard(predicted_ring, ring_members)
        score += ring_jaccard * 0.30

        classified_map = {
            c.transaction_id: c.label
            for c in state.classifications
        }
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
        score += f1 * 0.30

        evidence_sources = [e.source for e in state.gathered_evidence]
        evidence_types = set(evidence_sources)
        relevant_evidence_count = len(
            evidence_types & {
                "account_history",
                "cross_reference",
                "geolocation",
                "velocity_analysis",
                "device_fingerprint",
                "merchant_profile",
            }
        )
        total_key = max(len(key_evidence), 1)
        evidence_quality = min(relevant_evidence_count / total_key, 1.0)
        score += evidence_quality * 0.20

        summary = state.investigation_summary
        report_checks = [
            len(summary) > 100,
            "fraud" in summary.lower() or "ring" in summary.lower(),
            any(acc in summary for acc in ring_members),
            len(state.classifications) > 0,
        ]
        report_completeness = sum(report_checks) / len(report_checks)
        score += report_completeness * 0.20

        score = min(max(score, 0.0001), 0.9999)
        return round(score, 4)