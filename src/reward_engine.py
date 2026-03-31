"""Reward computation engine."""

from typing import Dict, Any, List
from src.models import RewardBreakdown, Reward


class RewardEngine:
    def __init__(self):
        self.cumulative_reward: float = 0.0
        self.breakdown = RewardBreakdown()

    def reset(self):
        self.cumulative_reward = 0.0
        self.breakdown = RewardBreakdown()

    def compute_step_reward(
        self,
        action_type: str,
        action_result: Dict[str, Any],
        ground_truth: Dict[str, str],
        key_evidence: List[str],
    ) -> Reward:
        step_reward = 0.0

        if action_type in [
            "query_account_history",
            "query_merchant_profile",
            "check_geolocation_consistency",
            "check_device_fingerprint",
        ]:
            if action_result.get("relevant", False):
                step_reward += 0.05
                self.breakdown.evidence_gathering += 0.05
            else:
                step_reward -= 0.02
                self.breakdown.evidence_gathering -= 0.02

        elif action_type in [
            "analyze_velocity_pattern",
            "cross_reference_accounts",
        ]:
            if action_result.get("pattern_found", False):
                step_reward += 0.15
                self.breakdown.pattern_detection += 0.15
            else:
                step_reward += 0.02
                self.breakdown.pattern_detection += 0.02

        elif action_type == "classify_transaction":
            txn_id = action_result.get("transaction_id", "")
            predicted = action_result.get("label", "")
            actual = ground_truth.get(txn_id, "legitimate")

            if predicted == actual:
                step_reward += 0.25
                self.breakdown.classification_accuracy += 0.25
            elif actual == "fraudulent" and predicted != "fraudulent":
                step_reward -= 0.40
                self.breakdown.false_negative_penalty -= 0.40
            elif actual == "legitimate" and predicted == "fraudulent":
                step_reward -= 0.15
                self.breakdown.false_positive_penalty -= 0.15
            else:
                step_reward -= 0.05
                self.breakdown.classification_accuracy -= 0.05

        elif action_type == "flag_linked_account":
            if action_result.get("correct_flag", False):
                step_reward += 0.10
                self.breakdown.pattern_detection += 0.10
            else:
                step_reward -= 0.05
                self.breakdown.false_positive_penalty -= 0.05

        elif action_type == "write_investigation_summary":
            completeness = action_result.get("completeness", 0.0)
            step_reward += 0.10 * completeness
            self.breakdown.evidence_gathering += (
                0.10 * completeness
            )

        elif action_type == "submit_investigation":
            steps_saved = action_result.get("steps_saved", 0)
            efficiency = steps_saved * 0.05
            step_reward += efficiency
            self.breakdown.efficiency_bonus += efficiency

        self.cumulative_reward += step_reward

        return Reward(
            step_reward=round(step_reward, 4),
            cumulative_reward=round(self.cumulative_reward, 4),
            breakdown=self.breakdown,
        )