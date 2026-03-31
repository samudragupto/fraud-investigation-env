"""Episode state tracking and management."""

import uuid
from typing import Dict, Any, Optional, List

from src.models import (
    InvestigationState,
    Observation,
    Evidence,
    Classification,
    Action,
    AccountSummary,
)


class StateManager:
    def __init__(self):
        self.state: Optional[InvestigationState] = None
        self.scenario: Optional[Dict[str, Any]] = None

    def reset(
        self,
        task_id: str,
        scenario: Dict[str, Any],
        max_steps: int,
    ) -> Observation:
        episode_id = f"EP_{uuid.uuid4().hex[:8]}"

        self.scenario = scenario
        self.state = InvestigationState(
            task_id=task_id,
            episode_id=episode_id,
            current_step=0,
            max_steps=max_steps,
            done=False,
            classifications=[],
            flagged_accounts=[],
            investigation_summary="",
            action_history=[],
            gathered_evidence=[],
            cumulative_reward=0.0,
        )

        flagged = scenario.get("flagged_transactions", [])
        first_txn = flagged[0] if flagged else None

        return Observation(
            current_transaction=first_txn,
            account_summary=None,
            gathered_evidence=[],
            investigation_progress=0.0,
            steps_remaining=max_steps,
            available_actions=[
                "query_account_history",
                "query_merchant_profile",
                "check_geolocation_consistency",
                "analyze_velocity_pattern",
                "cross_reference_accounts",
                "check_device_fingerprint",
                "classify_transaction",
                "flag_linked_account",
                "write_investigation_summary",
                "submit_investigation",
            ],
            alerts=[f"Flagged transaction: {first_txn.transaction_id}"
                    if first_txn else "No transactions"],
            task_id=task_id,
            episode_id=episode_id,
        )

    def process_action(
        self, action: Action
    ) -> Dict[str, Any]:
        if self.state is None or self.scenario is None:
            raise ValueError("Environment not initialized")

        self.state.current_step += 1
        self.state.action_history.append({
            "step": self.state.current_step,
            "action_type": action.action_type,
            "parameters": action.parameters,
        })

        result = {"relevant": False}
        accounts = self.scenario.get("accounts", {})
        merchants = self.scenario.get("merchants", {})
        key_evidence = self.scenario.get("key_evidence", [])

        if action.action_type == "query_account_history":
            acc_id = action.parameters.get("account_id", "")
            if acc_id in accounts:
                result = {"relevant": True, "account": accounts[acc_id]}
                evidence = Evidence(
                    evidence_id=f"EV_{uuid.uuid4().hex[:6]}",
                    source="account_history",
                    content=f"Retrieved history for {acc_id}",
                    relevance_score=0.7,
                )
                self.state.gathered_evidence.append(evidence)
            else:
                result = {"relevant": False}

        elif action.action_type == "query_merchant_profile":
            merchant_name = action.parameters.get(
                "merchant_name", ""
            )
            if merchant_name in merchants:
                result = {
                    "relevant": True,
                    "merchant": merchants[merchant_name],
                }
                evidence = Evidence(
                    evidence_id=f"EV_{uuid.uuid4().hex[:6]}",
                    source="merchant_profile",
                    content=(
                        f"Merchant {merchant_name}: "
                        f"risk={merchants[merchant_name]['risk_level']}, "
                        f"reports={merchants[merchant_name]['reports']}"
                    ),
                    relevance_score=0.6,
                )
                self.state.gathered_evidence.append(evidence)
            else:
                result = {"relevant": False}

        elif action.action_type == "check_geolocation_consistency":
            acc_id = action.parameters.get("account_id", "")
            if acc_id in accounts:
                acc = accounts[acc_id]
                locations = set()
                for txn in acc.recent_transactions:
                    locations.add(txn.location.country)
                inconsistent = len(locations) > 2
                result = {
                    "relevant": True,
                    "inconsistent": inconsistent,
                    "countries": list(locations),
                }
                if inconsistent and "geographic_impossibility" in key_evidence:
                    result["pattern_found"] = True
                evidence = Evidence(
                    evidence_id=f"EV_{uuid.uuid4().hex[:6]}",
                    source="geolocation",
                    content=(
                        f"Locations for {acc_id}: "
                        f"{list(locations)}, "
                        f"inconsistent={inconsistent}"
                    ),
                    relevance_score=0.8 if inconsistent else 0.3,
                )
                self.state.gathered_evidence.append(evidence)
            else:
                result = {"relevant": False}

        elif action.action_type == "analyze_velocity_pattern":
            acc_id = action.parameters.get("account_id", "")
            if acc_id in accounts:
                acc = accounts[acc_id]
                txn_count = len(acc.recent_transactions)
                high_velocity = txn_count > 20
                result = {
                    "relevant": True,
                    "pattern_found": high_velocity,
                    "transaction_count": txn_count,
                    "velocity_score": min(txn_count / 30.0, 1.0),
                }
                evidence = Evidence(
                    evidence_id=f"EV_{uuid.uuid4().hex[:6]}",
                    source="velocity_analysis",
                    content=(
                        f"Velocity for {acc_id}: "
                        f"{txn_count} txns, "
                        f"high_velocity={high_velocity}"
                    ),
                    relevance_score=0.7 if high_velocity else 0.3,
                )
                self.state.gathered_evidence.append(evidence)
            else:
                result = {"relevant": False}

        elif action.action_type == "cross_reference_accounts":
            acc_ids = action.parameters.get("account_ids", [])
            linked = []
            for acc_id in acc_ids:
                if acc_id in accounts:
                    linked.extend(
                        accounts[acc_id].linked_accounts
                    )
            linked = list(set(linked))
            pattern_found = len(linked) > 0
            result = {
                "relevant": True,
                "pattern_found": pattern_found,
                "linked_accounts": linked,
            }
            evidence = Evidence(
                evidence_id=f"EV_{uuid.uuid4().hex[:6]}",
                source="cross_reference",
                content=(
                    f"Cross-ref {acc_ids}: "
                    f"linked={linked}"
                ),
                relevance_score=0.9 if pattern_found else 0.2,
            )
            self.state.gathered_evidence.append(evidence)

        elif action.action_type == "check_device_fingerprint":
            acc_id = action.parameters.get("account_id", "")
            ground_truth = self.scenario.get("ground_truth", {})
            has_fraud = any(
                txn_id in ground_truth
                and ground_truth[txn_id] == "fraudulent"
                for txn in accounts.get(acc_id, AccountSummary(
                    account_id="", holder_name="",
                    account_age_days=0,
                    average_monthly_volume=0,
                )).recent_transactions
                for txn_id in [txn.transaction_id]
            ) if acc_id in accounts else False
            result = {
                "relevant": acc_id in accounts,
                "suspicious_device": has_fraud,
            }
            evidence = Evidence(
                evidence_id=f"EV_{uuid.uuid4().hex[:6]}",
                source="device_fingerprint",
                content=(
                    f"Device check for {acc_id}: "
                    f"suspicious={has_fraud}"
                ),
                relevance_score=0.8 if has_fraud else 0.2,
            )
            self.state.gathered_evidence.append(evidence)

        elif action.action_type == "classify_transaction":
            txn_id = action.parameters.get("transaction_id", "")
            label = action.parameters.get("label", "legitimate")
            confidence = action.parameters.get("confidence", 0.5)
            evidence_cited = action.parameters.get(
                "evidence_cited", []
            )
            classification = Classification(
                transaction_id=txn_id,
                label=label,
                confidence=confidence,
                evidence_cited=evidence_cited,
            )
            self.state.classifications.append(classification)
            ground_truth = self.scenario.get("ground_truth", {})
            actual = ground_truth.get(txn_id, "legitimate")
            result = {
                "transaction_id": txn_id,
                "label": label,
                "correct": label == actual,
            }

        elif action.action_type == "flag_linked_account":
            acc_id = action.parameters.get("account_id", "")
            self.state.flagged_accounts.append(acc_id)
            ring = self.scenario.get("fraud_ring")
            ring_members = self.scenario.get("ring_members", [])
            correct = acc_id in ring_members
            result = {
                "correct_flag": correct,
                "account_id": acc_id,
            }

        elif action.action_type == "write_investigation_summary":
            summary = action.parameters.get("summary", "")
            self.state.investigation_summary = summary
            has_classifications = len(
                self.state.classifications
            ) > 0
            has_evidence = len(self.state.gathered_evidence) > 3
            has_content = len(summary) > 50
            completeness = sum([
                0.4 if has_classifications else 0.0,
                0.3 if has_evidence else 0.0,
                0.3 if has_content else 0.0,
            ])
            result = {
                "completeness": completeness,
                "relevant": True,
            }

        elif action.action_type == "submit_investigation":
            self.state.done = True
            steps_saved = max(
                0,
                self.state.max_steps - self.state.current_step
            )
            result = {
                "submitted": True,
                "steps_saved": steps_saved,
            }

        if self.state.current_step >= self.state.max_steps:
            self.state.done = True
            result["forced_submit"] = True

        total_txns = len(
            self.scenario.get("flagged_transactions", [])
        )
        classified = len(self.state.classifications)
        progress = classified / max(total_txns, 1)
        self.state.cumulative_reward += result.get(
            "step_reward", 0
        )

        return result

    def get_observation(self) -> Observation:
        if self.state is None or self.scenario is None:
            raise ValueError("Environment not initialized")

        flagged = self.scenario.get("flagged_transactions", [])
        classified_ids = {
            c.transaction_id
            for c in self.state.classifications
        }
        remaining_txns = [
            t for t in flagged
            if t.transaction_id not in classified_ids
        ]
        current_txn = remaining_txns[0] if remaining_txns else None

        total_txns = len(flagged)
        classified = len(self.state.classifications)
        progress = classified / max(total_txns, 1)

        return Observation(
            current_transaction=current_txn,
            account_summary=None,
            gathered_evidence=self.state.gathered_evidence[-10:],
            investigation_progress=round(progress, 2),
            steps_remaining=(
                self.state.max_steps - self.state.current_step
            ),
            available_actions=[
                "query_account_history",
                "query_merchant_profile",
                "check_geolocation_consistency",
                "analyze_velocity_pattern",
                "cross_reference_accounts",
                "check_device_fingerprint",
                "classify_transaction",
                "flag_linked_account",
                "write_investigation_summary",
                "submit_investigation",
            ],
            alerts=[],
            task_id=self.state.task_id,
            episode_id=self.state.episode_id,
        )

    def get_state(self) -> InvestigationState:
        if self.state is None:
            raise ValueError("Environment not initialized")
        return self.state