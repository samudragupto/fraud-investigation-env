"""Main fraud investigation environment."""

from typing import Dict, Any, Optional

from src.models import (
    Action, Observation, InvestigationState,
    StepResult, GraderResult,
)
from src.state_manager import StateManager
from src.reward_engine import RewardEngine
from src.tasks.task_easy import TaskEasy
from src.tasks.task_medium import TaskMedium
from src.tasks.task_hard import TaskHard
from src.graders.grader_easy import GraderEasy
from src.graders.grader_medium import GraderMedium
from src.graders.grader_hard import GraderHard


class FraudInvestigationEnv:
    def __init__(self):
        self.state_manager = StateManager()
        self.reward_engine = RewardEngine()

        self.tasks = {
            "single_transaction_classification": TaskEasy(),
            "multi_account_pattern_detection": TaskMedium(),
            "fraud_ring_detection": TaskHard(),
        }

        self.graders = {
            "single_transaction_classification": GraderEasy(),
            "multi_account_pattern_detection": GraderMedium(),
            "fraud_ring_detection": GraderHard(),
        }

        self.current_task_id: Optional[str] = None

    def reset(self, task_id: str) -> Observation:
        if task_id not in self.tasks:
            raise ValueError(
                f"Unknown task: {task_id}. "
                f"Available: {list(self.tasks.keys())}"
            )

        self.current_task_id = task_id
        task = self.tasks[task_id]
        scenario = task.generate_scenario()

        self.reward_engine.reset()
        observation = self.state_manager.reset(
            task_id=task_id,
            scenario=scenario,
            max_steps=task.get_max_steps(),
        )

        return observation

    def step(self, action: Action) -> StepResult:
        if self.state_manager.state is None:
            raise ValueError(
                "Call reset() before step()"
            )

        if self.state_manager.state.done:
            raise ValueError("Episode already done")

        action_result = self.state_manager.process_action(
            action
        )

        ground_truth = self.state_manager.scenario.get(
            "ground_truth", {}
        )
        key_evidence = self.state_manager.scenario.get(
            "key_evidence", []
        )

        reward = self.reward_engine.compute_step_reward(
            action_type=action.action_type,
            action_result=action_result,
            ground_truth=ground_truth,
            key_evidence=key_evidence,
        )

        observation = self.state_manager.get_observation()
        done = self.state_manager.state.done

        return StepResult(
            observation=observation,
            reward=reward.step_reward,
            done=done,
            info={
                "action_result": action_result,
                "reward_breakdown": reward.breakdown.model_dump(),
                "cumulative_reward": reward.cumulative_reward,
            },
        )

    def state(self) -> InvestigationState:
        return self.state_manager.get_state()

    def grade(self) -> GraderResult:
        if self.current_task_id is None:
            raise ValueError("No active task")

        state = self.state_manager.get_state()
        scenario = self.state_manager.scenario
        grader = self.graders[self.current_task_id]

        score = grader.grade(state, scenario)

        return GraderResult(
            task_id=self.current_task_id,
            score=score,
            details={
                "steps_used": state.current_step,
                "classifications_made": len(
                    state.classifications
                ),
                "evidence_gathered": len(
                    state.gathered_evidence
                ),
                "accounts_flagged": len(
                    state.flagged_accounts
                ),
            },
        )

    def get_tasks(self):
        result = []
        for task_id, task in self.tasks.items():
            result.append({
                "id": task.get_id(),
                "name": task.get_name(),
                "difficulty": task.get_difficulty(),
                "description": task.get_description(),
                "max_steps": task.get_max_steps(),
                "action_schema": [
                    {
                        "action_type": "query_account_history",
                        "parameters": {"account_id": "string"},
                    },
                    {
                        "action_type": "query_merchant_profile",
                        "parameters": {
                            "merchant_name": "string"
                        },
                    },
                    {
                        "action_type":
                            "check_geolocation_consistency",
                        "parameters": {"account_id": "string"},
                    },
                    {
                        "action_type":
                            "analyze_velocity_pattern",
                        "parameters": {"account_id": "string"},
                    },
                    {
                        "action_type":
                            "cross_reference_accounts",
                        "parameters": {
                            "account_ids": ["string"]
                        },
                    },
                    {
                        "action_type":
                            "check_device_fingerprint",
                        "parameters": {"account_id": "string"},
                    },
                    {
                        "action_type": "classify_transaction",
                        "parameters": {
                            "transaction_id": "string",
                            "label": "legitimate|suspicious|fraudulent",
                            "confidence": "float",
                            "evidence_cited": ["string"],
                        },
                    },
                    {
                        "action_type": "flag_linked_account",
                        "parameters": {"account_id": "string"},
                    },
                    {
                        "action_type":
                            "write_investigation_summary",
                        "parameters": {"summary": "string"},
                    },
                    {
                        "action_type": "submit_investigation",
                        "parameters": {},
                    },
                ],
            })
        return result