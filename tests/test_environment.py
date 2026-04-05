"""Test the fraud investigation environment."""

import pytest
from src.environment import FraudInvestigationEnv
from src.models import Action


def test_reset():
    env = FraudInvestigationEnv()
    obs = env.reset("single_transaction_classification")
    assert obs.current_transaction is not None
    assert obs.steps_remaining == 10
    assert len(obs.available_actions) > 0


def test_step():
    env = FraudInvestigationEnv()
    env.reset("single_transaction_classification")
    action = Action(
        action_type="query_account_history",
        parameters={"account_id": "ACC_EASY_001"},
    )
    result = env.step(action)
    assert result.reward is not None
    assert result.done is False


def test_state():
    env = FraudInvestigationEnv()
    env.reset("single_transaction_classification")
    state = env.state()
    assert state.task_id == "single_transaction_classification"
    assert state.current_step == 0


def test_grader():
    env = FraudInvestigationEnv()
    env.reset("single_transaction_classification")

    action = Action(
        action_type="submit_investigation",
        parameters={},
    )
    env.step(action)

    result = env.grade()
    assert 0.0 <= result.score <= 1.0


def test_all_tasks_exist():
    env = FraudInvestigationEnv()
    tasks = env.get_tasks()
    assert len(tasks) >= 3
    task_ids = [t["id"] for t in tasks]
    assert "single_transaction_classification" in task_ids
    assert "multi_account_pattern_detection" in task_ids
    assert "fraud_ring_detection" in task_ids


def test_full_episode():
    env = FraudInvestigationEnv()
    obs = env.reset("single_transaction_classification")

    actions = [
        Action(
            action_type="query_account_history",
            parameters={"account_id": "ACC_EASY_001"},
        ),
        Action(
            action_type="check_geolocation_consistency",
            parameters={"account_id": "ACC_EASY_001"},
        ),
        Action(
            action_type="classify_transaction",
            parameters={
                "transaction_id": (
                    obs.current_transaction.transaction_id
                ),
                "label": "fraudulent",
                "confidence": 0.9,
                "evidence_cited": [],
            },
        ),
        Action(
            action_type="write_investigation_summary",
            parameters={
                "summary": (
                    "Investigation found fraudulent activity "
                    "based on geolocation anomaly and amount."
                ),
            },
        ),
        Action(
            action_type="submit_investigation",
            parameters={},
        ),
    ]

    for action in actions:
        result = env.step(action)
        if result.done:
            break

    grader_result = env.grade()
    assert 0.0 <= grader_result.score <= 1.0
    print(f"Full episode score: {grader_result.score}")