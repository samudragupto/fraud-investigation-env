"""Test reward engine."""

from src.reward_engine import RewardEngine


def test_evidence_reward():
    engine = RewardEngine()
    reward = engine.compute_step_reward(
        action_type="query_account_history",
        action_result={"relevant": True},
        ground_truth={},
        key_evidence=[],
    )
    assert reward.step_reward == 0.05


def test_irrelevant_penalty():
    engine = RewardEngine()
    reward = engine.compute_step_reward(
        action_type="query_account_history",
        action_result={"relevant": False},
        ground_truth={},
        key_evidence=[],
    )
    assert reward.step_reward == -0.02


def test_correct_classification():
    engine = RewardEngine()
    reward = engine.compute_step_reward(
        action_type="classify_transaction",
        action_result={
            "transaction_id": "TXN_001",
            "label": "fraudulent",
        },
        ground_truth={"TXN_001": "fraudulent"},
        key_evidence=[],
    )
    assert reward.step_reward == 0.25


def test_false_negative_penalty():
    engine = RewardEngine()
    reward = engine.compute_step_reward(
        action_type="classify_transaction",
        action_result={
            "transaction_id": "TXN_001",
            "label": "legitimate",
        },
        ground_truth={"TXN_001": "fraudulent"},
        key_evidence=[],
    )
    assert reward.step_reward == -0.40


def test_cumulative_reward():
    engine = RewardEngine()
    engine.compute_step_reward(
        "query_account_history",
        {"relevant": True},
        {},
        [],
    )
    reward = engine.compute_step_reward(
        "query_merchant_profile",
        {"relevant": True},
        {},
        [],
    )
    assert reward.cumulative_reward == 0.10