"""Test Pydantic models."""

from src.models import (
    Action, Observation, TransactionDetail,
    Location, Reward, RewardBreakdown,
)


def test_action_model():
    action = Action(
        action_type="query_account_history",
        parameters={"account_id": "ACC_001"},
    )
    assert action.action_type == "query_account_history"


def test_observation_model():
    obs = Observation(
        steps_remaining=10,
        available_actions=["query_account_history"],
    )
    assert obs.steps_remaining == 10


def test_transaction_detail():
    txn = TransactionDetail(
        transaction_id="TXN_001",
        amount=5000.0,
        currency="INR",
        merchant_name="TestMerchant",
        merchant_category="electronics",
        timestamp="2026-03-15T10:00:00Z",
        location=Location(
            city="Mumbai",
            country="India",
            latitude=19.076,
            longitude=72.877,
        ),
        channel="online",
    )
    assert txn.amount == 5000.0


def test_reward_model():
    reward = Reward(
        step_reward=0.05,
        cumulative_reward=0.15,
        breakdown=RewardBreakdown(
            evidence_gathering=0.10,
        ),
    )
    assert reward.step_reward == 0.05