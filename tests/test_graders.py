"""Test grading logic."""

import pytest
from src.graders.grader_easy import GraderEasy
from src.graders.grader_medium import GraderMedium
from src.graders.grader_hard import GraderHard
from src.models import InvestigationState, Classification, Evidence


def test_easy_grader_perfect():
    grader = GraderEasy()
    state = InvestigationState(
        task_id="single_transaction_classification",
        episode_id="test",
        current_step=5,
        max_steps=10,
        classifications=[
            Classification(
                transaction_id="TXN_001",
                label="fraudulent",
                confidence=0.9,
            )
        ],
        gathered_evidence=[
            Evidence(
                evidence_id="EV_1",
                source="account_history",
                content="test",
            ),
            Evidence(
                evidence_id="EV_2",
                source="geolocation",
                content="test",
            ),
        ],
    )
    scenario = {
        "ground_truth": {"TXN_001": "fraudulent"},
        "key_evidence": ["amount_anomaly"],
    }
    score = grader.grade(state, scenario)
    assert score == 1.0


def test_easy_grader_wrong():
    grader = GraderEasy()
    state = InvestigationState(
        task_id="single_transaction_classification",
        episode_id="test",
        current_step=10,
        max_steps=10,
        classifications=[
            Classification(
                transaction_id="TXN_001",
                label="legitimate",
                confidence=0.5,
            )
        ],
        gathered_evidence=[],
    )
    scenario = {
        "ground_truth": {"TXN_001": "fraudulent"},
        "key_evidence": [],
    }
    score = grader.grade(state, scenario)
    assert score < 0.5


def test_grader_returns_bounded_score():
    for GraderClass in [GraderEasy, GraderMedium, GraderHard]:
        grader = GraderClass()
        state = InvestigationState(
            task_id="test",
            episode_id="test",
        )
        scenario = {"ground_truth": {}, "key_evidence": []}
        score = grader.grade(state, scenario)
        assert 0.0 <= score <= 1.0