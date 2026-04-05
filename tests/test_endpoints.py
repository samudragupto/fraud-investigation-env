"""Test FastAPI endpoints."""

from fastapi.testclient import TestClient
from src.server import app

client = TestClient(app)


def test_health():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_tasks():
    response = client.get("/tasks")
    assert response.status_code == 200
    tasks = response.json()
    assert len(tasks) >= 3


def test_reset():
    response = client.post(
        "/reset",
        json={
            "task_id": "single_transaction_classification"
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "current_transaction" in data
    assert "steps_remaining" in data


def test_step():
    client.post(
        "/reset",
        json={
            "task_id": "single_transaction_classification"
        },
    )
    response = client.post(
        "/step",
        json={
            "action_type": "query_account_history",
            "parameters": {"account_id": "ACC_EASY_001"},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "observation" in data
    assert "reward" in data
    assert "done" in data


def test_state():
    client.post(
        "/reset",
        json={
            "task_id": "single_transaction_classification"
        },
    )
    response = client.get("/state")
    assert response.status_code == 200


def test_grader():
    client.post(
        "/reset",
        json={
            "task_id": "single_transaction_classification"
        },
    )
    client.post(
        "/step",
        json={
            "action_type": "submit_investigation",
            "parameters": {},
        },
    )
    response = client.get("/grader")
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["score"] <= 1.0