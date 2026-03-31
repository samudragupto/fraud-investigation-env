"""FastAPI server with all required endpoints."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.environment import FraudInvestigationEnv
from src.models import (
    Action, ResetRequest, StepResult,
    GraderResult,
)

import os
import json
import subprocess

app = FastAPI(
    title="Fraud Investigation OpenEnv",
    version="1.0.0",
)

env = FraudInvestigationEnv()


@app.get("/")
def health():
    return {
        "status": "ok",
        "environment": "fraud-investigation-env",
        "model": os.environ.get(
            "MODEL_NAME",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
        ),
    }


@app.post("/reset")
def reset(request: ResetRequest):
    try:
        observation = env.reset(task_id=request.task_id)
        return observation.model_dump()
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=str(e)
        )


@app.post("/step")
def step(action: Action):
    try:
        result = env.step(action)
        return result.model_dump()
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=str(e)
        )


@app.get("/state")
def state():
    try:
        current_state = env.state()
        return current_state.model_dump()
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=str(e)
        )


@app.get("/tasks")
def tasks():
    return env.get_tasks()


@app.get("/grader")
def grader():
    try:
        result = env.grade()
        return result.model_dump()
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=str(e)
        )


@app.get("/baseline")
def baseline():
    """
    Trigger baseline inference and return scores for all tasks.
    """
    try:
        result = subprocess.run(
            ["python", "baseline/inference.py"],
            capture_output=True,
            text=True,
            timeout=600,
            env={
                **os.environ,
                "ENV_URL": "http://127.0.0.1:7860",
            },
        )

        output = (result.stdout or "") + (
            ("\n" + result.stderr) if result.stderr else ""
        )

        scores = {}
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("single_transaction_classification:"):
                try:
                    scores["single_transaction_classification"] = float(
                        line.split(":", 1)[1].strip()
                    )
                except Exception:
                    pass
            elif line.startswith("multi_account_pattern_detection:"):
                try:
                    scores["multi_account_pattern_detection"] = float(
                        line.split(":", 1)[1].strip()
                    )
                except Exception:
                    pass
            elif line.startswith("fraud_ring_detection:"):
                try:
                    scores["fraud_ring_detection"] = float(
                        line.split(":", 1)[1].strip()
                    )
                except Exception:
                    pass

        return {
            "status": "ok" if result.returncode == 0 else "completed_with_errors",
            "model_used": os.environ.get(
                "MODEL_NAME",
                "meta-llama/Meta-Llama-3.1-8B-Instruct",
            ),
            "api_key_detected": bool(os.environ.get("OPENAI_API_KEY")),
            "scores": scores,
            "logs": output[-5000:],
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "scores": {},
            "logs": "Baseline timed out after 600 seconds",
        }
    except Exception as e:
        return {
            "status": "error",
            "scores": {},
            "logs": str(e),
        }