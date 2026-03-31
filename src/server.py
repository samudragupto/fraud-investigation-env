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
    Triggers the baseline script via automated judging.
    Complies with hackathon requirements by utilizing OPENAI_API_KEY.
    """
    try:
        import subprocess
        import os
        
        result = subprocess.run(
            ["python", "baseline/inference.py"],
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ},
        )

        output = result.stdout + result.stderr
        scores = {}
        for line in output.splitlines():
            if ":" in line and "0." in line and "Final" not in line and "Step" not in line:
                try:
                    parts = line.split(":")
                    if len(parts) == 2:
                        scores[parts[0].strip()] = float(parts[1].strip())
                except ValueError:
                    pass

        return {
            "status": "ok",
            "model_used": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "api_key_detected": bool(os.environ.get("OPENAI_API_KEY")),
            "scores": scores,
            "logs": output[-2000:]
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "scores": {}}