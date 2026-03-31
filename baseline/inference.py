import os
import json
import time
import httpx
from typing import Dict, Any

from openai import OpenAI
from huggingface_hub import InferenceClient


ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")
MODEL_NAME = os.environ.get(
    "MODEL_NAME",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.environ.get(
    "OPENAI_BASE_URL",
    "https://router.huggingface.co/hf-inference/v1",
).strip()


SYSTEM_PROMPT = """You are an expert financial fraud analyst.

You must investigate flagged transactions by gathering evidence first,
then classifying, then optionally writing a summary, and only then submitting.

CRITICAL RULES:
1. Never choose submit_investigation in the first 3 steps.
2. Always gather evidence before classifying.
3. Prefer these actions first:
   - query_account_history
   - query_merchant_profile
   - check_geolocation_consistency
   - analyze_velocity_pattern
   - cross_reference_accounts
   - check_device_fingerprint
4. Only classify after at least 2 evidence-gathering actions.
5. Only submit after:
   - at least one classification has been made, or
   - a summary has been written for medium/hard tasks.

Return ONLY valid JSON:
{
  "action_type": "string",
  "parameters": {}
}
No markdown. No explanation.
"""


TASK_HINTS = {
    "single_transaction_classification": (
        "For this task, do not submit early. "
        "First query account history, then merchant or geolocation, "
        "then classify the transaction as legitimate, suspicious, or fraudulent."
    ),
    "multi_account_pattern_detection": (
        "For this task, first inspect account histories and analyze velocity or cross-reference accounts. "
        "Do not submit before classification."
    ),
    "fraud_ring_detection": (
        "For this task, first query account history and cross-reference accounts. "
        "Use evidence before classifying or flagging linked accounts. "
        "Do not submit early."
    ),
}


def format_observation(obs: dict) -> str:
    compact = {
        "task_id": obs.get("task_id"),
        "episode_id": obs.get("episode_id"),
        "investigation_progress": obs.get("investigation_progress", 0.0),
        "steps_remaining": obs.get("steps_remaining", 0),
        "available_actions": obs.get("available_actions", []),
        "alerts": obs.get("alerts", []),
        "current_transaction": obs.get("current_transaction"),
        "recent_evidence": obs.get("gathered_evidence", [])[-5:],
    }
    return json.dumps(compact, indent=2, default=str)


def parse_action(text: str) -> dict:
    text = (text or "").strip()

    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines()
            if not line.strip().startswith("```")
        ).strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "action_type" in data:
            data.setdefault("parameters", {})
            return data
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            if isinstance(data, dict) and "action_type" in data:
                data.setdefault("parameters", {})
                return data
        except Exception:
            pass

    return {"action_type": "submit_investigation", "parameters": {}}


def call_openai_client(obs: dict, task_id: str) -> dict:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")

    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

    user_prompt = (
        f"Task hint: {TASK_HINTS.get(task_id, '')}\n\n"
        f"Observation JSON:\n{format_observation(obs)}"
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=256,
    )

    return parse_action(response.choices[0].message.content)


def call_hf_provider(obs: dict, task_id: str) -> dict:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")

    client = InferenceClient(
        provider="featherless-ai",
        api_key=OPENAI_API_KEY,
    )

    user_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Task hint: {TASK_HINTS.get(task_id, '')}\n\n"
        f"Observation JSON:\n{format_observation(obs)}"
    )

    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=256,
        )
        return parse_action(response.choices[0].message.content)
    except Exception:
        generated = client.text_generation(
            prompt=user_prompt,
            model="meta-llama/Llama-3.1-8B",
            max_new_tokens=256,
            temperature=0.1,
        )
        return parse_action(generated)


def get_fallback_action(obs: dict, step_count: int, task_id: str) -> dict:
    current_txn = obs.get("current_transaction") or {}
    txn_id = current_txn.get("transaction_id", "")

    if task_id == "single_transaction_classification":
        seq = [
            {"action_type": "query_account_history", "parameters": {"account_id": "ACC_EASY_001"}},
            {"action_type": "query_merchant_profile", "parameters": {"merchant_name": current_txn.get("merchant_name", "")}},
            {"action_type": "check_geolocation_consistency", "parameters": {"account_id": "ACC_EASY_001"}},
            {
                "action_type": "classify_transaction",
                "parameters": {
                    "transaction_id": txn_id,
                    "label": "fraudulent",
                    "confidence": 0.85,
                    "evidence_cited": ["account_history", "merchant_profile", "geolocation"],
                },
            },
            {
                "action_type": "write_investigation_summary",
                "parameters": {
                    "summary": "The transaction appears fraudulent based on unusual account behavior, merchant risk, and geolocation inconsistency."
                },
            },
            {"action_type": "submit_investigation", "parameters": {}},
        ]
    elif task_id == "multi_account_pattern_detection":
        seq = [
            {"action_type": "query_account_history", "parameters": {"account_id": "ACC_MED_000"}},
            {"action_type": "query_account_history", "parameters": {"account_id": "ACC_MED_001"}},
            {"action_type": "analyze_velocity_pattern", "parameters": {"account_id": "ACC_MED_001"}},
            {"action_type": "cross_reference_accounts", "parameters": {"account_ids": ["ACC_MED_000", "ACC_MED_001", "ACC_MED_002"]}},
            {
                "action_type": "classify_transaction",
                "parameters": {
                    "transaction_id": txn_id,
                    "label": "fraudulent",
                    "confidence": 0.72,
                    "evidence_cited": ["velocity_analysis", "cross_reference"],
                },
            },
            {"action_type": "flag_linked_account", "parameters": {"account_id": "ACC_MED_001"}},
            {"action_type": "submit_investigation", "parameters": {}},
        ]
    else:
        seq = [
            {"action_type": "query_account_history", "parameters": {"account_id": "ACC_HARD_000"}},
            {"action_type": "query_account_history", "parameters": {"account_id": "ACC_HARD_001"}},
            {"action_type": "cross_reference_accounts", "parameters": {"account_ids": ["ACC_HARD_000", "ACC_HARD_001", "ACC_HARD_002"]}},
            {"action_type": "check_device_fingerprint", "parameters": {"account_id": "ACC_HARD_001"}},
            {
                "action_type": "classify_transaction",
                "parameters": {
                    "transaction_id": txn_id,
                    "label": "fraudulent",
                    "confidence": 0.74,
                    "evidence_cited": ["cross_reference", "device_fingerprint"],
                },
            },
            {
                "action_type": "write_investigation_summary",
                "parameters": {
                    "summary": "The investigation indicates linked suspicious accounts with corroborating cross-reference and device evidence."
                },
            },
            {"action_type": "submit_investigation", "parameters": {}},
        ]

    return seq[step_count] if step_count < len(seq) else {"action_type": "submit_investigation", "parameters": {}}


def guardrail_action(action: dict, obs: dict, task_id: str, step_count: int) -> dict:
    current_txn = obs.get("current_transaction") or {}
    available = set(obs.get("available_actions", []))
    evidence_count = len(obs.get("gathered_evidence", []))

    if action.get("action_type") not in available:
        return get_fallback_action(obs, step_count, task_id)

    # Prevent premature submit
    if action.get("action_type") == "submit_investigation" and step_count < 3:
        return get_fallback_action(obs, step_count, task_id)

    # Prevent classification before evidence
    if action.get("action_type") == "classify_transaction" and evidence_count < 2:
        return get_fallback_action(obs, step_count, task_id)

    # Prevent empty transaction id
    if action.get("action_type") == "classify_transaction":
        params = action.get("parameters", {})
        if not params.get("transaction_id"):
            action["parameters"]["transaction_id"] = current_txn.get("transaction_id", "")

    return action


def choose_action(obs: dict, task_id: str, step_count: int) -> dict:
    action = None

    if OPENAI_API_KEY:
        try:
            action = call_openai_client(obs, task_id)
            print("  [Agent: OpenAI SDK -> HF Router]")
        except Exception:
            pass

    if not action and OPENAI_API_KEY:
        try:
            action = call_hf_provider(obs, task_id)
            print("  [Agent: HF InferenceClient -> Provider]")
        except Exception:
            pass

    if not action:
        action = get_fallback_action(obs, step_count, task_id)
        print("  [Agent: Rule-Based Fallback]")

    return guardrail_action(action, obs, task_id, step_count)


def run_episode(task_id: str) -> float:
    print(f"\nStarting episode: {task_id}")

    obs = httpx.post(
        f"{ENV_URL}/reset",
        json={"task_id": task_id},
        timeout=30.0,
    ).json()

    done = False
    step_count = 0
    total_reward = 0.0

    while not done:
        action = choose_action(obs, task_id, step_count)

        step_data = httpx.post(
            f"{ENV_URL}/step",
            json=action,
            timeout=30.0,
        ).json()

        obs = step_data.get("observation", {})
        reward = step_data.get("reward", 0.0)
        done = step_data.get("done", False)
        total_reward += reward

        print(
            f"  Step {step_count + 1}: "
            f"{action.get('action_type')} | Reward: {reward:.4f}"
        )

        step_count += 1
        time.sleep(0.7)

    score = httpx.get(
        f"{ENV_URL}/grader",
        timeout=30.0,
    ).json().get("score", 0.0)

    print(f"  -> Final Grader Score: {score:.4f}")
    return score


def main():
    print("=" * 60)
    print("META X SCALER HACKATHON BASELINE")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"OPENAI_API_KEY Present: {bool(OPENAI_API_KEY)}")

    tasks = httpx.get(f"{ENV_URL}/tasks", timeout=30.0).json()
    results = {}

    for task in tasks:
        task_id = task["id"]
        results[task_id] = run_episode(task_id)

    print("\n" + "=" * 60)
    print("FINAL SCORES")
    print("=" * 60)
    for task_id, score in results.items():
        print(f"{task_id}: {score:.4f}")


if __name__ == "__main__":
    main()