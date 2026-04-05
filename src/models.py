"""All Pydantic models for the environment."""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from datetime import datetime
from pydantic import BaseModel, Field


# --- Observation Models ---

class Location(BaseModel):
    city: str
    country: str
    latitude: float
    longitude: float


class TransactionDetail(BaseModel):
    transaction_id: str
    amount: float
    currency: str
    merchant_name: str
    merchant_category: str
    timestamp: str
    location: Location
    channel: Literal["online", "pos", "atm", "wire"]
    description: str = ""
    flagged_reason: str = ""


class AccountSummary(BaseModel):
    account_id: str
    holder_name: str
    account_age_days: int
    average_monthly_volume: float
    recent_transactions: List[TransactionDetail] = []
    risk_score: float = 0.0
    linked_accounts: List[str] = []


class Evidence(BaseModel):
    evidence_id: str
    source: str
    content: str
    relevance_score: float = 0.0


class Observation(BaseModel):
    current_transaction: Optional[TransactionDetail] = None
    account_summary: Optional[AccountSummary] = None
    gathered_evidence: List[Evidence] = []
    investigation_progress: float = 0.0
    steps_remaining: int = 0
    available_actions: List[str] = []
    alerts: List[str] = []
    task_id: str = ""
    episode_id: str = ""


# --- Action Model ---

ACTION_TYPES = [
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
]


class Action(BaseModel):
    action_type: str
    parameters: Dict[str, Any] = {}


# --- Reward Model ---

class RewardBreakdown(BaseModel):
    evidence_gathering: float = 0.0
    pattern_detection: float = 0.0
    classification_accuracy: float = 0.0
    efficiency_bonus: float = 0.0
    false_negative_penalty: float = 0.0
    false_positive_penalty: float = 0.0


class Reward(BaseModel):
    step_reward: float = 0.0
    cumulative_reward: float = 0.0
    breakdown: RewardBreakdown = RewardBreakdown()


# --- State Model ---

class Classification(BaseModel):
    transaction_id: str
    label: Literal["legitimate", "suspicious", "fraudulent"]
    confidence: float = 0.0
    evidence_cited: List[str] = []


class InvestigationState(BaseModel):
    task_id: str
    episode_id: str
    current_step: int = 0
    max_steps: int = 10
    done: bool = False
    classifications: List[Classification] = []
    flagged_accounts: List[str] = []
    investigation_summary: str = ""
    action_history: List[Dict[str, Any]] = []
    gathered_evidence: List[Evidence] = []
    cumulative_reward: float = 0.0


# --- API Models ---

class ResetRequest(BaseModel):
    task_id: str


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = {}


class GraderResult(BaseModel):
    task_id: str
    score: float
    details: Dict[str, Any] = {}


class TaskInfo(BaseModel):
    id: str
    name: str
    difficulty: str
    description: str
    max_steps: int
    action_schema: List[Dict[str, Any]] = []