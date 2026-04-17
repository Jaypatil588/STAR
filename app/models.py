from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

from app.constants import COMPLEX_MODEL_KEY, SIMPLE_MODEL_KEY


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RouteRequest(BaseModel):
    prompt: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    context: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    constraints: Optional[Dict[str, Any]] = None
    debug: bool = False


class SLMRoutingInput(BaseModel):
    prompt: str
    session_id: str
    prior_session_summary: Optional[str] = None
    previous_model_key: Optional[str] = None
    client_context: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, Any]] = None


class NormalizedSLMOutput(BaseModel):
    task_type: str = "general"
    complexity: Literal["low", "medium", "high"] = "medium"
    reasoning_required: bool = False
    is_continuation_of_prior_context: bool = False
    continuation_score: float = 0.0
    output_type: str = "text"
    expected_output_length: Literal["short", "medium", "long"] = "medium"
    recommended_model_key: str = COMPLEX_MODEL_KEY
    decision: Literal["continue", "switch"] = "continue"
    route_scores: Dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.0
    explanation: Optional[str] = None

    @classmethod
    def from_raw(cls, raw: Optional[Dict[str, Any]]) -> "NormalizedSLMOutput":
        raw = raw or {}
        payload = raw.get("result", raw) if isinstance(raw, dict) else {}

        complexity = str(payload.get("complexity", "medium")).lower()
        if complexity not in {"low", "medium", "high"}:
            complexity = "medium"

        expected_len = str(payload.get("expected_output_length", "medium")).lower()
        if expected_len not in {"short", "medium", "long"}:
            expected_len = "medium"

        decision = str(payload.get("decision", "continue")).lower()
        if decision not in {"continue", "switch"}:
            decision = "continue"

        continuation = bool(payload.get("is_continuation_of_prior_context", False))
        recommended = str(payload.get("recommended_model_key", "")).strip()
        if not recommended:
            recommended = SIMPLE_MODEL_KEY if complexity == "low" and not continuation else COMPLEX_MODEL_KEY

        route_scores = payload.get("route_scores")
        if not isinstance(route_scores, dict):
            route_scores = {}
        else:
            route_scores = {
                str(k): float(v)
                for k, v in route_scores.items()
                if isinstance(v, (int, float))
            }

        return cls(
            task_type=str(payload.get("task_type", "general")),
            complexity=complexity,  # type: ignore[arg-type]
            reasoning_required=bool(payload.get("reasoning_required", False)),
            is_continuation_of_prior_context=continuation,
            continuation_score=max(
                0.0, min(1.0, float(payload.get("continuation_score", 0.0)))
            ),
            output_type=str(payload.get("output_type", "text")),
            expected_output_length=expected_len,  # type: ignore[arg-type]
            recommended_model_key=recommended,
            decision=decision,  # type: ignore[arg-type]
            route_scores=route_scores,
            confidence=max(0.0, min(1.0, float(payload.get("confidence", 0.0)))),
            explanation=(
                str(payload["explanation"]) if payload.get("explanation") is not None else None
            ),
        )


class SummarizerInput(BaseModel):
    previous_session_summary: Optional[str] = None
    prompt: str
    selected_model_key: str
    slm_result: NormalizedSLMOutput
    downstream_response_excerpt: str


class SessionState(BaseModel):
    session_id: str
    current_model_key: str
    session_summary: Optional[str] = None
    turn_count: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    last_used_at: datetime = Field(default_factory=utc_now)
    last_decision: Literal["continue", "switch"] = "continue"
