from __future__ import annotations

import json
from typing import Any, Dict, Optional, Protocol

import httpx

from app.models import NormalizedSLMOutput, SLMRoutingInput, SummarizerInput


class SLMClient(Protocol):
    async def route(self, routing_input: SLMRoutingInput) -> NormalizedSLMOutput:
        ...


class DownstreamClient(Protocol):
    async def call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        ...


class SummarizerService(Protocol):
    async def summarize(self, payload: SummarizerInput) -> str:
        ...


class RemoteSLMClient:
    def __init__(self, api_url: str, api_key: Optional[str] = None, timeout_ms: int = 4_000) -> None:
        self._api_url = api_url
        self._api_key = api_key
        self._timeout = timeout_ms / 1_000

    async def route(self, routing_input: SLMRoutingInput) -> NormalizedSLMOutput:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "prompt": routing_input.prompt,
            "session_id": routing_input.session_id,
            "prior_session_summary": routing_input.prior_session_summary,
            "previous_model_key": routing_input.previous_model_key,
            "context": routing_input.client_context,
            "constraints": routing_input.constraints,
            "routing_instruction": (
                "Always decide continue or switch. Prefer continuing the same model when "
                "the prompt extends prior complex work. Switch for simple requests or intent changes."
            ),
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(self._api_url, json=payload, headers=headers)
            response.raise_for_status()
            raw = response.json()
        return NormalizedSLMOutput.from_raw(raw)


class RemoteDownstreamClient:
    def __init__(self, api_url: str, api_key: Optional[str] = None, timeout_ms: int = 4_000) -> None:
        self._api_url = api_url
        self._api_key = api_key
        self._timeout = timeout_ms / 1_000

    async def call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(self._api_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()


class RollingSummarizerService:
    async def summarize(self, payload: SummarizerInput) -> str:
        previous = payload.previous_session_summary or ""
        prompt = payload.prompt.strip().replace("\n", " ")
        prompt = prompt[:220]
        downstream_excerpt = payload.downstream_response_excerpt[:200]
        complexity = payload.slm_result.complexity
        continuation = payload.slm_result.is_continuation_of_prior_context
        unresolved = "yes" if "?" in payload.prompt or "and" in payload.prompt.lower() else "no"

        parts = [
            f"active_task={payload.slm_result.task_type}",
            f"complexity={complexity}",
            f"continuation={str(continuation).lower()}",
            f"selected_model={payload.selected_model_key}",
            f"latest_prompt={prompt}",
            f"unresolved_followups={unresolved}",
            f"downstream_excerpt={downstream_excerpt}",
        ]
        if previous:
            parts.append(f"previous_summary={previous[:220]}")
        return " | ".join(parts)


def make_excerpt(value: Any, max_chars: int = 200) -> str:
    serialized = json.dumps(value, default=str)
    return serialized[:max_chars]
