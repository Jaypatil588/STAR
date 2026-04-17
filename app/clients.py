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


class OpenAIDownstreamClient:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1/responses",
        timeout_ms: int = 4_000,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._timeout = timeout_ms / 1_000

    async def call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = str(payload.get("prompt", ""))
        session_summary = payload.get("session_summary") or "none"
        selected_model_key = payload.get("selected_model_key", "unknown")
        decision = payload.get("decision", "continue")

        system_text = (
            "You are the downstream response model in a task-aware router demo. "
            "Answer the user's prompt directly and clearly. "
            "Use session summary only for continuity when relevant.\n"
            "session_summary: {0}\n"
            "selected_model_key: {1}\n"
            "router_decision: {2}"
        ).format(session_summary, selected_model_key, decision)

        body = {
            "model": self._model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_text}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            ],
        }
        headers = {
            "Authorization": "Bearer {0}".format(self._api_key),
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(self._base_url, json=body, headers=headers)
            response.raise_for_status()
            raw = response.json()

        return {
            "provider": "openai",
            "model": self._model,
            "response_text": _extract_openai_text(raw),
            "response_id": raw.get("id"),
            "status": raw.get("status"),
            "usage": raw.get("usage"),
        }


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


def _extract_openai_text(raw: Dict[str, Any]) -> str:
    if isinstance(raw.get("output_text"), str) and raw["output_text"].strip():
        return raw["output_text"]

    output = raw.get("output")
    if not isinstance(output, list):
        return ""

    text_parts = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for chunk in content:
            if not isinstance(chunk, dict):
                continue
            chunk_type = chunk.get("type")
            if chunk_type in {"output_text", "text"}:
                text = chunk.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
    return "".join(text_parts).strip()
