from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import httpx

from app.slm_task_client import SLMTaskClient


class FakeResponse:
    def __init__(self, payload: Dict[str, Any]):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Dict[str, Any]:
        return self._payload


class FakeAsyncClient:
    def __init__(self, responses: List[Dict[str, Any]], calls: List[Dict[str, Any]], timeout: float):
        self._responses = responses
        self._calls = calls
        self._timeout = timeout

    async def __aenter__(self) -> "FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, url: str, json: Dict[str, Any]) -> FakeResponse:
        self._calls.append({"url": url, "json": json, "timeout": self._timeout})
        payload = self._responses.pop(0)
        return FakeResponse(payload)


def test_slm_task_client_uses_chat_completions_messages(monkeypatch):
    calls: List[Dict[str, Any]] = []
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"split": true, "prompts": [{"prompt":"Search AI news",'
                            '"tool":"web_search","complexity":"low"},{"prompt":"Summarize",'
                            '"tool":"summarization","complexity":"low"}]}'
                        )
                    }
                }
            ]
        }
    ]

    def fake_client_factory(*args, **kwargs):
        return FakeAsyncClient(responses=responses, calls=calls, timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr(httpx, "AsyncClient", fake_client_factory)

    client = SLMTaskClient(api_url="http://localhost:8001/v1/chat/completions")
    result = asyncio.run(client.analyze("Search latest AI news and summarize", "s1"))

    assert result.split is True
    assert len(result.prompts) == 2
    assert calls[0]["url"] == "http://localhost:8001/v1/chat/completions"
    body = calls[0]["json"]
    assert "messages" in body
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["role"] == "user"
    assert "prompt" not in body


def test_slm_task_client_retries_with_repair_prompt(monkeypatch):
    calls: List[Dict[str, Any]] = []
    responses = [
        {"choices": [{"message": {"content": "not json"}}]},
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"split": false, "prompts": [{"prompt":"What is 2+2?",'
                            '"tool":"math_compute","complexity":"low"}]}'
                        )
                    }
                }
            ]
        },
    ]

    def fake_client_factory(*args, **kwargs):
        return FakeAsyncClient(responses=responses, calls=calls, timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr(httpx, "AsyncClient", fake_client_factory)

    client = SLMTaskClient(api_url="http://localhost:8001/v1/chat/completions")
    result = asyncio.run(client.analyze("What is 2+2?", "s2"))

    assert result.split is False
    assert len(calls) == 2
    repair_user_message = calls[1]["json"]["messages"][1]["content"]
    assert "invalid JSON" in repair_user_message


def test_slm_task_client_force_split_retry_when_unsplit(monkeypatch):
    calls: List[Dict[str, Any]] = []
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"split": false, "prompts": [{"prompt":"write a creative story and make code game for it",'
                            '"tool":"logical_reasoning","complexity":"high"}]}'
                        )
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"split": true, "prompts": [{"prompt":"write a creative story",'
                            '"tool":"creative_writing","complexity":"high"},{"prompt":"make code game for it",'
                            '"tool":"code_generation","complexity":"high"}]}'
                        )
                    }
                }
            ]
        },
    ]

    def fake_client_factory(*args, **kwargs):
        return FakeAsyncClient(responses=responses, calls=calls, timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr(httpx, "AsyncClient", fake_client_factory)

    client = SLMTaskClient(api_url="http://localhost:8001/v1/chat/completions")
    result = asyncio.run(
        client.analyze("write a creative story and make code game for it", "s3")
    )

    assert result.split is True
    assert len(result.prompts) == 2
    assert len(calls) == 2
    assert "SPLIT_HINT" in calls[0]["json"]["messages"][1]["content"]
    assert "FORCE_SPLIT_MODE" in calls[1]["json"]["messages"][1]["content"]


def test_slm_task_client_deterministic_split_backstop(monkeypatch):
    calls: List[Dict[str, Any]] = []
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"split": false, "prompts": [{"prompt":"write a creative story and make code game for it",'
                            '"tool":"logical_reasoning","complexity":"high"}]}'
                        )
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"split": false, "prompts": [{"prompt":"write a creative story and make code game for it",'
                            '"tool":"logical_reasoning","complexity":"high"}]}'
                        )
                    }
                }
            ]
        },
    ]

    def fake_client_factory(*args, **kwargs):
        return FakeAsyncClient(responses=responses, calls=calls, timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr(httpx, "AsyncClient", fake_client_factory)

    client = SLMTaskClient(api_url="http://localhost:8001/v1/chat/completions")
    result = asyncio.run(
        client.analyze("write a creative story and make code game for it", "s4")
    )

    assert result.split is True
    assert len(result.prompts) == 2
    assert result.prompts[0].tool == "creative_writing"
    assert result.prompts[1].tool == "code_generation"


def test_slm_task_client_deterministic_split_with_or(monkeypatch):
    calls: List[Dict[str, Any]] = []
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"split": false, "prompts": [{"prompt":"write a poem or build a game",'
                            '"tool":"logical_reasoning","complexity":"high"}]}'
                        )
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"split": false, "prompts": [{"prompt":"write a poem or build a game",'
                            '"tool":"logical_reasoning","complexity":"high"}]}'
                        )
                    }
                }
            ]
        },
    ]

    def fake_client_factory(*args, **kwargs):
        return FakeAsyncClient(responses=responses, calls=calls, timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr(httpx, "AsyncClient", fake_client_factory)

    client = SLMTaskClient(api_url="http://localhost:8001/v1/chat/completions")
    result = asyncio.run(client.analyze("write a poem or build a game", "s5"))

    assert result.split is True
    assert len(result.prompts) == 2
