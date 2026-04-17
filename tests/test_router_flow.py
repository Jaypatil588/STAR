from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pytest
from pydantic import ValidationError

from app.constants import COMPLEX_MODEL_KEY, SIMPLE_MODEL_KEY
from app.models import NormalizedSLMOutput, RouteRequest, SLMRoutingInput, SummarizerInput
from app.router_service import RouterService
from app.session_store import InMemorySessionStore


class FakeSLMClient:
    def __init__(
        self,
        outputs: Optional[Sequence[Dict[str, Any]]] = None,
        fail_on_calls: Optional[Set[int]] = None,
        events: Optional[List[str]] = None,
    ) -> None:
        self.outputs = list(outputs or [])
        self.fail_on_calls = fail_on_calls or set()
        self.calls: List[SLMRoutingInput] = []
        self.events = events

    async def route(self, routing_input: SLMRoutingInput) -> NormalizedSLMOutput:
        self.calls.append(routing_input)
        if self.events is not None:
            self.events.append("slm")
        call_number = len(self.calls)
        if call_number in self.fail_on_calls:
            raise RuntimeError("slm unavailable")
        raw = self.outputs.pop(0) if self.outputs else {}
        return NormalizedSLMOutput.from_raw(raw)


class FakeDownstreamClient:
    def __init__(
        self,
        responses: Optional[Sequence[Dict[str, Any]]] = None,
        fail_on_calls: Optional[Set[int]] = None,
        events: Optional[List[str]] = None,
    ) -> None:
        self.responses = list(responses or [{"ok": True}])
        self.fail_on_calls = fail_on_calls or set()
        self.calls: List[Dict[str, Any]] = []
        self.events = events

    async def call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append(payload)
        if self.events is not None:
            self.events.append("downstream")
        call_number = len(self.calls)
        if call_number in self.fail_on_calls:
            raise RuntimeError("downstream unavailable")
        if len(self.responses) >= call_number:
            return self.responses[call_number - 1]
        return self.responses[-1]


class FakeSummarizer:
    def __init__(self, events: Optional[List[str]] = None) -> None:
        self.calls: List[SummarizerInput] = []
        self.events = events

    async def summarize(self, payload: SummarizerInput) -> str:
        self.calls.append(payload)
        if self.events is not None:
            self.events.append("summarizer")
        return "summary:{0}:{1}".format(len(self.calls), payload.selected_model_key)


def build_test_service(
    slm_outputs: Optional[Sequence[Dict[str, Any]]] = None,
    slm_fail_on_calls: Optional[Set[int]] = None,
    downstream_fail_on_calls: Optional[Set[int]] = None,
    events: Optional[List[str]] = None,
) -> Tuple[RouterService, FakeSLMClient, FakeDownstreamClient, FakeSummarizer, InMemorySessionStore]:
    store = InMemorySessionStore()
    slm_client = FakeSLMClient(slm_outputs, fail_on_calls=slm_fail_on_calls, events=events)
    downstream = FakeDownstreamClient(fail_on_calls=downstream_fail_on_calls, events=events)
    summarizer = FakeSummarizer(events=events)
    service = RouterService(
        slm_client=slm_client,
        downstream_client=downstream,
        summarizer=summarizer,
        session_store=store,
    )
    return service, slm_client, downstream, summarizer, store


def run_route(service: RouterService, payload: Dict[str, Any]) -> Dict[str, Any]:
    request = RouteRequest(**payload)
    return asyncio.run(service.route(request))


def test_request_requires_session_id() -> None:
    with pytest.raises(ValidationError):
        RouteRequest(prompt="hello")


def test_calls_slm_on_every_turn() -> None:
    slm_outputs = [
        {"complexity": "high", "is_continuation_of_prior_context": False, "recommended_model_key": COMPLEX_MODEL_KEY},
        {"complexity": "high", "is_continuation_of_prior_context": True, "recommended_model_key": COMPLEX_MODEL_KEY},
    ]
    service, slm_client, _, _, _ = build_test_service(slm_outputs=slm_outputs)

    first = run_route(service, {"prompt": "design a secure architecture", "session_id": "abc"})
    second = run_route(service, {"prompt": "also add threat modeling", "session_id": "abc"})

    assert first["status"] == "success"
    assert second["status"] == "success"
    assert len(slm_client.calls) == 2


def test_pre_route_summary_is_used_for_next_turn() -> None:
    slm_outputs = [
        {"complexity": "high", "task_type": "planning", "recommended_model_key": COMPLEX_MODEL_KEY},
        {"complexity": "high", "task_type": "planning", "recommended_model_key": COMPLEX_MODEL_KEY},
    ]
    service, slm_client, _, _, _ = build_test_service(slm_outputs=slm_outputs)

    run_route(service, {"prompt": "start architecture plan", "session_id": "s1"})
    run_route(service, {"prompt": "extend plan with retries", "session_id": "s1"})

    assert slm_client.calls[0].prior_session_summary is None
    assert slm_client.calls[1].prior_session_summary == "summary:1:complex_default_v1"


def test_continue_prefers_complex_for_complex_continuation() -> None:
    slm_outputs = [
        {"complexity": "high", "is_continuation_of_prior_context": False, "recommended_model_key": COMPLEX_MODEL_KEY},
        {"complexity": "high", "is_continuation_of_prior_context": True, "recommended_model_key": SIMPLE_MODEL_KEY},
    ]
    service, _, _, _, _ = build_test_service(slm_outputs=slm_outputs)

    first = run_route(service, {"prompt": "build deep migration plan", "session_id": "flow-1"})
    second = run_route(service, {"prompt": "continue with rollback plan", "session_id": "flow-1"})

    assert first["response_1"]["selected_model_key"] == COMPLEX_MODEL_KEY
    assert second["response_1"]["selected_model_key"] == COMPLEX_MODEL_KEY
    assert second["response_1"]["decision"] == "continue"


def test_switches_to_simple_for_simple_non_continuation() -> None:
    slm_outputs = [
        {"complexity": "high", "is_continuation_of_prior_context": False, "recommended_model_key": COMPLEX_MODEL_KEY},
        {"complexity": "low", "is_continuation_of_prior_context": False, "recommended_model_key": SIMPLE_MODEL_KEY},
    ]
    service, _, _, _, _ = build_test_service(slm_outputs=slm_outputs)

    run_route(service, {"prompt": "map complete security control matrix", "session_id": "flow-2"})
    second = run_route(service, {"prompt": "what is tls", "session_id": "flow-2"})

    assert second["response_1"]["previous_model_key"] == COMPLEX_MODEL_KEY
    assert second["response_1"]["selected_model_key"] == SIMPLE_MODEL_KEY
    assert second["response_1"]["decision"] == "switch"


def test_slm_failure_reuses_previous_model() -> None:
    slm_outputs = [{"complexity": "low", "is_continuation_of_prior_context": False, "recommended_model_key": SIMPLE_MODEL_KEY}]
    service, _, _, _, _ = build_test_service(slm_outputs=slm_outputs, slm_fail_on_calls={2})

    first = run_route(service, {"prompt": "what is udp", "session_id": "flow-3"})
    second = run_route(service, {"prompt": "and tcp?", "session_id": "flow-3"})

    assert first["response_1"]["selected_model_key"] == SIMPLE_MODEL_KEY
    assert second["response_1"]["selected_model_key"] == SIMPLE_MODEL_KEY
    assert second["response_1"]["routing_metadata"]["slm_failed"] is True


def test_post_route_summary_persists_for_next_turn() -> None:
    slm_outputs = [
        {"complexity": "high", "recommended_model_key": COMPLEX_MODEL_KEY},
        {"complexity": "high", "recommended_model_key": COMPLEX_MODEL_KEY},
    ]
    service, _, downstream, _, _ = build_test_service(slm_outputs=slm_outputs)

    first = run_route(service, {"prompt": "draft SOC2 checklist", "session_id": "flow-4"})
    second = run_route(service, {"prompt": "add evidence requirements", "session_id": "flow-4"})

    assert first["response_2"]["session_context_state"]["post_route_summary_updated"] is True
    assert downstream.calls[1]["session_summary"] == "summary:1:complex_default_v1"
    assert (
        second["response_2"]["session_context_state"]["post_route_summary_preview"]
        == "summary:2:complex_default_v1"
    )
    assert second["response_2"]["session_context_state"]["turn_count_after"] == 2
    assert second["response_1"]["previous_model_key"] == COMPLEX_MODEL_KEY


def test_routes_before_summarization() -> None:
    events: List[str] = []
    slm_outputs = [{"complexity": "high", "recommended_model_key": COMPLEX_MODEL_KEY}]
    service, _, _, _, _ = build_test_service(slm_outputs=slm_outputs, events=events)

    response = run_route(service, {"prompt": "plan incident workflow", "session_id": "flow-5"})
    assert response["status"] == "success"
    assert events == ["slm", "downstream", "summarizer"]


def test_slm_route_only_returns_only_stage_1() -> None:
    slm_outputs = [{"complexity": "high", "recommended_model_key": COMPLEX_MODEL_KEY}]
    service, _, downstream, summarizer, _ = build_test_service(slm_outputs=slm_outputs)

    request = RouteRequest(prompt="route me", session_id="flow-6")
    response = asyncio.run(service.slm_route_only(request))

    assert response["response_1"]["stage"] == "slm_route"
    assert "response_2" not in response
    assert len(downstream.calls) == 0
    assert len(summarizer.calls) == 0
