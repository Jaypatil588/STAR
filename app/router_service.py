from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from app.clients import DownstreamClient, SLMClient, SummarizerService, make_excerpt
from app.constants import COMPLEX_MODEL_KEY, MODEL_KEYS, SIMPLE_MODEL_KEY
from app.models import NormalizedSLMOutput, RouteRequest, SLMRoutingInput, SessionState, SummarizerInput
from app.session_store import SessionStore


class RouterService:
    def __init__(
        self,
        slm_client: SLMClient,
        downstream_client: DownstreamClient,
        summarizer: SummarizerService,
        session_store: SessionStore,
    ) -> None:
        self._slm_client = slm_client
        self._downstream_client = downstream_client
        self._summarizer = summarizer
        self._session_store = session_store

    async def slm_route_only(self, request: RouteRequest) -> dict[str, Any]:
        request_id = request.request_id or str(uuid4())
        stage_1, _ = await self._run_slm_stage(request=request, request_id=request_id)
        return {
            "request_id": request_id,
            "session_id": request.session_id,
            "input_prompt": request.prompt,
            "response_1": stage_1,
        }

    async def route(self, request: RouteRequest) -> dict[str, Any]:
        total_start = time.perf_counter()
        request_id = request.request_id or str(uuid4())

        stage_1, stage_ctx = await self._run_slm_stage(request=request, request_id=request_id)
        stage_2 = await self._run_model_stage(
            request=request,
            request_id=request_id,
            selected_model_key=stage_1["selected_model_key"],
            decision=stage_1["decision"],
            slm_result=stage_ctx["slm_result"],
            pre_route_summary=stage_ctx["pre_route_summary"],
            previous_model_key=stage_ctx["previous_model_key"],
            turn_count_before=stage_ctx["turn_count_before"],
            session_state=stage_ctx["session_state"],
        )

        total_ms = self._duration_ms(total_start)
        overall_status = (
            "success"
            if stage_1["status"] == "success" and stage_2["status"] == "success"
            else "error"
        )
        return {
            "request_id": request_id,
            "session_id": request.session_id,
            "input_prompt": request.prompt,
            "response_1": stage_1,
            "response_2": stage_2,
            "timing_ms": {"total": total_ms},
            "status": overall_status,
        }

    async def _run_slm_stage(
        self, request: RouteRequest, request_id: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        session_state = await self._session_store.get(request.session_id)
        pre_route_summary = session_state.session_summary if session_state else None
        previous_model_key = session_state.current_model_key if session_state else None
        turn_count_before = session_state.turn_count if session_state else 0

        slm_start = time.perf_counter()
        slm_error: Optional[str] = None
        try:
            slm_result = await self._slm_client.route(
                SLMRoutingInput(
                    prompt=request.prompt,
                    session_id=request.session_id,
                    prior_session_summary=pre_route_summary,
                    previous_model_key=previous_model_key,
                    client_context=request.context,
                    constraints=request.constraints,
                )
            )
        except Exception as exc:
            slm_error = str(exc)
            slm_result = self._fallback_slm_result(previous_model_key=previous_model_key)
        slm_ms = self._duration_ms(slm_start)

        selected_model_key, decision, selection_reason = self._select_model_key(
            slm_result=slm_result,
            previous_model_key=previous_model_key,
            session_exists=session_state is not None,
            slm_failed=slm_error is not None,
        )

        stage_1 = {
            "stage": "slm_route",
            "request_id": request_id,
            "session_id": request.session_id,
            "selected_model_key": selected_model_key,
            "previous_model_key": previous_model_key,
            "decision": decision,
            "routing_metadata": {
                "selection_reason": selection_reason,
                "slm_failed": slm_error is not None,
                "slm_error": slm_error,
                "route_scores": slm_result.route_scores,
                "continuation_score": slm_result.continuation_score,
                "complexity": slm_result.complexity,
                "task_type": slm_result.task_type,
            },
            "slm_result": slm_result.model_dump(mode="json"),
            "timing_ms": {"slm": slm_ms},
            "status": "success",
        }
        stage_ctx = {
            "slm_result": slm_result,
            "pre_route_summary": pre_route_summary,
            "previous_model_key": previous_model_key,
            "turn_count_before": turn_count_before,
            "session_state": session_state,
        }
        return stage_1, stage_ctx

    async def _run_model_stage(
        self,
        request: RouteRequest,
        request_id: str,
        selected_model_key: str,
        decision: str,
        slm_result: NormalizedSLMOutput,
        pre_route_summary: Optional[str],
        previous_model_key: Optional[str],
        turn_count_before: int,
        session_state: Optional[SessionState],
    ) -> Dict[str, Any]:
        downstream_payload = {
            "request_id": request_id,
            "session_id": request.session_id,
            "prompt": request.prompt,
            "session_summary": pre_route_summary,
            "selected_model_key": selected_model_key,
            "decision": decision,
            "slm_summary": slm_result.model_dump(mode="json"),
        }

        downstream_start = time.perf_counter()
        downstream_error: Optional[str] = None
        try:
            downstream_response = await self._downstream_client.call(downstream_payload)
            status = "success"
        except Exception as exc:
            downstream_error = str(exc)
            downstream_response = {"error": downstream_error}
            status = "error"
        downstream_ms = self._duration_ms(downstream_start)

        summarize_start = time.perf_counter()
        summary_updated = False
        summary_error: Optional[str] = None
        updated_summary: Optional[str] = pre_route_summary
        try:
            updated_summary = await self._summarizer.summarize(
                SummarizerInput(
                    previous_session_summary=pre_route_summary,
                    prompt=request.prompt,
                    selected_model_key=selected_model_key,
                    slm_result=slm_result,
                    downstream_response_excerpt=make_excerpt(downstream_response),
                )
            )
            summary_updated = True
        except Exception as exc:
            summary_error = str(exc)
        summarizer_ms = self._duration_ms(summarize_start)

        now = datetime.now(timezone.utc)
        await self._session_store.upsert(
            SessionState(
                session_id=request.session_id,
                current_model_key=selected_model_key,
                session_summary=updated_summary,
                turn_count=turn_count_before + 1,
                created_at=session_state.created_at if session_state else now,
                last_used_at=now,
                last_decision=decision,
            )
        )

        return {
            "stage": "model_execution",
            "request_id": request_id,
            "session_id": request.session_id,
            "selected_model_key": selected_model_key,
            "decision": decision,
            "model_response": downstream_response,
            "model_metadata": {
                "downstream_error": downstream_error,
                "downstream_status": status,
            },
            "summary_metadata": {
                "summary_error": summary_error,
            },
            "session_context_state": {
                "pre_route_summary": pre_route_summary,
                "post_route_summary_updated": summary_updated,
                "post_route_summary_preview": updated_summary,
                "previous_model_key": previous_model_key,
                "turn_count_before": turn_count_before,
                "turn_count_after": turn_count_before + 1,
            },
            "timing_ms": {
                "downstream": downstream_ms,
                "summarizer": summarizer_ms,
            },
            "status": status,
        }

    def _select_model_key(
        self,
        slm_result: NormalizedSLMOutput,
        previous_model_key: Optional[str],
        session_exists: bool,
        slm_failed: bool,
    ) -> Tuple[str, str, str]:
        if slm_failed:
            if previous_model_key:
                return previous_model_key, "continue", "slm_failure_reuse_previous_model"
            return COMPLEX_MODEL_KEY, "switch", "slm_failure_default_complex"

        continuation_of_complex = (
            session_exists
            and slm_result.is_continuation_of_prior_context
            and slm_result.complexity in {"medium", "high"}
        )
        if continuation_of_complex:
            decision = "continue" if previous_model_key == COMPLEX_MODEL_KEY else "switch"
            return COMPLEX_MODEL_KEY, decision, "continuation_complex_preference"

        if slm_result.complexity == "low" and not slm_result.is_continuation_of_prior_context:
            decision = "continue" if previous_model_key == SIMPLE_MODEL_KEY else "switch"
            return SIMPLE_MODEL_KEY, decision, "simple_non_continuation"

        selected = self._normalize_model_key(slm_result.recommended_model_key)
        decision = "continue" if selected == previous_model_key else "switch"
        if previous_model_key is None:
            decision = "continue"
        return selected, decision, "slm_recommendation"

    def _normalize_model_key(self, model_key: Optional[str]) -> str:
        if model_key in MODEL_KEYS:
            return model_key
        if model_key and "simple" in model_key.lower():
            return SIMPLE_MODEL_KEY
        return COMPLEX_MODEL_KEY

    def _fallback_slm_result(self, previous_model_key: Optional[str]) -> NormalizedSLMOutput:
        fallback_model = previous_model_key or COMPLEX_MODEL_KEY
        fallback_decision = "continue" if previous_model_key else "switch"
        return NormalizedSLMOutput(
            task_type="fallback",
            complexity="medium",
            reasoning_required=True,
            is_continuation_of_prior_context=bool(previous_model_key),
            continuation_score=1.0 if previous_model_key else 0.0,
            output_type="text",
            expected_output_length="medium",
            recommended_model_key=fallback_model,
            decision=fallback_decision,
            route_scores={fallback_model: 1.0},
            confidence=0.0,
            explanation="SLM unavailable; used fallback routing.",
        )

    @staticmethod
    def _duration_ms(start: float) -> int:
        return int((time.perf_counter() - start) * 1000)
