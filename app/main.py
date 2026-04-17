from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from app.clients import (
    EndpointCaller,
    InternalEndpointHTTPCaller,
    OpenAIDownstreamClient,
    RemoteDownstreamClient,
    RemoteSLMClient,
    RollingSummarizerService,
)
from app.config import get_settings
from app.router_service import RouterService
from app.session_store import InMemorySessionStore
from app.slm_task_client import SLMTaskClient
from app.agent_dispatch import callAgents
from app.ui import get_ui_html
from app.models import (
    GPT4OModelRequest,
    PILCleanRequest,
    RouteModelRequest,
    RouteRequest,
    CleanRequest,
)


def create_app(router_service: Optional[RouterService] = None) -> FastAPI:
    settings = get_settings()
    if router_service is None:
        if settings.downstream_provider == "openai":
            api_key = settings.openai_api_key or settings.placeholder_api_key
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY (or PLACEHOLDER_API_KEY) must be set when DOWNSTREAM_PROVIDER=openai"
                )
            downstream_client = OpenAIDownstreamClient(
                api_key=api_key,
                model=settings.openai_model,
                base_url=settings.openai_base_url,
                timeout_ms=settings.downstream_timeout_ms,
            )
        else:
            downstream_client = RemoteDownstreamClient(
                api_url=settings.placeholder_api_url,
                api_key=settings.placeholder_api_key,
                timeout_ms=settings.downstream_timeout_ms,
            )
        router_service = RouterService(
            slm_client=RemoteSLMClient(
                api_url=settings.slm_api_url,
                api_key=settings.slm_api_key,
                timeout_ms=settings.slm_timeout_ms,
            ),
            downstream_client=downstream_client,
            summarizer=RollingSummarizerService(),
            session_store=InMemorySessionStore(),
        )

    gpt4o_key = settings.openai_api_key or settings.placeholder_api_key
    gpt4o_client = (
        OpenAIDownstreamClient(
            api_key=gpt4o_key,
            model="gpt-4o",
            base_url=settings.openai_base_url,
            timeout_ms=settings.downstream_timeout_ms,
        )
        if gpt4o_key
        else None
    )
    endpoint_caller: EndpointCaller = InternalEndpointHTTPCaller(
        base_url=settings.internal_base_url,
        timeout_ms=settings.internal_timeout_ms,
    )

    app = FastAPI(title="STAR Router", version="0.1.0")
    app.state.endpoint_caller = endpoint_caller
    app.state.gpt4o_client = gpt4o_client
    app.state.slm_task_client = SLMTaskClient(
        api_url=settings.slm_task_api_url,
        timeout_ms=settings.slm_task_timeout_ms,
        model=settings.slm_task_model,
        temperature=settings.slm_task_temperature,
        top_p=settings.slm_task_top_p,
        max_tokens=settings.slm_task_max_tokens,
        enable_thinking=settings.slm_task_enable_thinking,
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    async def ui() -> str:
        return get_ui_html()

    @app.post("/v1/clean")
    async def clean_prompt(request: CleanRequest) -> StreamingResponse:
        request_id = request.request_id or str(uuid4())

        async def stream() -> AsyncGenerator[str, None]:
            # Header line
            yield f"[STAR] request_id={request_id} session={request.session_id}\n"
            yield f"[STAR] Analyzing prompt...\n"

            try:
                task_analysis = await app.state.slm_task_client.analyze(
                    prompt=request.prompt,
                    session_id=request.session_id,
                )
            except Exception as exc:
                yield f"[ERROR] SLM analysis failed: {exc}\n"
                return

            task_count = len(task_analysis.prompts)
            split_label = "split" if task_analysis.split else "single task"
            yield f"[STAR] {split_label} → {task_count} task(s) dispatched concurrently\n"
            yield "\n"

            task_idx = 0
            async for result in callAgents(task_analysis):
                task_idx += 1
                tool  = result.get("tool", "unknown").replace("_", " ").title()
                model = result.get("assigned_model", "unknown")
                output = result.get("output", "")

                yield f"--- Task {task_idx}: {tool} (via {model}) ---\n"
                yield output
                yield "\n\n"

            yield f"[STAR] Done. {task_idx} task(s) completed.\n"

        return StreamingResponse(stream(), media_type="text/plain")


    @app.post("/v1/pil-clean")
    async def pil_clean(request: PILCleanRequest) -> dict:
        request_id = request.request_id or str(uuid4())
        cleaned_prompt = _pil_clean_text(request.prompt)
        fixed_response = {
            "stage": "pil_clean",
            "decision": "clean_prompt_and_forward",
            "next_endpoint": "/v1/clean",
        }
        clean_payload = {
            "request_id": request_id,
            "session_id": request.session_id,
            "prompt": cleaned_prompt,
        }
        try:
            clean_response = await app.state.endpoint_caller.post("/v1/clean", clean_payload)
            final_status = clean_response.get("status", "success")
        except Exception as exc:
            clean_response = {"status": "error", "error": str(exc)}
            final_status = "error"
        return {
            "request_id": request_id,
            "session_id": request.session_id,
            "original_prompt": request.prompt,
            "cleaned_prompt": cleaned_prompt,
            "fixed_response": fixed_response,
            "clean_response": clean_response,
            "final_response": _extract_final_response(clean_response),
            "status": final_status,
        }

    @app.post("/v1/star")
    async def star(request: RouteRequest) -> dict:
        request_id = request.request_id or str(uuid4())
        fixed_response = {
            "stage": "star",
            "decision": "forward_to_pil_clean",
            "next_endpoint": "/v1/pil-clean",
        }
        pil_payload = {
            "request_id": request_id,
            "session_id": request.session_id,
            "prompt": request.prompt,
            "context": request.context,
        }
        try:
            pil_response = await app.state.endpoint_caller.post(
                "/v1/pil-clean", pil_payload
            )
            final_status = pil_response.get("status", "success")
        except Exception as exc:
            pil_response = {"status": "error", "error": str(exc)}
            final_status = "error"
        return {
            "request_id": request_id,
            "session_id": request.session_id,
            "input_prompt": request.prompt,
            "fixed_response": fixed_response,
            "pil_clean_response": pil_response,
            "final_response": _extract_final_response(pil_response),
            "status": final_status,
        }

    @app.post("/v1/routeModel")
    async def route_model(request: RouteModelRequest) -> dict:
        request_id = request.request_id or str(uuid4())
        model_id = request.modelID.strip().lower()
        try:
            endpoint = _resolve_model_endpoint(model_id)
        except ValueError as exc:
            return {
                "request_id": request_id,
                "session_id": request.session_id,
                "fixed_response": {
                    "stage": "routeModel",
                    "decision": "route_to_model_endpoint",
                    "modelID": model_id,
                    "target_endpoint": None,
                },
                "final_response": {"error": str(exc)},
                "status": "error",
            }
        fixed_response = {
            "stage": "routeModel",
            "decision": "route_to_model_endpoint",
            "modelID": model_id,
            "target_endpoint": endpoint,
        }

        model_payload = {
            "request_id": request_id,
            "session_id": request.session_id,
            "modelID": model_id,
            "prompt": request.prompt,
            "context": request.context,
        }
        try:
            model_endpoint_response = await app.state.endpoint_caller.post(
                endpoint, model_payload
            )
            final_status = model_endpoint_response.get("status", "success")
        except Exception as exc:
            model_endpoint_response = {"final_response": {"error": str(exc)}, "status": "error"}
            final_status = "error"

        return {
            "request_id": request_id,
            "session_id": request.session_id,
            "fixed_response": fixed_response,
            "model_endpoint_response": model_endpoint_response,
            "final_response": model_endpoint_response.get("final_response"),
            "status": final_status,
        }

    @app.post("/v1/models/gpt4o")
    async def model_gpt4o(request: GPT4OModelRequest) -> dict:
        request_id = request.request_id or str(uuid4())
        if app.state.gpt4o_client is None:
            return {
                "request_id": request_id,
                "session_id": request.session_id,
                "fixed_response": {
                    "stage": "models/gpt4o",
                    "decision": "invoke_gpt4o",
                    "modelID": "gpt4o",
                },
                "final_response": {"error": "OPENAI_API_KEY is required for /v1/models/gpt4o"},
                "status": "error",
            }

        model_payload: Dict[str, Any] = {
            "request_id": request_id,
            "session_id": request.session_id,
            "prompt": request.prompt,
            "session_summary": _context_to_summary(request.context),
            "selected_model_key": request.modelID,
            "decision": "continue",
            "slm_summary": {},
        }
        try:
            model_output = await app.state.gpt4o_client.call(model_payload)
            status = "success"
        except Exception as exc:
            model_output = {"error": str(exc)}
            status = "error"
        return {
            "request_id": request_id,
            "session_id": request.session_id,
            "fixed_response": {
                "stage": "models/gpt4o",
                "decision": "invoke_gpt4o",
                "modelID": "gpt4o",
            },
            "final_response": model_output,
            "status": status,
        }

    @app.post("/v1/route")
    async def route(request: RouteRequest) -> dict:
        return await router_service.route(request)

    @app.post("/v1/slm-route")
    async def slm_route(request: RouteRequest) -> dict:
        return await router_service.slm_route_only(request)

    return app


def _map_model_key_to_model_id(selected_model_key: str) -> str:
    # Current phase maps all logical SLM routing keys to gpt4o.
    _ = selected_model_key
    return "gpt4o"


def _resolve_model_endpoint(model_id: str) -> str:
    if model_id == "gpt4o":
        return "/v1/models/gpt4o"
    raise ValueError("Unsupported modelID: {0}".format(model_id))


def _context_to_summary(context: Optional[Dict[str, Any]]) -> str:
    if not context:
        return "none"
    parts = ["{0}={1}".format(k, context[k]) for k in sorted(context.keys())]
    return "; ".join(parts)


def _pil_clean_text(prompt: str) -> str:
    cleaned = " ".join(prompt.strip().split())
    return cleaned


def _extract_final_response(payload: Optional[Dict[str, Any]]) -> Any:
    if not isinstance(payload, dict):
        return None
    if "final_response" in payload and payload["final_response"] is not None:
        return payload["final_response"]
    if "output" in payload:
        return {"output": payload.get("output")}
    clean_response = payload.get("clean_response")
    if isinstance(clean_response, dict):
        if (
            "final_response" in clean_response
            and clean_response["final_response"] is not None
        ):
            return clean_response["final_response"]
        if "output" in clean_response:
            return {"output": clean_response.get("output")}
    return None


app = create_app()
