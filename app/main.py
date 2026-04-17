from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI

from app.clients import (
    EndpointCaller,
    InternalEndpointHTTPCaller,
    OpenAIDownstreamClient,
    RemoteDownstreamClient,
    RemoteSLMClient,
    RollingSummarizerService,
)
from app.config import get_settings
from app.models import GPT4OModelRequest, RouteModelRequest, RouteRequest
from app.router_service import RouterService
from app.session_store import InMemorySessionStore
from app.pii_service import PIIService


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
    app.state.pii_service = PIIService()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/clean")
    async def clean_prompt(request: RouteRequest) -> dict:
        request_id = request.request_id or str(uuid4())
        try:
            cleaned_prompt = app.state.pii_service.clean_text(request.prompt)
            return {
                "request_id": request_id,
                "session_id": request.session_id,
                "cleaned_prompt": cleaned_prompt,
                "status": "success",
            }
        except Exception as exc:
            return {
                "request_id": request_id,
                "session_id": request.session_id,
                "error": str(exc),
                "status": "error",
            }

    @app.post("/v1/star")
    async def star(request: RouteRequest) -> dict:
        request_id = request.request_id or str(uuid4())
        stage_1 = await router_service.slm_route_only(
            RouteRequest(
                prompt=request.prompt,
                session_id=request.session_id,
                context=request.context,
                user_id=request.user_id,
                request_id=request_id,
                constraints=request.constraints,
                debug=request.debug,
            )
        )
        response_1 = stage_1["response_1"]
        selected_model_key = str(response_1["selected_model_key"])
        model_id = _map_model_key_to_model_id(selected_model_key)

        fixed_response = {
            "stage": "star",
            "decision": "slm_selected_model",
            "modelID": model_id,
            "next_endpoint": "/v1/routeModel",
        }
        route_model_payload = {
            "request_id": request_id,
            "session_id": request.session_id,
            "modelID": model_id,
            "prompt": request.prompt,
            "context": request.context,
            "routing_metadata": response_1,
        }
        try:
            route_model_response = await app.state.endpoint_caller.post(
                "/v1/routeModel", route_model_payload
            )
            final_status = route_model_response.get("status", "success")
        except Exception as exc:
            route_model_response = {"final_response": {"error": str(exc)}, "status": "error"}
            final_status = "error"
        return {
            "request_id": request_id,
            "session_id": request.session_id,
            "input_prompt": request.prompt,
            "fixed_response": fixed_response,
            "response_1": response_1,
            "routeModel_response": route_model_response,
            "final_response": route_model_response.get("final_response"),
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


app = create_app()
