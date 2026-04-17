from __future__ import annotations

from typing import Optional

from fastapi import FastAPI

from app.clients import (
    OpenAIDownstreamClient,
    RemoteDownstreamClient,
    RemoteSLMClient,
    RollingSummarizerService,
)
from app.config import get_settings
from app.models import RouteRequest
from app.router_service import RouterService
from app.session_store import InMemorySessionStore


def create_app(router_service: Optional[RouterService] = None) -> FastAPI:
    if router_service is None:
        settings = get_settings()
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

    app = FastAPI(title="STAR Router", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/route")
    async def route(request: RouteRequest) -> dict:
        return await router_service.route(request)

    return app


app = create_app()
