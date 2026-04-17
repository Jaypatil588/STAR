from __future__ import annotations

import os
from functools import lru_cache

from typing import Optional

from pydantic import BaseModel, Field


class Settings(BaseModel):
    slm_api_url: str = Field(default="http://localhost:8001/route")
    slm_api_key: Optional[str] = Field(default=None)
    placeholder_api_url: str = Field(default="http://localhost:8002/respond")
    placeholder_api_key: Optional[str] = Field(default=None)
    slm_timeout_ms: int = Field(default=4_000)
    slm_task_api_url: str = Field(default="http://192.168.50.218:8001/v1/chat/completions")
    slm_task_timeout_ms: int = Field(default=15_000)
    slm_task_model: str = Field(default="Qwen3.5-0.8B")
    slm_task_temperature: float = Field(default=0.1)
    slm_task_top_p: float = Field(default=0.8)
    slm_task_max_tokens: int = Field(default=1024)
    slm_task_enable_thinking: bool = Field(default=False)
    downstream_timeout_ms: int = Field(default=4_000)
    downstream_provider: str = Field(default="http")
    openai_api_key: Optional[str] = Field(default=None)
    openai_model: str = Field(default="gpt-4o")
    openai_base_url: str = Field(default="https://api.openai.com/v1/responses")
    anthropic_api_key: Optional[str] = Field(default=None)
    internal_base_url: str = Field(default="http://127.0.0.1:8080")
    internal_timeout_ms: int = Field(default=15_000)

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            slm_api_url=os.getenv("SLM_API_URL", "http://localhost:8001/route"),
            slm_api_key=os.getenv("SLM_API_KEY"),
            placeholder_api_url=os.getenv(
                "PLACEHOLDER_API_URL", "http://localhost:8002/respond"
            ),
            placeholder_api_key=os.getenv("PLACEHOLDER_API_KEY"),
            slm_timeout_ms=int(os.getenv("SLM_TIMEOUT_MS", "4000")),
            slm_task_api_url=os.getenv(
                "SLM_TASK_API_URL", "http://192.168.50.218:8001/v1/chat/completions"
            ),
            slm_task_timeout_ms=int(os.getenv("SLM_TASK_TIMEOUT_MS", "15000")),
            slm_task_model=os.getenv("SLM_TASK_MODEL", "Qwen3.5-0.8B"),
            slm_task_temperature=float(os.getenv("SLM_TASK_TEMPERATURE", "0.1")),
            slm_task_top_p=float(os.getenv("SLM_TASK_TOP_P", "0.8")),
            slm_task_max_tokens=int(os.getenv("SLM_TASK_MAX_TOKENS", "1024")),
            slm_task_enable_thinking=os.getenv(
                "SLM_TASK_ENABLE_THINKING", "false"
            ).lower()
            in {"1", "true", "yes"},
            downstream_timeout_ms=int(os.getenv("DOWNSTREAM_TIMEOUT_MS", "4000")),
            downstream_provider=os.getenv("DOWNSTREAM_PROVIDER", "http").lower(),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            openai_base_url=os.getenv(
                "OPENAI_BASE_URL", "https://api.openai.com/v1/responses"
            ),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            internal_base_url=os.getenv("INTERNAL_BASE_URL", "http://127.0.0.1:8080"),
            internal_timeout_ms=int(os.getenv("INTERNAL_TIMEOUT_MS", "15000")),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
