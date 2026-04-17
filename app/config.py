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
    downstream_timeout_ms: int = Field(default=4_000)
    downstream_provider: str = Field(default="http")
    openai_api_key: Optional[str] = Field(default=None)
    openai_model: str = Field(default="gpt-4o")
    openai_base_url: str = Field(default="https://api.openai.com/v1/responses")

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
            downstream_timeout_ms=int(os.getenv("DOWNSTREAM_TIMEOUT_MS", "4000")),
            downstream_provider=os.getenv("DOWNSTREAM_PROVIDER", "http").lower(),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            openai_base_url=os.getenv(
                "OPENAI_BASE_URL", "https://api.openai.com/v1/responses"
            ),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
