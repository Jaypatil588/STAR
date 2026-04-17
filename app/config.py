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
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
