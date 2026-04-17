from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import httpx

from app.models import SLMTaskAnalysis

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a strict task-analysis engine.

Your job:
1) Decide whether the user prompt should be split into multiple actionable tasks.
2) Return strict JSON only with this exact schema:
{
  "split": boolean,
  "prompts": [
    {
      "prompt": string,
      "tool": string,
      "complexity": "high" | "low"
    }
  ]
}

Hard rules:
- Return ONLY raw JSON (no markdown fences, no extra text).
- Set split=true whenever the request contains two or more actionable intents.
- Each prompt must be actionable and atomic.
- "tool" must be one of:
[math_compute, logical_reasoning, data_analysis, code_generation, code_review, code_translation, web_search, internal_rag, fact_check, summarization, translation, creative_writing, copy_editing, entity_extraction, image_generation, vision_analysis, audio_processing, dom_manipulation, terminal_execution, api_call]
- complexity must be exactly "high" or "low".

Examples:
User: "What is 2 + 2?"
Output: {"split": false, "prompts": [{"prompt":"What is 2 + 2?","tool":"math_compute","complexity":"low"}]}

User: "Write python to parse CSV and summarize insights."
Output: {"split": true, "prompts": [{"prompt":"Write Python to parse a CSV file","tool":"code_generation","complexity":"high"},{"prompt":"Summarize insights from parsed CSV data","tool":"summarization","complexity":"low"}]}

User: "Translate this paragraph to French and check grammar."
Output: {"split": true, "prompts": [{"prompt":"Translate the paragraph to French","tool":"translation","complexity":"low"},{"prompt":"Check grammar of the French translation","tool":"copy_editing","complexity":"low"}]}
"""


class SLMTaskClient:
    def __init__(
        self,
        api_url: str,
        timeout_ms: int = 15_000,
        model: str = "Qwen3.5-0.8B",
        temperature: float = 0.1,
        top_p: float = 0.8,
        max_tokens: int = 1024,
        enable_thinking: bool = False,
    ):
        self._api_url = api_url
        self._timeout = timeout_ms / 1_000.0
        self._model = model
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._enable_thinking = enable_thinking

    async def analyze(self, prompt: str, session_id: str) -> SLMTaskAnalysis:
        raw_text = ""
        last_error: Optional[Exception] = None

        for attempt in range(2):
            try:
                raw_text = await self._call_chat_completions(
                    prompt=prompt,
                    session_id=session_id,
                    repair_input=raw_text if attempt == 1 else None,
                )
                payload = self._parse_json_output(raw_text)
                return SLMTaskAnalysis.model_validate(payload)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "SLM task parse attempt %s failed: %s | raw=%r",
                    attempt + 1,
                    exc,
                    raw_text[:400],
                )

        logger.error("Failed to parse SLM task analysis after retries: %s", last_error)
        # SLM-only strict policy: still return a schema-valid object, but mark as unsplit fallback.
        return SLMTaskAnalysis(
            split=False,
            prompts=[
                {
                    "prompt": prompt,
                    "tool": "logical_reasoning",
                    "complexity": "high",
                }
            ],
        )

    async def _call_chat_completions(
        self, prompt: str, session_id: str, repair_input: Optional[str]
    ) -> str:
        user_text = prompt
        if repair_input:
            user_text = (
                "Your previous response was invalid JSON for the required schema.\n"
                "Return only corrected JSON with no extra text.\n"
                "Previous output:\n{0}\n\nOriginal prompt:\n{1}"
            ).format(repair_input, prompt)

        body: Dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            "temperature": self._temperature,
            "top_p": self._top_p,
            "max_tokens": self._max_tokens,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": self._enable_thinking}
            },
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(self._api_url, json=body)
            response.raise_for_status()
            data = response.json()
        return self._extract_content_text(data)

    @staticmethod
    def _extract_content_text(data: Dict[str, Any]) -> str:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("Missing choices in chat-completions response")
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for chunk in content:
                if isinstance(chunk, dict):
                    text = chunk.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            if parts:
                return "".join(parts)
        raise ValueError("Missing message content in chat-completions response")

    @staticmethod
    def _parse_json_output(raw_text: str) -> Dict[str, Any]:
        cleaned = raw_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return json.loads(cleaned.strip())
