from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

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
        split_candidates = self._extract_split_candidates(prompt)
        should_force_split = len(split_candidates) >= 2
        force_mode_triggered = False

        for attempt in range(2):
            try:
                raw_text = await self._call_chat_completions(
                    prompt=prompt,
                    session_id=session_id,
                    repair_input=raw_text if (attempt == 1 and last_error is not None) else None,
                    force_split_candidates=split_candidates if should_force_split else None,
                    force_mode=force_mode_triggered,
                )
                payload = self._parse_json_output(raw_text)
                parsed = SLMTaskAnalysis.model_validate(payload)
                if should_force_split and (
                    (not parsed.split) or len(parsed.prompts) < 2
                ):
                    if not force_mode_triggered:
                        force_mode_triggered = True
                        continue
                    # Hard backstop for now: if conjunction exists and model still refuses,
                    # split deterministically so downstream can execute parallel tasks.
                    return self._deterministic_split_result(
                        prompt=prompt, split_candidates=split_candidates
                    )
                return parsed
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "SLM task parse attempt %s failed: %s | raw=%r",
                    attempt + 1,
                    exc,
                    raw_text[:400],
                )

        logger.error("Failed to parse SLM task analysis after retries: %s", last_error)
        if should_force_split:
            return self._deterministic_split_result(
                prompt=prompt, split_candidates=split_candidates
            )
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
        self,
        prompt: str,
        session_id: str,
        repair_input: Optional[str],
        force_split_candidates: Optional[List[str]],
        force_mode: bool,
    ) -> str:
        user_text = prompt
        if repair_input:
            user_text = (
                "Your previous response was invalid JSON for the required schema.\n"
                "Return only corrected JSON with no extra text.\n"
                "Previous output:\n{0}\n\nOriginal prompt:\n{1}"
            ).format(repair_input, prompt)
        elif force_split_candidates and not force_mode:
            intents = "\n".join(
                ["- {0}".format(item) for item in force_split_candidates]
            )
            user_text = (
                "SPLIT_HINT: the prompt appears to contain multiple actionable intents.\n"
                "Prefer split=true when intents are distinct.\n"
                "Detected intent fragments:\n{0}\n\nOriginal prompt:\n{1}"
            ).format(intents, prompt)
        elif force_mode and force_split_candidates:
            intents = "\n".join(
                ["- {0}".format(item) for item in force_split_candidates]
            )
            user_text = (
                "FORCE_SPLIT_MODE: this prompt contains multiple intents.\n"
                "You must return split=true and at least {0} prompts.\n"
                "Detected intent fragments:\n{1}\n\nOriginal prompt:\n{2}"
            ).format(len(force_split_candidates), intents, prompt)

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

    @staticmethod
    def _extract_split_candidates(prompt: str) -> List[str]:
        normalized = " ".join(prompt.strip().split())
        # Backstop connectors: and / then / also / &
        parts = re.split(
            r"\s+(?:and then|then|and|or|also|&)\s+",
            normalized,
            flags=re.IGNORECASE,
        )
        cleaned = [p.strip(" ,.;:") for p in parts if p.strip(" ,.;:")]
        return cleaned if len(cleaned) >= 2 else []

    def _deterministic_split_result(
        self, prompt: str, split_candidates: List[str]
    ) -> SLMTaskAnalysis:
        prompts: List[Dict[str, str]] = []
        for fragment in split_candidates:
            tool, complexity = self._infer_tool_complexity(fragment)
            prompts.append(
                {
                    "prompt": fragment,
                    "tool": tool,
                    "complexity": complexity,
                }
            )
        return SLMTaskAnalysis.model_validate({"split": True, "prompts": prompts})

    @staticmethod
    def _infer_tool_complexity(fragment: str) -> Tuple[str, str]:
        text = fragment.lower()
        if any(k in text for k in ["code", "python", "javascript", "program", "game"]):
            return "code_generation", "high"
        if any(k in text for k in ["story", "poem", "write", "creative"]):
            return "creative_writing", "high"
        if any(k in text for k in ["summarize", "summary"]):
            return "summarization", "low"
        if any(k in text for k in ["translate"]):
            return "translation", "low"
        return "logical_reasoning", "low"
