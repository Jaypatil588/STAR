from __future__ import annotations

import json
from typing import AsyncGenerator, Dict, Any, Optional
import httpx
import logging

from app.models import SLMTaskAnalysis, ALLOWED_TOOLS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a strict task analysis engine. Given a user prompt, you must:
1. Decide if the prompt should be split into multiple sub-tasks (split: true/false).
2. For each sub-task, output its prompt, tool, and complexity.

STRICT RULES:
- For each sub-task, you must assign a category strictly from the following allowed list. Do not use any category outside of this list:
[math_compute, logical_reasoning, data_analysis, code_generation, code_review, code_translation, web_search, internal_rag, fact_check, summarization, translation, creative_writing, copy_editing, entity_extraction, image_generation, vision_analysis, audio_processing, dom_manipulation, terminal_execution, api_call]
- complexity MUST be "high" or "low"
- DO NOT wrap the output in markdown code blocks like ```json ... ```. Just return the raw JSON string.

SCHEMA:
{{
  "split": boolean,
  "prompts": [
    {{
      "prompt": string,
      "tool": string,
      "complexity": string
    }}
  ]
}}

EXAMPLES:

User: What is 2 + 2?
Output: {{"split": false, "prompts": [{{"prompt": "What is 2 + 2?", "tool": "math_compute", "complexity": "low"}}]}}

User: Write a python script to parse a CSV and then summarize the key findings.
Output: {{"split": true, "prompts": [{{"prompt": "Write a python script to parse a CSV", "tool": "code_generation", "complexity": "high"}}, {{"prompt": "Summarize the key findings from the parsed CSV", "tool": "summarization", "complexity": "low"}}]}}

User: Translate this text to french and fix any grammar errors.
Output: {{"split": true, "prompts": [{{"prompt": "Translate this text to french", "tool": "translation", "complexity": "low"}}, {{"prompt": "Fix any grammar errors in the translated text", "tool": "copy_editing", "complexity": "low"}}]}}

User: Analyze this image and extract the text.
Output: {{"split": true, "prompts": [{{"prompt": "Analyze this image", "tool": "vision_analysis", "complexity": "high"}}, {{"prompt": "Extract the text from the image", "tool": "entity_extraction", "complexity": "low"}}]}}

User: Search the web for latest AI news.
Output: {{"split": false, "prompts": [{{"prompt": "Search the web for latest AI news", "tool": "web_search", "complexity": "low"}}]}}

User: Generate a picture of a cat.
Output: {{"split": false, "prompts": [{{"prompt": "Generate a picture of a cat.", "tool": "image_generation", "complexity": "low"}}]}}

User: Can you check if this URL is safe?
Output: {{"split": false, "prompts": [{{"prompt": "Can you check if this URL is safe?", "tool": "logical_reasoning", "complexity": "low"}}]}}

User: Summarize this document.
Output: {{"split": false, "prompts": [{{"prompt": "Summarize this document.", "tool": "summarization", "complexity": "low"}}]}}

User: I have a complex problem, I need a python script to download a webpage, parse its DOM, extract all the images, and then save them to my local disk.
Output: {{"split": true, "prompts": [{{"prompt": "Write a python script to download a webpage", "tool": "code_generation", "complexity": "low"}}, {{"prompt": "Parse the DOM of the downloaded webpage", "tool": "dom_manipulation", "complexity": "low"}}, {{"prompt": "Extract all the images", "tool": "entity_extraction", "complexity": "low"}}, {{"prompt": "Save the images to my local disk", "tool": "terminal_execution", "complexity": "low"}}]}}

User: Delete all files in the current folder.
Output: {{"split": false, "prompts": [{{"prompt": "Delete all files in the current folder.", "tool": "terminal_execution", "complexity": "high"}}]}}
"""

class SLMTaskClient:
    def __init__(self, api_url: str, timeout_ms: int = 15_000):
        self._api_url = api_url
        self._timeout = timeout_ms / 1_000.0

    async def analyze(self, prompt: str, session_id: str) -> SLMTaskAnalysis:
        # We try up to 2 times to get valid JSON
        for attempt in range(2):
            try:
                raw_json = await self._call_vllm(prompt)
                # Clean up any potential markdown formatting the SLM might add
                cleaned = raw_json.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                
                parsed = json.loads(cleaned.strip())
                return SLMTaskAnalysis.model_validate(parsed)
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed to parse SLM task analysis: {e}. Raw: {raw_json if 'raw_json' in locals() else 'None'}")
        
        # Fallback to a safe single-task default
        logger.error("Failed to parse SLM task analysis after 2 attempts. Falling back to default.")
        return SLMTaskAnalysis(
            split=False,
            prompts=[{
                "prompt": prompt,
                "tool": "logical_reasoning",
                "complexity": "high"
            }]
        )

    async def _call_vllm(self, prompt: str) -> str:
        body = {
            "model": "Qwen3.5-0.8B",
            "prompt": f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            "max_tokens": 1024,
            "temperature": 0.1,
            "stop": ["<|im_end|>"]
        }
        
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(self._api_url, json=body)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["text"]
