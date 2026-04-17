import asyncio
import httpx
from typing import Any, Dict

from app.models import SLMTaskAnalysis, ALLOWED_TOOLS
from app.config import get_settings


# Group configurations mapping tools to capability sectors
CODE_GROUP = {
    "code_generation", "code_review", "code_translation", 
    "dom_manipulation", "terminal_execution", "api_call"
}

TEXT_GROUP = {
    "summarization", "translation", "creative_writing", 
    "copy_editing", "web_search", "internal_rag", 
    "fact_check", "entity_extraction"
}

REASONING_GROUP = {
    "math_compute", "logical_reasoning", "data_analysis"
}

MULTIMODAL_GROUP = {
    "image_generation", "vision_analysis", "audio_processing"
}

def resolve_model_for_task(tool: str, complexity: str) -> str:
    """
    Determines the target LLM based on tool category and complexity.
    """
    # Group 1: Code & Execution
    if tool in CODE_GROUP:
        return "claude-opus-4.7" if complexity == "high" else "claude-sonnet-4.6"
    
    # Group 2: Instructions, RAG, & Text
    if tool in TEXT_GROUP:
        return "gpt-5.4" if complexity == "high" else "gpt-5.4-nano"
        
    # Group 3: Math, Logic, & Data
    if tool in REASONING_GROUP:
        return "gpt-5.4-thinking" if complexity == "high" else "claude-sonnet-4.6"
        
    # Group 4: Vision & Audio Processing
    if tool in MULTIMODAL_GROUP:
        return "gpt-5.4" if complexity == "high" else "gpt-5.4-nano"
        
    # Fallback default
    return "gpt-5.4-nano"


async def _call_openai(model: str, prompt: str) -> str:
    settings = get_settings()
    api_key = settings.openai_api_key
    if not api_key:
        return f"[Error: openai_api_key not configured for model {model}]"
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[Error calling OpenAI API: {str(e)}]"

async def _call_anthropic(model: str, prompt: str) -> str:
    settings = get_settings()
    api_key = settings.anthropic_api_key
    if not api_key:
        return f"[Error: anthropic_api_key not configured for model {model}]"
        
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    payload = {
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]
    except Exception as e:
        return f"[Error calling Anthropic API: {str(e)}]"

async def _execute_task(task: Dict[str, Any]) -> Dict[str, Any]:
    model = task["assigned_model"]
    prompt = task["prompt"]
    
    if "claude" in model.lower():
        output = await _call_anthropic(model, prompt)
    else:
        output = await _call_openai(model, prompt)
        
    task["output"] = output
    task["status"] = "completed"
    return task

async def callAgents(task_analysis: SLMTaskAnalysis) -> dict[str, Any]:
    """
    Agent dispatch. Iterates over sub-tasks from the SLM,
    resolves the accurate model mapping for each, and executes them concurrently.
    """
    execution_plan = []
    
    for task in task_analysis.prompts:
        target_model = resolve_model_for_task(task.tool, task.complexity)
        
        assigned_task = task.model_dump()
        assigned_task["assigned_model"] = target_model
        assigned_task["status"] = "executing"
        
        execution_plan.append(assigned_task)
        
    # Concurrently execute all assigned tasks
    executed_tasks = await asyncio.gather(*(
        _execute_task(t) for t in execution_plan
    ))
        
    return {
        "status": "completed",
        "message": "Tasks successfully executed by target models",
        "task_count": len(executed_tasks),
        "execution_plan": list(executed_tasks),
    }
