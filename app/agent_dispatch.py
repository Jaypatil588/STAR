import asyncio
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
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
        return "gpt-4o" if complexity == "high" else "gpt-4o-mini"
        
    # Group 3: Math, Logic, & Data
    if tool in REASONING_GROUP:
        return "gpt-4o" if complexity == "high" else "claude-sonnet-4.6"
        
    # Group 4: Vision & Audio Processing
    if tool in MULTIMODAL_GROUP:
        return "gpt-4o" if complexity == "high" else "gpt-4o-mini"
        
    # Fallback default
    return "gpt-4o-mini"


async def _call_openai(model: str, prompt: str) -> str:
    settings = get_settings()
    api_key = settings.openai_api_key
    if not api_key:
        return f"[Error: openai_api_key not configured for model {model}]"
        
    client = AsyncOpenAI(api_key=api_key)
    
    try:
        # Using the specialized Responses API as requested
        response = await client.responses.create(
            model=model,
            input=[{
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}]
            }],
            text={
                "format": {
                    "type": "text"
                }
            },
            reasoning={},
            tools=[],
            temperature=1,
            max_output_tokens=2048,
            top_p=1,
            store=True,
            include=["web_search_call.action.sources"],
        )
        # Inspect response if output_text is missing
        try:
            return response.output_text
        except AttributeError:
            # Fallback for different SDK versions or structured output
            outputitems = getattr(response, "output", [])
            for item in outputitems:
                if hasattr(item, "content"):
                    for content_part in item.content:
                        if hasattr(content_part, "text"):
                            return content_part.text
            
            # Final fallback: dump the whole thing to see what's inside
            return f"[Error: Responses API returned unexpected structure. Available fields: {list(response.__dict__.keys())}]"
    except Exception as e:
        return f"[Error calling OpenAI Responses API: {str(e)}]"

async def _call_anthropic(model: str, prompt: str) -> str:
    settings = get_settings()
    api_key = settings.anthropic_api_key
    if not api_key:
        return f"[Error: anthropic_api_key not configured for model {model}]"
        
    client = AsyncAnthropic(api_key=api_key)
    
    try:
        response = await client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            timeout=30.0
        )
        return response.content[0].text
    except Exception as e:
        return f"[Error calling Anthropic SDK: {str(e)}]"

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
    resolves the accurate model mapping for each, executes them concurrently,
    and combines all outputs into a unified response.
    """
    execution_plan = []
    
    for task in task_analysis.prompts:
        target_model = resolve_model_for_task(task.tool, task.complexity)
        
        assigned_task = task.model_dump()
        assigned_task["assigned_model"] = target_model
        assigned_task["status"] = "executing"
        
        execution_plan.append(assigned_task)
        
    # Concurrently execute all assigned tasks
    executed_tasks = list(await asyncio.gather(*(
        _execute_task(t) for t in execution_plan
    )))
    
    # Combine all outputs into a single message
    if len(executed_tasks) == 1:
        combined_output = executed_tasks[0].get("output", "")
    else:
        parts = []
        for i, task in enumerate(executed_tasks, 1):
            label = task.get("tool", f"task_{i}").replace("_", " ").title()
            model = task.get("assigned_model", "unknown")
            output = task.get("output", "")
            parts.append(f"### [{i}] {label} (via {model})\n\n{output}")
        combined_output = "\n\n---\n\n".join(parts)
        
    return {
        "status": "completed",
        "message": "Tasks successfully executed by target models",
        "task_count": len(executed_tasks),
        "execution_plan": executed_tasks,
        "combined_output": combined_output,
    }
