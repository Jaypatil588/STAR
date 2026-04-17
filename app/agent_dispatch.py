from typing import Any, Dict

from app.models import SLMTaskAnalysis, ALLOWED_TOOLS


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


async def callAgents(task_analysis: SLMTaskAnalysis) -> dict[str, Any]:
    """
    Dummy agent dispatch. Iterates over sub-tasks from the SLM,
    resolves the accurate model mapping for each, and returning the structured plan.
    """
    execution_plan = []
    
    for task in task_analysis.prompts:
        target_model = resolve_model_for_task(task.tool, task.complexity)
        
        assigned_task = task.model_dump()
        assigned_task["assigned_model"] = target_model
        assigned_task["status"] = "queued_for_dispatch"
        
        execution_plan.append(assigned_task)
        
    return {
        "status": "pending_execution",
        "message": "Tasks successfully queued grouped by target models",
        "task_count": len(execution_plan),
        "execution_plan": execution_plan,
    }
