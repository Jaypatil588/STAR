from app.models import SLMTaskAnalysis

async def callAgents(task_analysis: SLMTaskAnalysis) -> dict:
    """Dummy agent dispatch — will be implemented later."""
    return {
        "status": "pending",
        "message": "Agent dispatch not yet implemented",
        "task_count": len(task_analysis.prompts),
        "tasks_received": [t.model_dump() for t in task_analysis.prompts],
    }
