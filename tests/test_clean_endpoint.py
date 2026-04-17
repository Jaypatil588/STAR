import asyncio
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.models import SLMTaskAnalysis

class MockSLMTaskClient:
    async def analyze(self, prompt, session_id):
        return SLMTaskAnalysis(
            split=True,
            prompts=[
                {"prompt": "Get docs", "tool": "web_search", "complexity": "low"},
                {"prompt": "Write code", "tool": "code_generation", "complexity": "high"}
            ]
        )

@pytest.fixture
def override_slm_client():
    app.state.slm_task_client = MockSLMTaskClient()

def test_clean_endpoint(override_slm_client):
    client = TestClient(app)
    response = client.post("/v1/clean", json={"prompt": "Get docs and write code", "session_id": "s1"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["task_analysis"]["split"] is True
    assert len(data["task_analysis"]["prompts"]) == 2
    
    agent_result = data["agent_result"]
    assert agent_result["status"] == "pending_execution"
    assert len(agent_result["execution_plan"]) == 2
    
    # Check resolution logic:
    # web_search, low -> chatgpt-nano
    assert agent_result["execution_plan"][0]["assigned_model"] == "chatgpt-nano"
    # code_generation, high -> claude-opus-4.7
    assert agent_result["execution_plan"][1]["assigned_model"] == "claude-opus-4.7"
