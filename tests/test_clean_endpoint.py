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
                {"prompt": "Write code", "tool": "code_generation", "complexity": "high"},
            ],
        )


@pytest.fixture
def override_slm_client():
    app.state.slm_task_client = MockSLMTaskClient()


def test_clean_endpoint_streams(override_slm_client):
    client = TestClient(app)
    response = client.post(
        "/v1/clean",
        json={"prompt": "Get docs and write code", "session_id": "s1"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")

    text = response.text

    # Header lines present
    assert "[STAR]" in text
    assert "task(s) dispatched concurrently" in text

    # Both tasks streamed
    assert "Web Search" in text or "web_search" in text.lower()
    assert "Code Generation" in text or "code_generation" in text.lower()

    # Model routing correct per dispatch matrix
    # web_search low -> gpt-4o-mini
    assert "gpt-4o-mini" in text
    # code_generation high -> claude-opus-4-7
    assert "claude-opus-4-7" in text

    # Footer present
    assert "Done." in text
