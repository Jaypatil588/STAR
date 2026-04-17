from fastapi.testclient import TestClient

from app.main import app


class MockEndpointCaller:
    def __init__(self):
        self.calls = []

    async def post(self, path, payload):
        self.calls.append({"path": path, "payload": payload})
        if path == "/v1/pil-clean":
            return {
                "status": "success",
                "final_response": {"split": False, "prompts": []},
                "clean_response": {"status": "success"},
            }
        if path == "/v1/clean":
            return {
                "status": "success",
                "output": "[STAR] split → 2 task(s) dispatched concurrently",
            }
        return {"status": "error", "error": "unexpected path"}


def test_star_forwards_to_pil_clean():
    caller = MockEndpointCaller()
    app.state.endpoint_caller = caller
    client = TestClient(app)

    response = client.post(
        "/v1/star",
        json={"prompt": "hello    world", "session_id": "s-star"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["fixed_response"]["next_endpoint"] == "/v1/pil-clean"
    assert len(caller.calls) == 1
    assert caller.calls[0]["path"] == "/v1/pil-clean"


def test_pil_clean_forwards_to_clean():
    caller = MockEndpointCaller()
    app.state.endpoint_caller = caller
    client = TestClient(app)

    response = client.post(
        "/v1/pil-clean",
        json={"prompt": "hi   there", "session_id": "s-pil"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["cleaned_prompt"] == "hi there"
    assert data["fixed_response"]["next_endpoint"] == "/v1/clean"
    assert "output" in data["final_response"]
    assert len(caller.calls) == 1
    assert caller.calls[0]["path"] == "/v1/clean"
