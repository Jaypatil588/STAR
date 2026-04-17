# STAR Router (Phase 1)

FastAPI service that:
- Accepts `prompt + session_id`
- Calls SLM on every request
- Supports explicit endpoint chaining:
- `POST /v1/star` -> `POST /v1/pil-clean` -> `POST /v1/clean` -> QWEN
- Returns a fixed decision response at each stage plus the next-stage response

## Run

```bash
uvicorn app.main:app --reload
```

## Endpoints

- `POST /v1/star`: prompt + session, forwards to `/v1/pil-clean`
- `POST /v1/pil-clean`: prompt cleaning layer, forwards to `/v1/clean`
- `POST /v1/clean`: calls QWEN task analyzer and agent dispatch stub
- `POST /v1/routeModel`: central routing hub with `modelID`, `prompt`, optional `context`
- `POST /v1/models/gpt4o`: calls GPT-4o using `modelID`, `prompt`, optional `context`
- `POST /v1/slm-route`: stage-1 only legacy helper
- `POST /v1/route`: legacy combined orchestrator response
- `GET /health`

## Environment

Default downstream mode is HTTP placeholder (`DOWNSTREAM_PROVIDER=http`).

To use OpenAI as downstream:

```bash
export DOWNSTREAM_PROVIDER=openai
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Current key vars:
- `SLM_API_KEY`
- `PLACEHOLDER_API_KEY`
- `OPENAI_API_KEY`
- `INTERNAL_BASE_URL` (used for internal endpoint-to-endpoint calls)

## Test

```bash
pytest
```
