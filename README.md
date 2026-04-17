# STAR Router (Phase 1)

FastAPI service that:
- Accepts `prompt + session_id`
- Calls SLM on every request
- Routes first to a placeholder downstream target
- Summarizes context after routing
- Persists rolling per-session context in memory
- Returns two explicit stages:
- `response_1`: SLM routing decision (`model_id` + routing metadata)
- `response_2`: model execution result from the selected model

## Run

```bash
uvicorn app.main:app --reload
```

## Endpoints

- `POST /v1/slm-route`: stage-1 only (SLM routing response)
- `POST /v1/route`: full flow (stage-1 routing + stage-2 model response)
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

## Test

```bash
pytest
```
