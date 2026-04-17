# STAR Router (Phase 1)

FastAPI service that:
- Accepts `prompt + session_id`
- Calls SLM on every request
- Routes first to a placeholder downstream target
- Summarizes context after routing
- Persists rolling per-session context in memory

## Run

```bash
uvicorn app.main:app --reload
```

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
