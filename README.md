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

## Test

```bash
pytest
```
