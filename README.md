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
- `SLM_TASK_API_URL` (use `/v1/chat/completions` for Qwen split tasking)
- `SLM_TASK_MODEL`
- `SLM_TASK_TEMPERATURE`
- `SLM_TASK_TOP_P`
- `SLM_TASK_MAX_TOKENS`
- `SLM_TASK_ENABLE_THINKING` (set `false` for deterministic non-thinking classification)

## Test

```bash
pytest
```

## Split Evaluation

Run labeled split evaluation against your Qwen endpoint:

```bash
python3 scripts/eval_split.py \
  --dataset tests/data/split_eval_sample.jsonl \
  --api-url http://192.168.50.218:8001/v1/chat/completions \
  --model Qwen3.5-0.8B \
  --temperature 0.1 \
  --top-p 0.8
```

## Validation Run (Documented)

Prompt tested through full chain (`/v1/star`):

```json
{
  "session_id": "doc-run-1",
  "prompt": "write a creative story and make code game for it"
}
```

Observed behavior from live GPU endpoint (`http://192.168.50.218:8080/v1/star`):
- Route chain executed: `/v1/star -> /v1/pil-clean -> /v1/clean`
- SLM split decision: `split -> 2 task(s)`
- Task 1 routed to `gpt-4o` (creative writing output returned)
- Task 2 routed to `claude-opus-4-7` (code generation output returned)
- Timing lines present in `final_response.output`:
  - `slm_inference_ms=3412.488`
  - `task_index=1 model_inference_ms=9193.853`
  - `task_index=2 model_inference_ms=10906.005`
  - `total_ms=14342.954`

Reproduce with:

```bash
curl -sS -X POST http://192.168.50.218:8080/v1/star \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"doc-run-1","prompt":"write a creative story and make code game for it"}'
```
