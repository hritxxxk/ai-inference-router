# AI Inference Router

A production-grade FastAPI application that intelligently routes AI inference requests to optimize cost, latency, and user satisfaction. The router analyzes every prompt, logs the full feature vector, and makes data-backed decisions on whether to invoke high-end or budget models. 

## Quick Start
Required API keys:
   ```
   GOOGLE_API_KEY=your-google-key
   OPENAI_API_KEY=your-openai-key
   ROUTER_FEEDBACK_API_KEY=dev-feedback-key
   ```
To start - `uvicorn src.main:app --reload --host 0.0.0.0 --port 8000` or `docker compose up --build` 
   - See `AGENTS.md` for contributor etiquette once you are running locally.

## Features
- **Intelligent Routing** – Complexity + task analyzers feed a weight provider that can hot-reload learned coefficients.
- **Quota Enforcement** – Client-level limits with pluggable storage to stop quota abuse.
- **Semantic Cache** – Shared embedding model + async-safe Chroma integration with deterministic in-memory fallback for tests.
- **Cost Awareness** – Responses include savings vs. gemini-3.1-pro so teams can prove ROI.
- **Observability** – Structured logging, latency middleware, and telemetry rows for every request.
- **HITL Feedback** – `/feedback` endpoint stores reviewer labels without blocking the FastAPI event loop.

## Project Layout
- `src/main.py` wires routes, middleware, dependency injection, and shared singletons.
- `src/services/` contains router heads, quota manager, cache, and telemetry store.
- `src/models/` defines Pydantic schemas for requests/responses.
- `src/utils/` houses helpers such as token counters and embedding adapters.
- `scripts/train_router.py` consumes telemetry + feedback to export weights.
- `tests/` mirrors the service layout for pytest coverage.

## Configuration & Environment
Settings live in `src/config.py` (Pydantic `Settings` class) and accept `ROUTER_*` environment variables or `config.yaml` overrides. Common knobs:
- `ROUTER_FEEDBACK_API_KEY` – required `x-api-key` header for `/feedback`.
- `ROUTER_TELEMETRY_DB_PATH` – SQLite path (default `data/router.db`).
- `ROUTER_ROUTER_WEIGHTS_PATH` – JSON file watched by `WeightProvider`.
- `ROUTER_SIMILARITY_THRESHOLD`, `ROUTER_TOKEN_THRESHOLD`, `ROUTER_QUOTA_DEFAULT_LIMIT` for routing/cost heuristics.
- `GOOGLE_API_KEY` / `OPENAI_API_KEY` – loaded via python-dotenv to unlock the real gemini/gemma and GPT-5.3 Codex calls (NOTE: the router falls back to deterministic simulations when unset).

## Usage & API
```bash
uvicorn src.main:app --reload
```
Endpoints:
- `POST /generate` – Core inference request.
- `GET /health` – Liveness probe.
- `POST /feedback` – Human label submission (requires `x-api-key`).

Example generate payload:
```json
{
  "prompt": "Explain quantum computing in simple terms",
  "client_id": "client_001"
}
```
Responses echo routing metadata such as `model_used`, `latency_ms`, `cost_avoided_usd`, and `complexity_analysis`.

Feedback submission:
```bash
curl -X POST http://localhost:8000/feedback   -H "Content-Type: application/json"   -H "x-api-key: dev-feedback-key"   -d '{
    "request_id": "<uuid-from-logs>",
    "label": "incorrect",
    "preferred_model": "gpt-5.3-codex",
    "notes": "Prefer a code-focused model",
    "quality_score": 4
  }'
```

## Telemetry & Feedback Data Model
- `telemetry_decisions` – One row per `/generate` call storing `request_id`, selected model, latency, cost estimates, prompt embedding, and serialized feature vector.
- `feedback` – Optional reviewer labels joined via the same `request_id`, including `label`, `preferred_model`, and `quality_score` weights.
- Writes are queued through `TelemetryStore` so API handlers remain async-friendly.
- SQLite lives under `data/router.db`; back it up before destructive work.

## Training & Weight Reload
1. Capture traffic + reviewer labels as described above.
2. Run `python scripts/train_router.py --db data/router.db --out data/router_weights.json`.
3. The script fits least-squares heads per model (with quality-score weighting) and writes JSON.
4. The running app reloads new weights via `WeightProvider` automatically—no restart required.

## Testing
Use pytest + pytest-asyncio:
```bash
pytest tests
```
The suite exercises router scoring, cache fallbacks, `/feedback` auth, telemetry store behavior, and the training pipeline.
