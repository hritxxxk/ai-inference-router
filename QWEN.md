# AI Inference Router - Project Context

## Project Overview

A production-grade FastAPI application that intelligently routes AI inference requests to optimize cost, latency, and user satisfaction. The router analyzes every prompt using a multi-signal classification system (keyword matching + semantic similarity) and makes data-backed decisions on whether to invoke high-end models (gemini-3.1-pro, gemini-3-deep-think) or budget models (gemma-3-27b, gpt-5.3-codex).

### Core Architecture

- **Routing Engine** (`src/services/routing_engine.py`): Scores prompts against multiple routing heads using logistic regression with hot-reloadable weights
- **Task Analyzer** (`src/services/task_analyzer.py`): Extracts keyword signals and semantic signals to classify prompts into task types (code, math, reasoning, translation, summarization)
- **Semantic Cache** (`src/services/semantic_cache.py`): ChromaDB-backed embedding cache with configurable similarity thresholds
- **Quota Manager** (`src/services/quota_store.py`): Redis-based client-level rate limiting
- **Telemetry Store** (`src/services/telemetry_store.py`): SQLite persistence for routing decisions and human feedback
- **Weight Provider** (`src/services/weight_provider.py`): File-watching service that hot-reloads trained routing weights

### Tech Stack

- **Framework**: FastAPI 0.104.1 with async/await patterns
- **ML/Embeddings**: sentence-transformers (all-MiniLM-L6-v2), ChromaDB 0.4.15
- **Cache/Queue**: Redis 5.0+
- **Persistence**: SQLite for telemetry, ChromaDB for semantic cache
- **AI Providers**: Google GenAI, OpenAI (with deterministic fallbacks when keys unset)
- **Testing**: pytest + pytest-asyncio

## Project Structure

```
ai-inference-router/
├── src/
│   ├── main.py                 # FastAPI app, routes, middleware wiring
│   ├── config.py               # Pydantic Settings with ROUTER_* env vars
│   ├── models/
│   │   └── schemas.py          # Pydantic models (AIRequest, AIResponse, FeedbackPayload)
│   ├── services/
│   │   ├── routing_engine.py   # Core routing logic with multi-head scoring
│   │   ├── task_analyzer.py    # Keyword + semantic signal extraction
│   │   ├── weight_provider.py  # Hot-reloadable weight management
│   │   ├── semantic_cache.py   # ChromaDB embedding cache
│   │   ├── quota_store.py      # Redis quota enforcement
│   │   ├── fallback_handler.py # Model call orchestration with fallback
│   │   ├── model_calls.py      # Direct AI provider calls
│   │   ├── telemetry_store.py  # SQLite persistence layer
│   │   └── telemetry.py        # Event logging helpers
│   ├── middleware/
│   │   └── timing_middleware.py # Request timing instrumentation
│   └── utils/
│       ├── security.py         # API key verification
│       └── token_utils.py      # Token counting helpers
├── scripts/
│   └── train_router.py         # Training script for routing weights
├── tests/
│   ├── test_main_app.py        # API endpoint tests
│   └── test_core_components.py # Service layer unit tests
├── data/                       # SQLite DB, trained weights
├── chroma_db/                  # ChromaDB persistence
└── config.yaml                 # Static configuration overrides
```

## Building and Running

### Local Development

```bash
# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (.env file)
cat > .env << EOF
GOOGLE_API_KEY=your-google-key
OPENAI_API_KEY=your-openai-key
ROUTER_FEEDBACK_API_KEY=dev-feedback-key
EOF

# Run the server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Deployment

```bash
# Build and run full stack (app + Redis + Chroma)
docker compose up --build

# Services available:
# - FastAPI: http://localhost:8000
# - Redis: localhost:6379
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/generate` | Core inference request with routing |
| POST | `/generate-stream` | Streaming inference (Gemini only) |
| GET | `/health` | Liveness probe with dependency checks |
| POST | `/feedback` | Human-in-the-loop feedback (requires `x-api-key`) |

### Example Requests

**Generate:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing", "client_id": "client_001"}'
```

**Feedback:**
```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -H "x-api-key: dev-feedback-key" \
  -d '{
    "request_id": "<uuid-from-response>",
    "label": "incorrect",
    "preferred_model": "gpt-5.3-codex",
    "quality_score": 4
  }'
```

## Testing

```bash
# Run full test suite
pytest tests -vv

# Run specific test categories
pytest tests -k semantic_cache
pytest tests -k routing
pytest tests --maxfail=1
```

## Training Routing Weights

The router uses a least-squares regression model trained on telemetry + human feedback:

```bash
# After collecting traffic and feedback labels:
python scripts/train_router.py \
  --db data/router.db \
  --out data/router_weights.json

# The WeightProvider automatically reloads new weights without restart
```

## Configuration

Settings use Pydantic Settings with `ROUTER_` environment variable prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `ROUTER_FEEDBACK_API_KEY` | `dev-feedback-key` | API key for `/feedback` endpoint |
| `ROUTER_TELEMETRY_DB_PATH` | `data/router.db` | SQLite path for telemetry |
| `ROUTER_ROUTER_WEIGHTS_PATH` | `data/router_weights.json` | Trained weights file |
| `ROUTER_SIMILARITY_THRESHOLD` | `0.9` | Semantic cache match threshold |
| `ROUTER_TOKEN_THRESHOLD` | `128` | Complexity classifier threshold |
| `ROUTER_QUOTA_DEFAULT_LIMIT` | `10` | Default client quota |
| `ROUTER_REDIS_HOST` | `localhost` | Redis connection host |
| `ROUTER_CHROMA_DB_PATH` | `./chroma_db` | ChromaDB persistence path |

## Development Conventions

### Code Style
- PEP 8 with 4-space indentation
- Type hints required on all functions
- Module-level loggers: `logging.getLogger(__name__)`
- snake_case for files/functions, PascalCase for Pydantic models, SHOUTY_SNAKE_CASE for constants

### Testing Practices
- Mirror source directory structure in `tests/`
- Use `test_<unit>_<behavior>` naming convention
- Use pytest-asyncio for async tests
- Include sample request/response payloads in PR descriptions

### Commit Guidelines
- Conventional commit prefixes: `feat:`, `fix:`, `chore:`, `docs:`
- Imperative mood in commit messages
- Flag API/contract changes early for SDK coordination

## Key Design Decisions

1. **Multi-Signal Classification**: Combines keyword matching (fast, interpretable) with semantic similarity (robust to paraphrasing)
2. **Hot-Reloadable Weights**: `WeightProvider` watches the weights JSON file, enabling model updates without downtime
3. **Fallback Chain**: Every routing decision includes a fallback model for resilience
4. **Async-First**: All I/O operations (DB, cache, AI calls) use async patterns
5. **Cost Awareness**: Every response includes `cost_estimate` and `cost_avoided_usd` for ROI tracking
6. **HITL Pipeline**: Human feedback is decoupled from the request path via async task queues

## Common Tasks

### Add a new routing signal
1. Update `PromptAnalysis` in `task_analyzer.py` to extract the signal
2. Add the signal to `features_from_analysis()` in `routing_engine.py`
3. Add default weights in `config.py` under `router_heads`

### Adjust routing behavior
1. Modify `router_heads` in `config.py` for immediate effect
2. Or train new weights using `scripts/train_router.py` with collected feedback

### Debug routing decisions
Check the `metadata` field in `/generate` responses for:
- `task_type`: Detected task category
- `confidence`: Routing confidence (0-1)
- `features`: Normalized feature values used in scoring
- `reasons`: Human-readable explanation of the decision
