# Repository Guidelines

## Project Structure & Module Organization
The FastAPI app lives in `src/`. `main.py` wires routes, middleware, and shared services, while `config.py` exposes Pydantic settings driven by `ROUTER_*` environment variables or `config.yaml`. `services/` hosts quota, caching, classifier, and fallback logic; `models/` define Pydantic schemas; `utils/` holds helper functions; and `middleware/` collects timing/logging hooks. Keep tests in `tests/` mirroring module names (`test_main_app.py`, `test_core_components.py`) so contributors can trace fixtures and mocks quickly.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create an isolated environment.
- `pip install -r requirements.txt`: install FastAPI, Redis, Chroma, and testing dependencies.
- `uvicorn src.main:app --reload --host 0.0.0.0 --port 8000`: launch the API locally; exercise `POST /generate` and `GET /health` with curl or HTTPie.
- `docker compose up --build`: run the full stack (app + Redis/Chroma) using the provided Docker artifacts.
- `pytest tests -vv`: execute the suite; append `-k semantic_cache` when focusing on a single component.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, descriptive type hints, and module-level loggers (`logging.getLogger(__name__)`). Files and functions stay in snake_case, Pydantic models use PascalCase, and configuration constants use SHOUTY_SNAKE_CASE. Keep route handlers thin—delegate business logic to `services/` classes and always return validated `AIResponse` objects. Document new tunable knobs inside `config.py` and mention the matching `ROUTER_*` variable.

## Testing Guidelines
Use pytest plus pytest-asyncio for coroutine-heavy services. Name tests `test_<unit>_<behavior>` and mirror source directories to keep workflows discoverable. Cover routing decisions (quota, cache, classifier branches) and any new metadata fields, and run `pytest tests/test_main_app.py --maxfail=1` before submitting. Attach sample request/response payloads in PR descriptions whenever new flows are introduced.

## Commit & Pull Request Guidelines
Adopt the conventional-commit prefixes already in history (`feat:`, `fix:`, `chore:`) and keep messages in imperative mood. Each PR should include a concise summary, linked issue, configuration changes (new env vars, ports, or quotas), and manual or automated test evidence. Flag API or contract changes early so SDK and client updates stay coordinated.

## Security & Configuration Tips
Never commit provider keys or client quotas—load them through the Pydantic settings layer or a local `.env` that stays untracked. Validate configuration with `curl localhost:8000/health` or `docker compose logs` before pushing. When introducing a dependency (e.g., another cache backend), document its ports, required secrets, and failure behavior alongside the affected service docstring.
