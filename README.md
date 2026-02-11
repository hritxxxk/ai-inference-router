# AI Inference Router

A production-grade FastAPI application that intelligently routes AI inference requests to optimize cost and performance. The router evaluates incoming prompts and decides whether to use expensive, high-capability models or cheaper, faster alternatives based on complexity analysis.

## Features

- **Intelligent Routing**: Analyzes prompts using configurable token count threshold and reasoning keywords to determine optimal model selection
- **Quota Management**: Tracks and enforces per-client request limits with an extensible storage interface
- **Semantic Caching**: Implements vector similarity-based caching to avoid recomputation of similar requests
- **Cost Optimization**: Routes simple requests to cheaper models, saving up to 93% on costs
- **Reliability**: Includes fallback mechanisms to ensure requests are processed even if preferred models fail
- **Observability**: Comprehensive logging and timing middleware for performance monitoring
- **Configurable**: Uses Pydantic settings for easy configuration of costs, thresholds, and limits
- **Response Enrichment**: Provides detailed metadata including cost savings, latency, and token counts

## Architecture

The application follows a service-oriented architecture with clear separation of concerns:

- `models/` - Pydantic schemas for request/response validation
- `services/` - Business logic for routing, caching, classification, etc.
- `utils/` - Utility functions like token counting
- `middleware/` - Cross-cutting concerns like request timing
- `config.py` - Application configuration using Pydantic Settings

## Configuration

The application uses Pydantic Settings for configuration management. Key configurable parameters include:

- Model costs (gemini_pro, gemma3)
- Semantic cache settings (similarity threshold, max cache size, eviction policy)
- Quota settings (default limit)
- Classifier settings (token threshold)

Configuration can be set via environment variables prefixed with `ROUTER_` or by modifying the settings class directly.

## Components

### Complexity Classifier
Analyzes prompts using:
- Configurable token count threshold (default: 128 tokens)
- Reasoning keywords (analyze, explain, summarize, etc.)

### Quota Manager
Abstract interface supporting different storage backends (in-memory for POC, Redis-ready for production)

### Semantic Cache
Simulates vector database lookup using mocked cosine similarity with configurable threshold

### Model Call Handlers
Mock implementations showing latency differences between expensive and cheap models

### Response Aggregator
Builds comprehensive responses with cost and performance metrics

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

- `POST /generate` - Submit a prompt for AI processing
- `GET /health` - Health check endpoint

### Example Request

```json
{
  "prompt": "Explain quantum computing in simple terms",
  "client_id": "client_001"
}
```

### Example Response

```json
{
  "result": "Quantum computing explained...",
  "model_used": "Gemma3",
  "latency_ms": 450.23,
  "cached": false,
  "cost_estimate": 0.001,
  "metadata": {
    "tokens": 120,
    "latency_ms": 450.23,
    "optimization_strategy": "Routed to LOW Complexity",
    "cost_avoided_usd": 0.014,
    "savings_multiplier": "15.0x cheaper",
    "complexity_analysis": {
      "token_count": 8,
      "has_complex_intent": false,
      "is_high_token_count": false,
      "complex_keywords_found": []
    }
  }
}
```

## Testing

Run the tests using pytest:

```bash
pytest tests/
```

## Design Principles

- **Extensibility**: Abstract interfaces allow easy swapping of components (e.g., cache storage, quota backend)
- **Configurability**: Key parameters are configurable via environment variables
- **Observability**: Comprehensive logging and timing information for monitoring
- **Cost Efficiency**: Intelligent routing minimizes expensive model usage
- **Reliability**: Fallback mechanisms ensure service availability
- **Performance**: Caching and intelligent routing reduce response times