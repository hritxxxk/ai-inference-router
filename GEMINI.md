# Gemini Project Context: AI Inference Router

This project is a FastAPI-based AI inference router designed to optimize costs and performance by intelligently routing requests between high-capability (expensive) and lower-capability (cheap) models.

## Project Overview

- **Purpose**: Intelligently route AI prompts to minimize cost while maintaining response quality.
- **Main Technologies**: 
  - **Framework**: FastAPI
  - **Validation**: Pydantic v2
  - **Persistence**: 
    - **Redis**: For global quota management and rate limiting.
    - **ChromaDB**: For persistent semantic cache storage.
  - **Intelligence**:
    - **Sentence Transformers**: Using `all-MiniLM-L6-v2` for semantic search and intent classification.
  - **NLP Utilities**: Tiktoken (for token counting)
- **Architecture**: Service-oriented architecture with clear separation between routing logic, model execution, caching, and quota management.

## Key Components

- **`src/main.py`**: Entry point with support for both standard and streaming (`/generate-stream`) responses.
- **`src/services/classifier.py`**: Hybrid complexity detection (Token count + Keywords + Embedding similarity to known complex tasks).
- **`src/services/semantic_cache.py`**: Persistent vector-based cache using ChromaDB.
- **`src/services/quota_store.py`**: Redis-backed quota management for horizontal scalability.

## Building and Running

- **Prerequisites**: 
  - Redis server running on `localhost:6379` (configurable).
- **Install Dependencies**:
  ```bash
  pip install -r requirements.txt
  ```
- **Run the Application**:
  ```bash
  uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
  ```

## Development Conventions

- **Semantic Cache**: Uses `similarity_threshold` (default 0.9). Values closer to 1.0 require higher similarity.
- **Complexity Analysis**: Now includes a `semantic_complexity_score` based on cosine similarity to reference complex tasks.
- **Streaming**: Supports Server-Sent Events style streaming via `/generate-stream`.


## API Endpoints

- `POST /generate`: Submit a prompt for routing and processing.
- `GET /health`: Basic health check.

## Testing Practices

- The project uses `pytest` with `httpx` for integration testing of the FastAPI app.
- Tests cover health checks, basic generation, quota enforcement, and semantic caching.
