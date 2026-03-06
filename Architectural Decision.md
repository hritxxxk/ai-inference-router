# Architectural Decision Records (ADR)

### 1. Why Redis + Celery instead of standard API calls?
**Context:** Our catalog enrichment pipeline required processing 5,000+ brochure pages. 
**Decision:** Implemented an asynchronous worker pattern using Redis as a broker and Celery for execution.
**Rationale:** Standard synchronous FastAPI endpoints would time out during heavy GPU inference (YOLOv8/Gemini). This pattern ensures 100% task durability, allows for horizontal scaling of workers, and provides a seamless "polling" experience for the client UI.

### 2. Task-Based Model Routing (Gemini vs. Gemma3)
**Context:** LLM API costs were scaling linearly with volume, threatening unit economics.
**Decision:** Built a lightweight complexity classifier to route "Simple Extraction" to a fine-tuned Gemma3 (local/low-cost) and "Complex Reasoning" to Gemini Pro.
**Rationale:** Reduced inference costs by ~40% while maintaining a 10s/product latency SLA. It proves that the "largest" model isn't always the "best" model for production.

### 3. State Management in Postgres
**Context:** Agentic workflows (LangGraph) require persistent memory for "Human-in-the-Loop" interactions.
**Decision:** Utilized Postgres to store agent state and "checkpoints."
**Rationale:** Allows the system to "pause" if confidence scores fall below 0.7, enabling human resolution without losing the execution context.
