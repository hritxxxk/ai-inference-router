from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings
from typing import Dict, List


class Settings(BaseSettings):
    """Application settings loaded from environment variables or config file."""
    
    # Model costs in USD per request
    gemini_cost: float = 0.015
    gemma_cost: float = 0.001
    code_model_cost: float = 0.008
    math_model_cost: float = 0.012
    
    # Routing/task analysis configuration
    task_keywords: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "code": [
                "function", "class", "bug", "stack trace", "compile", "python", "javascript",
                "typescript", "exception", "refactor", "optimize", "sql", "query"
            ],
            "math": [
                "integral", "derivative", "equation", "matrix", "proof", "theorem", "sigma",
                "pi", "log", "limit", "algebra", "probability", "gradient"
            ],
            "reasoning": [
                "analyze", "compare", "assess", "explain", "strategy", "evaluate", "why",
                "how", "justify", "debate", "contrast"
            ],
            "translation": [
                "translate", "in spanish", "in french", "english to", "rewrite in", "portuguese",
                "german", "mandarin"
            ],
            "summarization": [
                "summarize", "tl;dr", "bullet", "key points", "abstract", "recap", "overview"
            ],
        }
    )
    task_prototypes: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "code": [
                "Find the bug in this function and rewrite it",
                "Explain what this Python snippet does",
                "Refactor the JavaScript component for better readability",
                "Optimize this SQL query for performance",
                "Generate unit tests for the following class"
            ],
            "math": [
                "Solve the differential equation step by step",
                "Calculate the integral and justify each manipulation",
                "Prove the given theorem rigorously",
                "Evaluate this probability distribution",
                "Compute the eigenvalues of the matrix"
            ],
            "reasoning": [
                "Analyze the business strategy and highlight risks",
                "Compare historical events with modern implications",
                "Discuss the ethical trade-offs in detail",
                "Explain the geopolitical consequences of the policy",
                "Assess the strengths and weaknesses of the proposal"
            ],
            "translation": [
                "Translate the following paragraph to Spanish",
                "Rewrite this in Mandarin",
                "Provide the German equivalent of the sentence",
                "Convert the statement from English to French",
                "Translate this product description to Portuguese"
            ],
            "summarization": [
                "Summarize this document in three bullet points",
                "Give me a concise TLDR of the article",
                "Extract the key highlights from the memo",
                "Provide a short abstract for the report",
                "Summarize this conversation for executives"
            ],
        }
    )
    router_heads: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {
            "Gemma3": {
                "bias": -0.15,
                "normalized_tokens": -1.5,
                "complexity_score": -1.0,
                "code_signal": -1.1,
                "math_signal": -0.7,
                "reasoning_signal": -0.3,
            },
            "Gemini-2.5-Pro": {
                "bias": 0.35,
                "normalized_tokens": 1.5,
                "complexity_score": 1.6,
                "reasoning_signal": 0.6,
                "summarization_signal": 0.4,
            },
            "CodeLlama-Sim": {
                "bias": -0.1,
                "code_signal": 2.2,
                "normalized_tokens": 0.4,
            },
            "MathHammer": {
                "bias": -0.35,
                "math_signal": 2.4,
                "normalized_tokens": 0.5,
            },
        }
    )
    router_weights_path: str = "data/router_weights.json"
    
    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Vector DB settings
    chroma_db_path: str = "./chroma_db"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    
    # Semantic cache settings
    similarity_threshold: float = 0.9
    max_cache_size: int = 10000
    eviction_policy: str = "LRU"  # Options: LRU, FIFO, TTL
    cache_ttl_seconds: int = 86400
    semantic_distance_metric: str = "cosine"  # cosine or l2
    
    # Quota settings
    quota_default_limit: int = 10
    
    # Complexity classifier settings
    token_threshold: int = 128  # Threshold for high complexity
    
    # Telemetry
    telemetry_enabled: bool = True
    telemetry_channel: str = "router.telemetry"
    telemetry_db_path: str = "data/router.db"

    # Feedback / HITL pipeline
    feedback_api_key: str = "dev-feedback-key"
    feedback_api_enabled: bool = True

    model_config = ConfigDict(env_prefix="ROUTER_", case_sensitive=False)


settings = Settings()
