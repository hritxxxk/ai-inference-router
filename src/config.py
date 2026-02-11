from pydantic_settings import BaseSettings
from typing import Dict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or config file."""
    
    # Model costs in USD per request
    gemini_cost: float = 0.015
    gemma_cost: float = 0.001
    
    # Semantic cache settings
    similarity_threshold: float = 0.9
    max_cache_size: int = 10000
    eviction_policy: str = "LRU"  # Options: LRU, FIFO, TTL
    
    # Quota settings
    quota_default_limit: int = 10
    
    # Complexity classifier settings
    token_threshold: int = 128  # Threshold for high complexity
    
    class Config:
        env_prefix = "ROUTER_"
        case_sensitive = False


settings = Settings()