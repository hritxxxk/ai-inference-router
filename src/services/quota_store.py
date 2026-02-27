"""
Module for managing client quotas in the AI inference router.

This module implements a flexible quota management system with an abstract base class
that allows for different storage backends. The design supports easy swapping of
implementations (e.g., from in-memory to Redis) without changing the business logic.
"""

from abc import ABC, abstractmethod
from typing import Dict
import redis
from src.config import settings


class BaseQuotaStore(ABC):
    """
    Abstract base class for quota storage.
    
    This interface enables the implementation of different storage backends
    (in-memory, Redis, database) while maintaining the same contract for
    the business logic. This follows the Dependency Inversion Principle
    allowing for easy testing and production deployment with different
    storage systems.
    """
    
    @abstractmethod
    def check_quota(self, client_id: str, increment: bool = True) -> bool:
        """
        Check if a client has exceeded their quota.
        
        This method checks if the client has remaining quota and optionally
        increments their usage count. The increment parameter allows for
        checking quota without necessarily consuming it, which is useful
        for validation purposes.
        
        Args:
            client_id: Unique identifier for the client
            increment: Whether to increment usage after checking
            
        Returns:
            True if quota is available, False if exceeded
        """
        pass
    
    @abstractmethod
    def get_usage(self, client_id: str) -> int:
        """
        Get current usage for a client.
        
        This method retrieves the current request count for a specific client.
        It's useful for monitoring and analytics purposes.
        
        Args:
            client_id: Unique identifier for the client
            
        Returns:
            Current usage count
        """
        pass
    
    @abstractmethod
    def reset_usage(self, client_id: str) -> None:
        """
        Reset usage for a client.
        
        This method resets the usage counter for a specific client to zero.
        It's typically used for administrative purposes or at the beginning
        of a new billing cycle.
        
        Args:
            client_id: Unique identifier for the client
        """
        pass


class InMemoryQuotaStore(BaseQuotaStore):
    """
    In-memory implementation of quota store.
    
    This implementation stores quota information in application memory.
    It's suitable for proof-of-concept and testing environments but not
    recommended for production due to lack of persistence across restarts.
    
    For production deployments, this can be easily swapped with a Redis
    or database-backed implementation by changing a single line of code
    while maintaining the same interface.
    """
    
    def __init__(self):
        # Use limits from configuration instead of hardcoded values
        self._limits: Dict[str, int] = settings.quota_limits.copy()
        self._usage: Dict[str, int] = {k: 0 for k in self._limits.keys()}
        self._default_limit = settings.quota_default_limit
    
    def check_quota(self, client_id: str, increment: bool = True) -> bool:
        """
        Check if a client has exceeded their quota and optionally increment usage.
        
        Args:
            client_id: Unique identifier for the client
            increment: Whether to increment usage after checking
            
        Returns:
            True if quota is available, False if exceeded
        """
        current_usage = self._usage.get(client_id, 0)
        limit = self._limits.get(client_id, self._default_limit)  # Use configurable default
        
        if current_usage >= limit:
            return False
        
        if increment:
            self._usage[client_id] = current_usage + 1
        
        return True
    
    def get_usage(self, client_id: str) -> int:
        """
        Get current usage for a client.
        
        Args:
            client_id: Unique identifier for the client
            
        Returns:
            Current usage count
        """
        return self._usage.get(client_id, 0)
    
    def reset_usage(self, client_id: str) -> None:
        """
        Reset usage for a client.
        
        Args:
            client_id: Unique identifier for the client
        """
        self._usage[client_id] = 0


class RedisQuotaStore(BaseQuotaStore):
    """
    Redis implementation of quota store for production environments.
    
    This implementation uses Redis to store and track client quotas,
    providing persistence and horizontal scalability.
    Includes fail-open logic for production resilience.
    """
    
    def __init__(self):
        try:
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                decode_responses=True,
                socket_connect_timeout=2  # Short timeout for resilience
            )
            self._limits_key = "client_limits"
            self._default_limit = settings.quota_default_limit
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            self.redis_client = None

    def check_quota(self, client_id: str, increment: bool = True) -> bool:
        if self.redis_client is None:
            logger.warning("Redis client not initialized, failing open.")
            return True
            
        try:
            limit = self.redis_client.hget(self._limits_key, client_id)
            if limit is None:
                limit = self._default_limit
            else:
                limit = int(limit)
                
            usage_key = f"usage:{client_id}"
            current_usage = self.redis_client.get(usage_key)
            current_usage = int(current_usage) if current_usage else 0
            
            if current_usage >= limit:
                return False
                
            if increment:
                self.redis_client.incr(usage_key)
                
            return True
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Redis connection error in check_quota: {e}. Failing open.")
            return True
        except Exception as e:
            logger.error(f"Unexpected error in Redis check_quota: {e}. Failing open.")
            return True

    def get_usage(self, client_id: str) -> int:
        if self.redis_client is None:
            return 0
        try:
            usage_key = f"usage:{client_id}"
            usage = self.redis_client.get(usage_key)
            return int(usage) if usage else 0
        except Exception:
            return 0

    def reset_usage(self, client_id: str) -> None:
        if self.redis_client is None:
            return
        try:
            usage_key = f"usage:{client_id}"
            self.redis_client.delete(usage_key)
        except Exception as e:
            logger.error(f"Failed to reset usage in Redis: {e}")