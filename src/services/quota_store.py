"""
Module for managing client quotas in the AI inference router.

This module implements a flexible quota management system with an abstract base class
that allows for different storage backends. The design supports easy swapping of
implementations (e.g., from in-memory to Redis) without changing the business logic.
"""

from abc import ABC, abstractmethod
from typing import Dict
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
        # Default limits and usage for demonstration
        # In a real application, these would come from a configuration source
        self._limits: Dict[str, int] = {"client_001": 100, "client_002": 5}
        self._usage: Dict[str, int] = {"client_001": 0, "client_002": 0}
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