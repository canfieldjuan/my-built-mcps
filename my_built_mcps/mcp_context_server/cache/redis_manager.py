"""
cache/redis_manager.py

Redis cache management and connection handling.
"""
import pickle
import logging
from typing import Optional, Any, Callable
import redis.asyncio as redis

from ..config.settings import settings
from ..interfaces import AbstractCache
from ..core.exceptions import CacheError

logger = logging.getLogger(__name__)


class RedisCache(AbstractCache):
    """Redis-based cache implementation."""
    
    def __init__(self, url: str = None):
        self.url = url or settings.REDIS_URL
        self._client: Optional[redis.Redis] = None
    
    async def connect(self) -> None:
        """Establish Redis connection."""
        try:
            self._client = await redis.from_url(self.url)
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise CacheError(f"Failed to connect to Redis: {e}")
    
    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
    
    @property
    def client(self) -> redis.Redis:
        """Get Redis client, raise if not connected."""
        if not self._client:
            raise CacheError("Redis client not connected")
        return self._client
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            cached = await self.client.get(key)
            if cached:
                return pickle.loads(cached)
            return None
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL."""
        try:
            serialized = pickle.dumps(value)
            if ttl:
                await self.client.setex(key, ttl, serialized)
            else:
                await self.client.set(key, serialized)
        except Exception as e:
            logger.warning(f"Cache set error for key {key}: {e}")
    
    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        try:
            await self.client.delete(key)
        except Exception as e:
            logger.warning(f"Cache delete error for key {key}: {e}")
    
    async def invalidate_pattern(self, pattern: str) -> None:
        """Invalidate all keys matching pattern."""
        try:
            async for key in self.client.scan_iter(match=pattern):
                await self.client.delete(key)
        except Exception as e:
            logger.warning(f"Cache invalidation error for pattern {pattern}: {e}")


class CacheManager:
    """Manages cache operations with session scoping."""
    
    def __init__(self, cache: AbstractCache, default_ttl: int = None):
        self.cache = cache
        self.default_ttl = default_ttl or settings.CACHE_TTL
    
    def _get_cache_key(self, session_id: str, key: str) -> str:
        """Get session-scoped cache key."""
        return f"{session_id}:{key}"
    
    async def get_cached_or_fetch(
        self,
        session_id: str,
        key: str,
        fetch_func: Callable,
        ttl: Optional[int] = None
    ) -> Any:
        """Get from cache or fetch from source."""
        cache_key = self._get_cache_key(session_id, key)
        
        # Try cache first
        cached = await self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Fetch from source
        result = await fetch_func()
        
        # Cache the result
        if result is not None:
            ttl = ttl or self.default_ttl
            await self.cache.set(cache_key, result, ttl)
        
        return result
    
    async def invalidate_session_cache(self, session_id: str, pattern: str = "*") -> None:
        """Invalidate cache entries for a session."""
        full_pattern = self._get_cache_key(session_id, pattern)
        await self.cache.invalidate_pattern(full_pattern)