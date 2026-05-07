"""
Unified Web Cache with TTL support.

Two global instances:
- search_cache: Caches web search results (query -> results)
- content_cache: Caches fetched page content (url -> content)

Default TTL: 10 hours (configurable via constructor)
"""

import threading
import time
from typing import Any

DEFAULT_CACHE_TTL_HOURS = 10.0


class WebCache:
    """Thread-safe cache with TTL expiration.

    Used for caching web search results and fetched page content.
    Entries automatically expire after TTL.
    """

    def __init__(self, ttl_hours: float = DEFAULT_CACHE_TTL_HOURS, name: str = "cache"):
        self._cache: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.ttl_seconds = ttl_hours * 3600
        self.name = name

    def get(self, key: str) -> Any | None:
        """Get value if exists and not expired."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if time.time() - entry["timestamp"] > self.ttl_seconds:
                del self._cache[key]
                return None
            return entry["value"]

    def set(self, key: str, value: Any) -> None:
        """Set value with current timestamp."""
        with self._lock:
            self._cache[key] = {
                "value": value,
                "timestamp": time.time(),
            }

    def has(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        return self.get(key) is not None

    def clear(self) -> int:
        """Clear all entries. Returns count of entries cleared."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count of entries removed."""
        with self._lock:
            now = time.time()
            expired = [k for k, v in self._cache.items() if now - v["timestamp"] > self.ttl_seconds]
            for k in expired:
                del self._cache[k]
            return len(expired)

    def __len__(self) -> int:
        """Return count of entries (including possibly expired)."""
        return len(self._cache)

    def stats(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            now = time.time()
            valid = sum(1 for v in self._cache.values() if now - v["timestamp"] <= self.ttl_seconds)
            return {
                "name": self.name,
                "total_entries": len(self._cache),
                "valid_entries": valid,
                "expired_entries": len(self._cache) - valid,
                "ttl_hours": self.ttl_seconds / 3600,
            }


# =============================================================================
# Global cache instances
# =============================================================================

search_cache = WebCache(ttl_hours=DEFAULT_CACHE_TTL_HOURS, name="search_cache")
"""Global cache for web search results. Key: 'query|max_results'"""

content_cache = WebCache(ttl_hours=DEFAULT_CACHE_TTL_HOURS, name="content_cache")
"""Global cache for fetched page content. Key: url"""


# =============================================================================
# Helper functions
# =============================================================================


def get_search_cache_key(query: str, max_results: int) -> str:
    """Generate cache key for search query."""
    return f"{query}|{max_results}"


def get_cached_search(query: str, max_results: int) -> dict | None:
    """Get cached search result if available and not expired."""
    key = get_search_cache_key(query, max_results)
    return search_cache.get(key)


def cache_search_result(query: str, max_results: int, result: dict) -> None:
    """Cache a search result."""
    key = get_search_cache_key(query, max_results)
    search_cache.set(key, result)


def get_cached_content(url: str) -> str | None:
    """Get cached page content if available and not expired."""
    return content_cache.get(url)


def cache_content(url: str, content: str) -> None:
    """Cache fetched page content."""
    content_cache.set(url, content)


def has_cached_content(url: str) -> bool:
    """Check if URL content is cached and not expired."""
    return content_cache.has(url)


def clear_all_caches() -> dict:
    """Clear both caches. Returns counts of entries cleared."""
    return {
        "search_cleared": search_cache.clear(),
        "content_cleared": content_cache.clear(),
    }
