"""Regime result cache with TTL.

Prevents excessive LLM API calls by caching regime classifications.
Cache keys are (asset, timeframe) tuples. Each entry has a TTL
after which it's considered stale and the LLM is re-queried.
"""

from __future__ import annotations

import time
import json
import os
from typing import Optional, Dict, Tuple
from llm_regime.schema import RegimeResult


class RegimeCache:
    """In-memory cache for regime results with optional file persistence.

    Parameters
    ----------
    default_ttl : seconds before a cached result is stale (default 30 min)
    persist_path : optional file path to persist cache across restarts
    """

    def __init__(self, default_ttl: int = 1800, persist_path: Optional[str] = None):
        self._cache: Dict[str, Tuple[RegimeResult, float]] = {}
        self._ttl = default_ttl
        self._persist_path = persist_path

        # Load from disk if available
        if persist_path and os.path.exists(persist_path):
            self._load()

    def get(self, asset: str, timeframe: str) -> Optional[RegimeResult]:
        """Get cached result if not stale."""
        key = f"{asset}:{timeframe}"
        if key not in self._cache:
            return None

        result, expires_at = self._cache[key]
        if time.time() > expires_at:
            del self._cache[key]
            return None

        return result

    def put(self, result: RegimeResult, ttl: Optional[int] = None):
        """Cache a regime result."""
        key = f"{result.asset}:{result.timeframe}"
        ttl = ttl or self._ttl
        self._cache[key] = (result, time.time() + ttl)

        if self._persist_path:
            self._save()

    def invalidate(self, asset: str, timeframe: str):
        """Force invalidate a cached entry."""
        key = f"{asset}:{timeframe}"
        self._cache.pop(key, None)

    def invalidate_all(self):
        """Clear all cached entries."""
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)

    def get_all_valid(self) -> Dict[str, RegimeResult]:
        """Get all non-stale cached results."""
        now = time.time()
        return {
            key: result
            for key, (result, expires_at) in self._cache.items()
            if now <= expires_at
        }

    def _save(self):
        """Persist cache to disk."""
        try:
            data = {}
            for key, (result, expires_at) in self._cache.items():
                data[key] = {
                    "result": result.to_dict(),
                    "expires_at": expires_at,
                }
            with open(self._persist_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # non-critical

    def _load(self):
        """Load cache from disk."""
        try:
            with open(self._persist_path) as f:
                data = json.load(f)
            now = time.time()
            for key, entry in data.items():
                if entry["expires_at"] > now:
                    result = RegimeResult.from_dict(entry["result"])
                    self._cache[key] = (result, entry["expires_at"])
        except Exception:
            pass  # non-critical
