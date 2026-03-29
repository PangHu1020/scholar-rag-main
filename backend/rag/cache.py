"""Retrieval result caching."""

import hashlib
from collections import OrderedDict
from typing import List, Optional
from langchain_core.documents import Document


class RetrievalCache:
    """LRU cache for retrieval results."""
    
    def __init__(self, max_size: int = 1000):
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def _make_key(self, query: str, k: int, rerank: bool, expand_parent: bool) -> str:
        """Generate cache key from query parameters."""
        params = f"{query}|{k}|{rerank}|{expand_parent}"
        return hashlib.md5(params.encode()).hexdigest()
    
    def get(self, query: str, k: int, rerank: bool, expand_parent: bool) -> Optional[List[Document]]:
        """Get cached results."""
        key = self._make_key(query, k, rerank, expand_parent)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, query: str, k: int, rerank: bool, expand_parent: bool, results: List[Document]):
        """Cache results with LRU eviction."""
        key = self._make_key(query, k, rerank, expand_parent)
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = results
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def clear(self):
        """Clear all cached results."""
        self.cache.clear()
