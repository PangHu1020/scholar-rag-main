"""Embedding-based retrieval cache.

Instead of exact key matching, caches query embeddings and returns results
for semantically similar queries above a cosine similarity threshold.
"""

import logging
from collections import OrderedDict
from typing import Optional
import numpy as np
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


class RetrievalCache:
    """LRU cache keyed by query embedding similarity.

    On get(): computes the query embedding, finds the most similar cached
    embedding; if cosine similarity >= threshold, returns cached results.

    On put(): stores the embedding alongside results; evicts LRU when full.
    """

    def __init__(
        self,
        embeddings,
        max_size: int = 1000,
        similarity_threshold: float = 0.95,
    ):
        """
        Args:
            embeddings: HuggingFaceEmbeddings instance (shared singleton).
            max_size: Max number of cached queries.
            similarity_threshold: Cosine similarity cutoff for a cache hit.
        """
        self.embeddings = embeddings
        self.max_size = max_size
        self.threshold = similarity_threshold
        # OrderedDict: key -> (embedding np.ndarray, results list)
        self._store: OrderedDict[str, tuple[np.ndarray, list[Document]]] = OrderedDict()

    def _embed(self, query: str) -> np.ndarray:
        vec = self.embeddings.embed_query(query)
        return np.array(vec, dtype=np.float32)

    def _find_best(self, vec: np.ndarray) -> Optional[tuple[str, float]]:
        """Return (key, similarity) of the best matching cache entry, or None."""
        best_key, best_sim = None, -1.0
        for key, (cached_vec, _) in self._store.items():
            sim = _cosine(vec, cached_vec)
            if sim > best_sim:
                best_sim, best_key = sim, key
        if best_key is not None and best_sim >= self.threshold:
            return best_key, best_sim
        return None

    def get(self, query: str) -> Optional[list[Document]]:
        """Return cached results if a similar query exists, else None."""
        if not self._store:
            return None
        vec = self._embed(query)
        match = self._find_best(vec)
        if match:
            key, sim = match
            self._store.move_to_end(key)
            logger.debug(f"Cache hit (sim={sim:.3f}): {query[:60]}")
            return self._store[key][1]
        return None

    def put(self, query: str, results: list[Document]) -> None:
        """Cache results for query; evict LRU entry if at capacity."""
        vec = self._embed(query)
        # use query string as key (unique enough, embedding stored separately)
        if query in self._store:
            self._store.move_to_end(query)
        self._store[query] = (vec, results)
        if len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()
