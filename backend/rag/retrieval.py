"""Retrieval module: hybrid search + rerank + parent-child recall + optional HyDE."""

import logging
from typing import Optional
from langchain_core.documents import Document
from langchain_milvus import Milvus
from pymilvus import Function, FunctionType
from .cache import RetrievalCache
from .factory import EmbeddingService, RerankerService, MilvusStoreFactory

logger = logging.getLogger(__name__)

RERANK_FETCH_MULTIPLIER = 2
DEDUP_FETCH_MULTIPLIER = 2


class Retriever:
    """Hybrid retriever with BM25+dense fusion, reranking, and parent-child recall."""

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        milvus_uri: str = "http://localhost:19530",
        collection_name: str = "papers",
        llm: Optional[object] = None,
        enable_cache: bool = True,
        child_store: Optional[Milvus] = None,
        parent_store: Optional[Milvus] = None,
    ):
        self.embeddings = EmbeddingService.get_embeddings(embedding_model)
        self.reranker = RerankerService.get_reranker(reranker_model)
        self.milvus_uri = milvus_uri
        self.collection_name = collection_name
        self.llm = llm
        self.cache = RetrievalCache() if enable_cache else None

        self._child_store = child_store or MilvusStoreFactory.create_store(
            self.embeddings, milvus_uri, collection_name, is_child=True
        )
        self._parent_store = parent_store or MilvusStoreFactory.create_store(
            self.embeddings, milvus_uri, collection_name, is_child=False
        )

    def retrieve(
        self,
        query: str,
        k: int = 5,
        use_hyde: bool = False,
        rerank: bool = True,
        expand_parent: bool = True,
        rrf_k: int = 60,
        fetch_k: int = 20,
    ) -> list[Document]:
        """Full retrieval pipeline.

        Args:
            query: User query string.
            k: Number of final results.
            use_hyde: Whether to expand query with HyDE before search.
            rerank: Whether to rerank results with CrossEncoder.
            expand_parent: Whether to expand child hits to parent chunks.
            rrf_k: RRF constant for hybrid fusion.
            fetch_k: Number of candidates to fetch before reranking.
        """
        if self.cache:
            cached = self.cache.get(query, k, rerank, expand_parent)
            if cached is not None:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached
        
        search_query = self._hyde(query) if use_hyde and self.llm else query

        if rerank and self.reranker:
            children = self._hybrid_search(self._child_store, search_query, fetch_k * RERANK_FETCH_MULTIPLIER, rrf_k)
            if not children:
                logger.warning(f"No results found for query: {query[:50]}...")
                return []
            children = self._rerank(query, children, fetch_k)
        else:
            children = self._hybrid_search(self._child_store, search_query, fetch_k, rrf_k)
            if not children:
                logger.warning(f"No results found for query: {query[:50]}...")
                return []

        if expand_parent:
            results = self._expand_to_parents(children[:k * DEDUP_FETCH_MULTIPLIER])
        else:
            results = children[:k * DEDUP_FETCH_MULTIPLIER]

        seen = set()
        deduped = []
        for doc in results:
            cid = doc.metadata.get("chunk_id", id(doc))
            if cid not in seen:
                seen.add(cid)
                deduped.append(doc)
        
        final = deduped[:k]
        
        if self.cache:
            self.cache.put(query, k, rerank, expand_parent, final)
        
        logger.info(f"Retrieved {len(final)} results for query: {query[:50]}...")
        return final
    
    def get_updater(self):
        """Get IncrementalUpdater for this retriever's stores."""
        from .incremental import IncrementalUpdater
        return IncrementalUpdater(self._parent_store, self._child_store)

    def _hybrid_search(
        self, store: Milvus, query: str, k: int, rrf_k: int
    ) -> list[Document]:
        reranker = Function(
            name="rrf_reranker",
            function_type=FunctionType.RERANK,
            input_field_names=["dense", "sparse"],
            params={"k": rrf_k},
        )
        results = store.similarity_search(
            query, k=k, reranker=reranker, fetch_k=k
        )
        return results

    def _rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        if not docs:
            return docs
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]

    def _expand_to_parents(self, children: list[Document]) -> list[Document]:
        parent_ids = list(dict.fromkeys(
            doc.metadata.get("chunk_parent_id")
            for doc in children
            if doc.metadata.get("chunk_parent_id")
        ))

        if not parent_ids:
            return children

        parents = []
        for pid in parent_ids:
            try:
                expr = f'chunk_id == "{pid}"'
                hits = self._parent_store.similarity_search("dummy", k=1, expr=expr)
                parents.extend(hits)
            except Exception as e:
                logger.error(f"Failed to fetch parent {pid}: {e}")

        return parents if parents else children

    def _hyde(self, query: str) -> str:
        prompt = (
            "Please write a short passage from an academic paper that would answer "
            f"the following question. Do not explain, just write the passage.\n\n"
            f"Question: {query}\n\nPassage:"
        )
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            return content.strip()
        except Exception:
            return query
