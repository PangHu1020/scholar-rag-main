"""Retrieval module: hybrid search + rerank + parent-child recall + optional HyDE."""

from typing import Optional
from langchain_core.documents import Document
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import Function, FunctionType
from sentence_transformers import CrossEncoder


class Retriever:
    """Hybrid retriever with BM25+dense fusion, reranking, and parent-child recall."""

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        milvus_uri: str = "http://localhost:19530",
        collection_name: str = "papers",
        llm: Optional[object] = None,
    ):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.reranker = CrossEncoder(reranker_model)
        self.milvus_uri = milvus_uri
        self.collection_name = collection_name
        self.llm = llm

        bm25 = BM25BuiltInFunction(input_field_names="text", output_field_names="sparse")
        conn = {"uri": self.milvus_uri}

        self._child_store = Milvus(
            self.embeddings,
            builtin_function=bm25,
            vector_field=["dense", "sparse"],
            collection_name=f"{self.collection_name}_children",
            connection_args=conn,
        )
        self._parent_store = Milvus(
            self.embeddings,
            builtin_function=bm25,
            vector_field=["dense", "sparse"],
            collection_name=f"{self.collection_name}_parents",
            connection_args=conn,
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
        search_query = self._hyde(query) if use_hyde and self.llm else query

        if rerank and self.reranker:
            children = self._hybrid_search(self._child_store, search_query, fetch_k * 2, rrf_k)
            if not children:
                return []
            children = self._rerank(query, children, fetch_k)
        else:
            children = self._hybrid_search(self._child_store, search_query, fetch_k, rrf_k)
            if not children:
                return []

        if expand_parent:
            results = self._expand_to_parents(children[:k * 2])
        else:
            results = children[:k * 2]

        seen = set()
        deduped = []
        for doc in results:
            cid = doc.metadata.get("chunk_id", id(doc))
            if cid not in seen:
                seen.add(cid)
                deduped.append(doc)
        return deduped[:k]

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
            expr = f'chunk_id == "{pid}"'
            hits = self._parent_store.similarity_search("dummy", k=1, expr=expr)
            parents.extend(hits)

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
