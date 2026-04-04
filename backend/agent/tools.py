"""Tools for LangGraph agent."""

import sys
from pathlib import Path
from typing import Optional
from contextvars import ContextVar

from langchain_core.tools import tool

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from rag.retrieval import Retriever
from rag.citation import CitationExtractor

_retriever: Optional[Retriever] = None
_query_type_ctx: ContextVar[str] = ContextVar("query_type", default="general")

_ROUTE_CONFIG: dict[str, list[str] | None] = {
    "experimental_result": ["experiment"],
    "method": ["method"],
    "background": ["background"],
    "general": None,
}


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever(
            embedding_model=Config.EMBEDDING_MODEL,
            reranker_model=Config.RERANKER_MODEL,
            milvus_uri=Config.MILVUS_URI,
            collection_name=Config.COLLECTION_NAME,
            enable_cache=Config.ENABLE_CACHE,
        )
    return _retriever


def set_query_type(query_type: str):
    _query_type_ctx.set(query_type)


@tool
def paper_retrieval(query: str) -> str:
    """Search the academic paper knowledge base and return relevant text chunks with citation info.

    Use this tool when you need to find information from indexed papers to answer
    research-related questions. Returns the most relevant passages along with
    their source metadata (paper, section, page).
    """
    retriever = get_retriever()
    query_type = _query_type_ctx.get()
    section_type_filter = _ROUTE_CONFIG.get(query_type)

    docs = retriever.retrieve(
        query=query,
        k=Config.TOP_K,
        use_hyde=False,
        rerank=True,
        expand_parent=True,
        rrf_k=Config.RRF_K,
        fetch_k=Config.FETCH_K,
        section_type_filter=section_type_filter,
    )

    if not docs:
        return "No relevant documents found for the given query."

    citations = CitationExtractor.extract_all(docs)
    parts = []
    for i, (doc, cite) in enumerate(zip(docs, citations), 1):
        source = CitationExtractor.format_citation(cite)
        parts.append(f"[{i}] {doc.page_content}\n    Source: {source}")

    return "\n\n".join(parts)
