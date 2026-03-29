"""Node functions for the multi-agent RAG graph."""

import re
import logging

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document

from .states import AgentState, SubAgentState
from .prompts import QUERY_ANALYZER, SYNTHESIZER, GENERATOR, REFLECTOR, SUMMARIZER

logger = logging.getLogger(__name__)

WINDOW_SIZE = 6


class QueryAnalysis(BaseModel):
    sub_queries: list[str] = Field(description="List of sub-queries, original query first")


class ReflectionResult(BaseModel):
    is_sufficient: bool = Field(description="Whether the answer is sufficient")
    retry_queries: list[str] = Field(default_factory=list, description="Queries for missing info")


def _build_context_header(state: AgentState) -> str:
    parts = []
    summary = state.get("summary", "")
    if summary:
        parts.append(f"<conversation_summary>\n{summary}\n</conversation_summary>")

    recent = state.get("messages", [])[-WINDOW_SIZE:]
    if recent:
        lines = []
        for msg in recent:
            if isinstance(msg, HumanMessage):
                role = "User"
            elif isinstance(msg, SystemMessage):
                continue
            else:
                role = "Assistant"
            lines.append(f"{role}: {msg.content}")
        if lines:
            parts.append("<recent_conversation>\n" + "\n".join(lines) + "\n</recent_conversation>")

    return "\n\n".join(parts)


def _remap_citations(answer: str, offset: int) -> str:
    def _replace(m):
        return f"[{int(m.group(1)) + offset}]"
    return re.sub(r"\[(\d+)\]", _replace, answer)


# --------------- Top-level agent nodes ---------------

def analyze_query(state: AgentState, llm: BaseChatModel) -> dict:
    query = state["query"]
    context_header = _build_context_header(state)

    msgs = [SystemMessage(content=QUERY_ANALYZER)]
    if context_header:
        msgs.append(SystemMessage(content=f"# Conversation Context\n{context_header}"))
    msgs.append(HumanMessage(content=query))

    structured_llm = llm.with_structured_output(QueryAnalysis)
    try:
        result: QueryAnalysis = structured_llm.invoke(msgs)
        sub_queries = result.sub_queries
    except Exception:
        sub_queries = [query]

    if query not in sub_queries:
        sub_queries.insert(0, query)

    logger.info(f"Decomposed into {len(sub_queries)} sub-queries")
    return {"sub_queries": sub_queries}


def synthesize(state: AgentState, llm: BaseChatModel) -> dict:
    query = state["query"]
    sub_answers = state.get("sub_answers", [])
    context_header = _build_context_header(state)

    context_parts = []
    all_citations = []
    global_idx = 0
    for sa in sub_answers:
        remapped = _remap_citations(sa["answer"], global_idx)
        context_parts.append(f"Q: {sa['query']}\nA: {remapped}")
        sa_citations = sa.get("citations", [])
        all_citations.extend(sa_citations)
        global_idx += max(len(sa_citations), 1)

    sub_context = "\n\n".join(context_parts)

    msgs = [SystemMessage(content=SYNTHESIZER.format(context=sub_context))]
    if context_header:
        msgs.append(SystemMessage(content=f"# Conversation Context\n{context_header}"))
    msgs.append(HumanMessage(content=f"Original question: {query}"))

    resp = llm.invoke(msgs)

    return {
        "answer": resp.content,
        "citations": all_citations,
        "messages": [HumanMessage(content=query), resp],
    }


def summarize_conversation(state: AgentState, llm: BaseChatModel) -> dict:
    messages = state.get("messages", [])
    existing_summary = state.get("summary", "")

    if len(messages) <= WINDOW_SIZE:
        return {}

    to_summarize = messages[:-WINDOW_SIZE]

    lines = []
    if existing_summary:
        lines.append(f"Previous summary: {existing_summary}")
    for msg in to_summarize:
        if isinstance(msg, HumanMessage):
            role = "User"
        elif isinstance(msg, SystemMessage):
            continue
        else:
            role = "Assistant"
        lines.append(f"{role}: {msg.content}")

    if not lines:
        return {}

    history = "\n".join(lines)

    resp = llm.invoke([
        SystemMessage(content=SUMMARIZER.format(history=history)),
        HumanMessage(content="Summarize the above conversation."),
    ])

    return {
        "summary": resp.content,
        "messages": [RemoveMessage(id=m.id) for m in to_summarize],
    }


# --------------- Sub-agent nodes ---------------

def retrieve(state: SubAgentState, retriever, citation_extractor) -> dict:
    query = state["query"]

    docs: list[Document] = retriever.invoke(query)

    citations = citation_extractor.extract_all(docs) if docs else []
    documents = []
    for doc, cite in zip(docs, citations):
        source = citation_extractor.format_citation(cite)
        documents.append(f"{doc.page_content}\n[Source: {source}]")

    if len(docs) != len(citations):
        logger.warning(f"Doc/citation count mismatch: {len(docs)} docs vs {len(citations)} citations")

    truncated = query[:50] + ("..." if len(query) > 50 else "")
    logger.info(f"Retrieved {len(documents)} docs for: {truncated}")
    return {"documents": documents, "citations": citations}


def generate(state: SubAgentState, llm: BaseChatModel) -> dict:
    query = state["query"]
    documents = state.get("documents", [])

    if not documents:
        return {"answer": "No relevant information found.", "citations": []}

    context = "\n\n".join(f"[{i}] {d}" for i, d in enumerate(documents, 1))

    resp = llm.invoke([
        SystemMessage(content=GENERATOR),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
    ])

    return {"answer": resp.content}


def reflect(state: SubAgentState, llm: BaseChatModel) -> dict:
    query = state["query"]
    answer = state.get("answer", "")
    documents = state.get("documents", [])
    retries = state.get("retries", 0)

    if not documents:
        return {"is_sufficient": True, "retry_queries": [], "retries": retries + 1}

    structured_llm = llm.with_structured_output(ReflectionResult)
    try:
        result: ReflectionResult = structured_llm.invoke([
            SystemMessage(content=REFLECTOR),
            HumanMessage(content=f"Question: {query}\nAnswer: {answer}"),
        ])
        is_sufficient = result.is_sufficient
        retry_queries = result.retry_queries
    except Exception:
        is_sufficient = True
        retry_queries = []

    return {
        "is_sufficient": is_sufficient,
        "retry_queries": retry_queries,
        "retries": retries + 1,
    }


# --------------- Routing functions ---------------

def should_retry(state: SubAgentState, max_retries: int = 2) -> str:
    if not state.get("is_sufficient", True) and state.get("retries", 0) < max_retries:
        logger.info(f"Reflection: insufficient, retrying ({state.get('retries', 0)}/{max_retries})")
        return "retry"
    return "done"


def prepare_retry(state: SubAgentState) -> dict:
    retry_queries = state.get("retry_queries", [])
    query = retry_queries[-1] if retry_queries else state["query"]
    return {"query": query}
