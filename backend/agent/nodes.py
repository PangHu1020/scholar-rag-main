"""Node functions for the multi-agent RAG graph."""

import re
import logging

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document

from .states import AgentState, SubAgentState
from .prompts import QUERY_ANALYZER, QUERY_CLASSIFIER, SYNTHESIZER, GENERATOR, REFLECTOR, SUMMARIZER

logger = logging.getLogger(__name__)

WINDOW_SIZE = 6


class QueryAnalysis(BaseModel):
    sub_queries: list[str] = Field(description="List of sub-queries, original query first")


class QueryClassification(BaseModel):
    query_type: str = Field(description="One of: experimental_result, method, background, general")


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
            elif isinstance(msg, AIMessage):
                role = "Assistant"
            else:
                continue
            lines.append(f"{role}: {msg.content}")
        if lines:
            parts.append("<recent_conversation>\n" + "\n".join(lines) + "\n</recent_conversation>")

    return "\n\n".join(parts)


def _remap_citations(answer: str, offset: int) -> str:
    def _replace(m):
        return f"[{int(m.group(1)) + offset}]"
    return re.sub(r"\[(\d+)\]", _replace, answer)


# --------------- Top-level agent nodes ---------------

async def analyze_query(state: AgentState, llm: BaseChatModel) -> dict:
    query = state["query"]
    context_header = _build_context_header(state)

    system_content = QUERY_ANALYZER
    if context_header:
        system_content += f"\n\n# Conversation Context\n{context_header}"
    msgs = [SystemMessage(content=system_content), HumanMessage(content=query)]

    structured_llm = llm.with_structured_output(QueryAnalysis)
    try:
        result: QueryAnalysis = await structured_llm.ainvoke(msgs)
        sub_queries = result.sub_queries
    except Exception:
        sub_queries = [query]

    if query not in sub_queries:
        sub_queries.insert(0, query)

    logger.info(f"Decomposed into {len(sub_queries)} sub-queries")
    return {"sub_queries": sub_queries}


async def classify_query(state: AgentState, llm: BaseChatModel) -> dict:
    from .tools import set_query_type
    query = state["query"]

    structured_llm = llm.with_structured_output(QueryClassification)
    try:
        result: QueryClassification = await structured_llm.ainvoke([
            SystemMessage(content=QUERY_CLASSIFIER),
            HumanMessage(content=query),
        ])
        query_type = result.query_type
        if query_type not in ("experimental_result", "method", "background", "general"):
            query_type = "general"
    except Exception:
        query_type = "general"

    set_query_type(query_type)
    logger.info(f"Query classified as: {query_type}")
    return {"query_type": query_type}


def prepare_synthesis(state: AgentState) -> dict:
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

    system_content = SYNTHESIZER.format(context=sub_context)
    if context_header:
        system_content += f"\n\n# Conversation Context\n{context_header}"

    return {
        "synth_messages": [
            SystemMessage(content=system_content),
            HumanMessage(content=f"Original question: {state['query']}"),
        ],
        "citations": all_citations,
    }


async def summarize_conversation(state: AgentState, llm: BaseChatModel) -> dict:
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
        elif isinstance(msg, AIMessage):
            role = "Assistant"
        else:
            continue
        lines.append(f"{role}: {msg.content}")

    if not lines:
        return {}

    history = "\n".join(lines)

    resp = await llm.ainvoke([
        SystemMessage(content=SUMMARIZER.format(history=history)),
        HumanMessage(content="Summarize the above conversation."),
    ])

    return {
        "summary": resp.content,
        "messages": [RemoveMessage(id=m.id) for m in to_summarize],
    }


# --------------- Sub-agent nodes ---------------

async def retrieve(state: SubAgentState, retriever, citation_extractor) -> dict:
    from rag.factory import is_visual_query
    query = state["query"]
    query_type = state.get("query_type", "general")

    _ROUTE_CONFIG: dict[str, list[str] | None] = {
        "experimental_result": ["experiment"],
        "method": ["method"],
        "background": ["background"],
        "general": None,
    }
    section_type_filter = _ROUTE_CONFIG.get(query_type)

    docs: list[Document] = retriever.invoke(query, section_type_filter=section_type_filter)

    if not docs and section_type_filter:
        logger.info(f"No results with section filter, retrying without filter")
        docs = retriever.invoke(query, section_type_filter=None)

    citations = citation_extractor.extract_all(docs) if docs else []
    documents = []
    for doc, cite in zip(docs, citations):
        source = citation_extractor.format_citation(cite)
        documents.append(f"{doc.page_content}\n[Source: {source}]")

    if len(docs) != len(citations):
        logger.warning(f"Doc/citation count mismatch: {len(docs)} docs vs {len(citations)} citations")

    # Determine if VLM should be invoked in generate step
    has_figure = any(
        c.get("node_type") == "figure" and c.get("metadata", {}).get("image_path")
        for c in citations
    )
    needs_vlm = is_visual_query(query) and has_figure

    truncated = query[:50] + ("..." if len(query) > 50 else "")
    logger.info(f"Retrieved {len(documents)} docs for: {truncated} | needs_vlm={needs_vlm}")
    return {"documents": documents, "citations": citations, "needs_vlm": needs_vlm}


async def generate(state: SubAgentState, llm: BaseChatModel, vision_service=None) -> dict:
    query = state["query"]
    documents = state.get("documents", [])
    citations = state.get("citations", [])
    needs_vlm = state.get("needs_vlm", False)

    if not documents:
        return {"answer": "No relevant information found.", "citations": []}

    # VLM enhancement: inject figure descriptions before generation
    if needs_vlm and vision_service:
        from rag.factory import should_invoke_vlm
        has_figure = any(
            c.get("node_type") == "figure" and c.get("metadata", {}).get("image_path")
            for c in citations
        )
        if should_invoke_vlm(query, has_figure):
            for cite in citations:
                if cite.get("node_type") != "figure":
                    continue
                image_path = cite.get("metadata", {}).get("image_path")
                if not image_path:
                    continue
                vlm_desc = cite.get("metadata", {}).get("vlm_description")
                if not vlm_desc:
                    caption = cite.get("text", "")
                    vlm_desc = vision_service.analyze_figure(image_path, caption)
                    if vlm_desc:
                        cite.setdefault("metadata", {})["vlm_description"] = vlm_desc
                if vlm_desc:
                    documents = list(documents) + [f"[Figure Analysis] {vlm_desc}"]
                    logger.info(f"VLM analysis injected for figure: {image_path}")

    context = "\n\n".join(f"[{i}] {d}" for i, d in enumerate(documents, 1))

    resp = await llm.ainvoke([
        SystemMessage(content=GENERATOR),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
    ])

    return {"answer": resp.content}


async def reflect(state: SubAgentState, llm: BaseChatModel, vision_service=None) -> dict:
    query = state["query"]
    answer = state.get("answer", "")
    documents = state.get("documents", [])
    citations = state.get("citations", [])
    retries = state.get("retries", 0)

    if not documents:
        return {"is_sufficient": True, "retry_queries": [], "retries": retries + 1}

    structured_llm = llm.with_structured_output(ReflectionResult)
    try:
        result: ReflectionResult = await structured_llm.ainvoke([
            SystemMessage(content=REFLECTOR),
            HumanMessage(content=f"Question: {query}\nAnswer: {answer}"),
        ])
        is_sufficient = result.is_sufficient
        retry_queries = result.retry_queries
    except Exception:
        is_sufficient = True
        retry_queries = []

    # VLM fallback: answer insufficient + has figures + VLM not yet used
    if not is_sufficient and vision_service and not state.get("needs_vlm", False):
        from rag.factory import should_invoke_vlm
        figure_citations = [
            c for c in citations
            if c.get("node_type") == "figure" and c.get("metadata", {}).get("image_path")
        ]
        has_figure = bool(figure_citations)
        if should_invoke_vlm(query, has_figure, answer=answer):
            # Inject VLM descriptions and re-generate
            extra_docs = []
            for cite in figure_citations[:2]:  # cap at 2 to control cost
                image_path = cite["metadata"]["image_path"]
                vlm_desc = cite["metadata"].get("vlm_description")
                if not vlm_desc:
                    caption = cite.get("text", "")
                    vlm_desc = vision_service.analyze_figure(image_path, caption)
                    if vlm_desc:
                        cite["metadata"]["vlm_description"] = vlm_desc
                if vlm_desc:
                    extra_docs.append(f"[Figure Analysis] {vlm_desc}")
                    logger.info(f"VLM fallback triggered for: {image_path}")

            if extra_docs:
                enhanced_docs = list(documents) + extra_docs
                context = "\n\n".join(f"[{i}] {d}" for i, d in enumerate(enhanced_docs, 1))
                resp = await llm.ainvoke([
                    SystemMessage(content=GENERATOR),
                    HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
                ])
                return {
                    "answer": resp.content,
                    "documents": enhanced_docs,
                    "needs_vlm": True,   # mark as used to prevent re-trigger
                    "is_sufficient": True,
                    "retry_queries": [],
                    "retries": retries + 1,
                }

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
