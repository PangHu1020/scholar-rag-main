"""Chat API — SSE streaming with session management."""

import json
import uuid
import logging
from typing import AsyncGenerator

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

from config import Config
from app.dependencies import get_llm, get_retriever_tool, get_checkpointer
from app.store import create_session, get_session, update_session
from agent.graph import build_graph
from rag.citation import CitationExtractor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None


def _build_graph():
    return build_graph(
        llm=get_llm(),
        retriever=get_retriever_tool(),
        citation_extractor=CitationExtractor,
        max_retries=Config.MAX_RETRIES,
        checkpointer=get_checkpointer(),
    )


async def _stream_response(graph, query: str, session_id: str) -> AsyncGenerator[str, None]:
    config = {"configurable": {"thread_id": session_id}}
    graph_input = {"query": query}

    yield json.dumps({"type": "session_id", "data": session_id})
    yield json.dumps({"type": "status", "data": "analyzing"})

    try:
        synth_msgs = []
        final_citations = []

        async for chunk in graph.astream(graph_input, config=config, stream_mode="updates"):
            for node_name, node_output in chunk.items():
                if node_name == "analyze":
                    sq = node_output.get("sub_queries", [])
                    if sq:
                        yield json.dumps({"type": "sub_queries", "data": sq})
                        yield json.dumps({"type": "status", "data": "searching"})

                if node_name == "prepare_synthesis":
                    synth_msgs = node_output.get("synth_messages", [])
                    final_citations = node_output.get("citations", [])
                    logger.info(f"prepare_synthesis: {len(final_citations)} citations")

        llm = get_llm()
        answer_buf = ""
        if synth_msgs:
            async for token in llm.astream(synth_msgs):
                if token.content:
                    answer_buf += token.content
                    yield json.dumps({"type": "answer", "data": answer_buf})

        if not answer_buf:
            yield json.dumps({"type": "answer", "data": ""})

        await graph.aupdate_state(config, {
            "messages": [
                HumanMessage(content=query),
                AIMessage(content=answer_buf, additional_kwargs={"citations": final_citations}),
            ],
            "answer": answer_buf,
        })

        yield json.dumps({"type": "citations", "data": final_citations})

        title_hint = query[:50] + ("…" if len(query) > 50 else "")
        session = get_session(session_id)
        if session and not session.get("title"):
            update_session(session_id, title=title_hint)

        yield json.dumps({"type": "done", "data": None})

    except Exception as e:
        logger.exception("Chat error")
        yield json.dumps({"type": "error", "data": str(e)})


@router.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    if not get_session(session_id):
        create_session(session_id)

    graph = _build_graph()

    return EventSourceResponse(
        _stream_response(graph, req.query, session_id),
        media_type="text/event-stream",
    )
