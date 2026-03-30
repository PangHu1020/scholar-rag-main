"""Graph assembly for the multi-agent RAG system."""

from typing import Optional

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Send

from .states import AgentState, SubAgentState, SubAnswer
from .nodes import (
    analyze_query,
    prepare_synthesis,
    summarize_conversation,
    retrieve,
    generate,
    reflect,
    should_retry,
    prepare_retry,
)


def _collect_sub_answer(state: SubAgentState) -> dict:
    return {
        "sub_answers": [SubAnswer(
            query=state["query"],
            answer=state.get("answer", ""),
            citations=state.get("citations", []),
        )]
    }


def _build_sub_agent_graph(
    llm: BaseChatModel,
    retriever,
    citation_extractor,
    max_retries: int,
) -> StateGraph:
    sg = StateGraph(SubAgentState)

    async def retrieve_node(state: SubAgentState) -> dict:
        return await retrieve(state, retriever=retriever, citation_extractor=citation_extractor)

    async def generate_node(state: SubAgentState) -> dict:
        return await generate(state, llm=llm)

    async def reflect_node(state: SubAgentState) -> dict:
        return await reflect(state, llm=llm)

    def retry_router(state: SubAgentState) -> str:
        return should_retry(state, max_retries=max_retries)

    sg.add_node("retrieve", retrieve_node)
    sg.add_node("generate", generate_node)
    sg.add_node("reflect", reflect_node)
    sg.add_node("prepare_retry", prepare_retry)

    sg.add_edge(START, "retrieve")
    sg.add_edge("retrieve", "generate")
    sg.add_edge("generate", "reflect")
    sg.add_conditional_edges("reflect", retry_router, {
        "retry": "prepare_retry",
        "done": END,
    })
    sg.add_edge("prepare_retry", "retrieve")

    return sg


def build_graph(
    llm: BaseChatModel,
    retriever,
    citation_extractor,
    max_retries: int = 2,
    checkpointer: Optional[BaseCheckpointSaver] = None,
):
    sub_graph = _build_sub_agent_graph(llm, retriever, citation_extractor, max_retries).compile()

    def dispatch(state: AgentState):
        return [Send("sub_agent", {"query": q}) for q in state["sub_queries"]]

    async def sub_agent_node(state: dict) -> dict:
        sub_input = {
            "query": state["query"],
            "documents": [],
            "answer": "",
            "citations": [],
            "is_sufficient": False,
            "retry_queries": [],
            "retries": 0,
        }
        result = await sub_graph.ainvoke(sub_input)
        return _collect_sub_answer(result)

    async def summarize_node(state: AgentState) -> dict:
        return await summarize_conversation(state, llm=llm)

    async def analyze_node(state: AgentState) -> dict:
        return await analyze_query(state, llm=llm)

    graph = StateGraph(AgentState)

    graph.add_node("summarize", summarize_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("sub_agent", sub_agent_node)
    graph.add_node("prepare_synthesis", prepare_synthesis)

    graph.add_edge(START, "summarize")
    graph.add_edge("summarize", "analyze")
    graph.add_conditional_edges("analyze", dispatch, ["sub_agent"])
    graph.add_edge("sub_agent", "prepare_synthesis")
    graph.add_edge("prepare_synthesis", END)

    return graph.compile(checkpointer=checkpointer)
