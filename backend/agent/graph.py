"""Graph assembly for the multi-agent RAG system."""

from typing import Optional

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Send

from .states import AgentState, SubAgentState, SubAnswer
from .nodes import (
    analyze_query,
    synthesize,
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

    def retrieve_node(state: SubAgentState) -> dict:
        return retrieve(state, retriever=retriever, citation_extractor=citation_extractor)

    def generate_node(state: SubAgentState) -> dict:
        return generate(state, llm=llm)

    def reflect_node(state: SubAgentState) -> dict:
        return reflect(state, llm=llm)

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

    def sub_agent_node(state: dict) -> dict:
        sub_input = {
            "query": state["query"],
            "documents": [],
            "answer": "",
            "citations": [],
            "is_sufficient": False,
            "retry_queries": [],
            "retries": 0,
        }
        result = sub_graph.invoke(sub_input)
        return _collect_sub_answer(result)

    def summarize_node(state: AgentState) -> dict:
        return summarize_conversation(state, llm=llm)

    def analyze_node(state: AgentState) -> dict:
        return analyze_query(state, llm=llm)

    def synthesize_node(state: AgentState) -> dict:
        return synthesize(state, llm=llm)

    graph = StateGraph(AgentState)

    graph.add_node("summarize", summarize_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("sub_agent", sub_agent_node)
    graph.add_node("synthesize", synthesize_node)

    graph.add_edge(START, "summarize")
    graph.add_edge("summarize", "analyze")
    graph.add_conditional_edges("analyze", dispatch, ["sub_agent"])
    graph.add_edge("sub_agent", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile(checkpointer=checkpointer)
