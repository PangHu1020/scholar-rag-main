"""State definitions for the multi-agent RAG system."""

import operator
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class SubAnswer(TypedDict):
    query: str
    answer: str
    citations: list[dict]


def merge_sub_answers(left: list[SubAnswer], right: list[SubAnswer]) -> list[SubAnswer]:
    existing = {a["query"] for a in left}
    merged = list(left)
    for a in right:
        if a["query"] in existing:
            merged = [a if item["query"] == a["query"] else item for item in merged]
        else:
            merged.append(a)
    return merged


def merge_citations(left: list[dict], right: list[dict]) -> list[dict]:
    seen = set()
    merged = []
    for c in left + right:
        key = c.get("chunk_id") or id(c)
        if key not in seen:
            seen.add(key)
            merged.append(c)
    return merged


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    summary: str
    sub_queries: list[str]
    sub_answers: Annotated[list[SubAnswer], merge_sub_answers]
    answer: str
    citations: Annotated[list[dict], merge_citations]


class SubAgentState(TypedDict):
    query: str
    documents: list[str]
    answer: str
    citations: list[dict]
    is_sufficient: bool
    retry_queries: Annotated[list[str], operator.add]
    retries: int
