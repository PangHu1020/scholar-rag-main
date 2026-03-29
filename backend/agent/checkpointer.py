"""Checkpointer factories for the multi-agent RAG system."""

from contextlib import asynccontextmanager

from langgraph.checkpoint.base import BaseCheckpointSaver


def create_memory_checkpointer() -> BaseCheckpointSaver:
    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()


@asynccontextmanager
async def create_postgres_checkpointer(conn_string: str):
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    async with AsyncPostgresSaver.from_conn_string(conn_string) as checkpointer:
        await checkpointer.setup()
        yield checkpointer
