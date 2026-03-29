"""Data models for RAG system."""

from dataclasses import dataclass, field
from typing import Optional, Literal, Any

NodeType = Literal[
    "section_header",
    "paragraph",
    "table",
    "figure",
    "caption"
]


@dataclass
class PaperNode:
    """Represents a semantic node in a parsed paper."""
    
    node_id: str
    paper_id: str
    node_type: NodeType

    text: str
    page_num: int
    order: int

    section_path: list[str] = field(default_factory=list)
    bbox: Optional[tuple[float, float, float, float]] = None

    parent_id: Optional[str] = None
    related_ids: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)
