"""Node content generator for different node types."""

from abc import ABC, abstractmethod
from .models import PaperNode, NodeType


class NodeContentGenerator(ABC):
    """Base class for generating node content."""

    @abstractmethod
    def generate_text(
        self,
        node: PaperNode,
        raw_content: str,
        context: dict
    ) -> str:
        """Generate the text field for a node.
        
        Args:
            node: The PaperNode being populated
            raw_content: Raw extracted content
            context: Additional context (section_path, nearby_nodes, etc.)
            
        Returns:
            Generated text for retrieval
        """
        pass


class SectionHeaderGenerator(NodeContentGenerator):
    """Generate content for section headers."""

    def generate_text(
        self,
        _node: PaperNode,
        raw_content: str,
        _context: dict
    ) -> str:
        """Generate section header text."""
        return f"Section: {raw_content}"


class ParagraphGenerator(NodeContentGenerator):
    """Generate content for paragraphs."""

    def generate_text(
        self,
        node: PaperNode,
        raw_content: str,
        _context: dict
    ) -> str:
        """Generate paragraph text with section context."""
        if node.section_path:
            section_path = " > ".join(node.section_path)
            return f"Section: {section_path}\n\nParagraph:\n{raw_content}"
        else:
            return f"Paragraph:\n{raw_content}"


class CaptionGenerator(NodeContentGenerator):
    """Generate content for captions."""

    def generate_text(
        self,
        _node: PaperNode,
        raw_content: str,
        _context: dict
    ) -> str:
        """Generate caption text."""
        return f"Caption: {raw_content}"


class FigureGenerator(NodeContentGenerator):
    """Generate content for figures."""

    def generate_text(
        self,
        _node: PaperNode,
        _raw_content: str,
        context: dict
    ) -> str:
        """Generate figure text from caption and nearby context."""
        caption_text = context.get("caption_text", "")
        nearby_context = context.get("nearby_context", "")
        
        parts = []
        if caption_text:
            parts.append(f"Figure: {caption_text}")
        
        if nearby_context:
            parts.append(f"\nRelated description:\n{nearby_context}")
        
        return "\n".join(parts) if parts else "Figure"


class TableGenerator(NodeContentGenerator):
    """Generate content for tables."""

    def generate_text(
        self,
        _node: PaperNode,
        _raw_content: str,
        context: dict
    ) -> str:
        """Generate table text from caption and linearized content."""
        caption_text = context.get("caption_text", "")
        linearized_table = context.get("linearized_table", "")
        
        parts = []
        if caption_text:
            parts.append(f"Table: {caption_text}")
        
        if linearized_table:
            parts.append(f"\nTable content:\n{linearized_table}")
        
        return "\n".join(parts) if parts else "Table"

    @staticmethod
    def linearize_table(
        headers: list[str],
        rows: list[list[str]]
    ) -> str:
        """Linearize table into key-value format.
        
        Args:
            headers: Column headers
            rows: Table rows
            
        Returns:
            Linearized table string
        """
        lines = []
        for i, row in enumerate(rows, 1):
            pairs = [f"{h}={v}" for h, v in zip(headers, row)]
            lines.append(f"Row {i}: {', '.join(pairs)}")
        return "\n".join(lines)


class NodeContentGeneratorFactory:
    """Factory for creating node content generators."""

    _generators = {
        "section_header": SectionHeaderGenerator(),
        "paragraph": ParagraphGenerator(),
        "caption": CaptionGenerator(),
        "figure": FigureGenerator(),
        "table": TableGenerator(),
    }

    @classmethod
    def get_generator(cls, node_type: NodeType) -> NodeContentGenerator:
        """Get generator for a specific node type.
        
        Args:
            node_type: Type of node
            
        Returns:
            Appropriate content generator
            
        Raises:
            ValueError: If node_type is not supported
        """
        generator = cls._generators.get(node_type)
        if generator is None:
            raise ValueError(f"Unsupported node type: {node_type}")
        return generator
