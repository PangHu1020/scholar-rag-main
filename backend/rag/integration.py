"""PDF parsing and integration with RAG system."""

from pathlib import Path
import uuid
import re
from typing import Optional, Any
from docling.document_converter import DocumentConverter
from .models import PaperNode, NodeType
from .node_generator import NodeContentGeneratorFactory, TableGenerator


class TextCleaner:
    """Clean and normalize text content."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Remove extra whitespace and normalize text."""
        text = re.sub(r'-\n', '', text)
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()

    @staticmethod
    def is_header_footer(text: str, page_height: float, bbox: Optional[tuple]) -> bool:
        """Detect if text is likely header/footer based on position."""
        if not bbox or len(bbox) != 4:
            return False
        _, top, _, bottom = bbox
        margin_threshold = page_height * 0.1
        return top < margin_threshold or bottom > (page_height - margin_threshold)

    @staticmethod
    def is_page_number(text: str) -> bool:
        """Detect if text is a page number."""
        return bool(re.match(r'^\d+$', text.strip()))


class PDFParser:
    """Parse PDF documents into PaperNode structures."""

    def __init__(self):
        self.converter = DocumentConverter()
        self.cleaner = TextCleaner()

    def parse(self, pdf_path: str, paper_id: str) -> list[PaperNode]:
        """Parse PDF file into list of PaperNodes.
        
        Args:
            pdf_path: Path to PDF file
            paper_id: Unique identifier for the paper
            
        Returns:
            List of PaperNode objects
        """
        result = self.converter.convert(pdf_path)
        doc = result.document
        
        page_height = self._get_page_height(doc)
        
        raw_items = [item for item, _ in doc.iterate_items()]
        filtered_items = self._filter_items(raw_items, page_height)
        sorted_items = self._sort_reading_order(filtered_items)
        
        nodes = []
        order = 0
        section_stack = []
        node_map = {}
        
        for i, item in enumerate(sorted_items):
            prev_item = sorted_items[i-1] if i > 0 else None
            next_item = sorted_items[i+1] if i < len(sorted_items)-1 else None
            
            node = self._process_item(
                item, paper_id, order, section_stack, node_map, prev_item, next_item
            )
            if node:
                nodes.append(node)
                node_map[node.node_id] = node
                order += 1
        
        self._link_captions_to_figures_tables(nodes)
        
        return nodes

    def _get_page_height(self, doc) -> float:
        """Extract page height from document."""
        if hasattr(doc, 'pages') and doc.pages:
            try:
                if isinstance(doc.pages, dict):
                    first_page = next(iter(doc.pages.values()))
                else:
                    first_page = doc.pages[0]
                
                if hasattr(first_page, 'size') and hasattr(first_page.size, 'height'):
                    return first_page.size.height
            except (KeyError, IndexError, StopIteration):
                pass
        return 792.0

    def _filter_items(self, items: list[Any], page_height: float) -> list[Any]:
        """Filter out headers, footers, and page numbers."""
        filtered = []
        
        for item in items:
            if not hasattr(item, 'text'):
                filtered.append(item)
                continue
            
            text = item.text.strip()
            if not text:
                continue
            
            bbox = self._extract_bbox(item)
            
            if self.cleaner.is_page_number(text):
                continue
            
            if bbox and self.cleaner.is_header_footer(text, page_height, bbox):
                continue
            
            filtered.append(item)
        
        return filtered

    def _sort_reading_order(self, items: list[Any]) -> list[Any]:
        """Sort items by reading order (handles single/double column)."""
        items_with_pos = []
        page_widths = {}
        
        for item in items:
            bbox = self._extract_bbox(item)
            page_num = item.prov[0].page_no if item.prov and len(item.prov) > 0 else 0
            
            if bbox:
                left, top, right, _ = bbox
                if page_num not in page_widths:
                    page_widths[page_num] = right
                else:
                    page_widths[page_num] = max(page_widths[page_num], right)
                
                items_with_pos.append((item, page_num, top, left, bbox))
            else:
                items_with_pos.append((item, page_num, 0, 0, None))
        
        for i, (item, page_num, top, left, bbox) in enumerate(items_with_pos):
            if bbox:
                page_width = page_widths.get(page_num, 612.0)
                column = 0 if left < page_width * 0.5 else 1
                items_with_pos[i] = (item, page_num, column, top, left)
            else:
                items_with_pos[i] = (item, page_num, 0, top, left)
        
        items_with_pos.sort(key=lambda x: (x[1], x[2], x[3], x[4]))
        
        return [item for item, _, _, _, _ in items_with_pos]

    def _process_item(
        self,
        item,
        paper_id: str,
        order: int,
        section_stack: list[str],
        node_map: dict[str, PaperNode],
        prev_item,
        next_item
    ) -> Optional[PaperNode]:
        """Process a single document item into a PaperNode."""
        item_type = type(item).__name__
        raw_text = item.text if hasattr(item, 'text') else ""
        raw_text = self.cleaner.clean_text(raw_text)
        
        if self._is_caption_text(raw_text):
            node_type = "caption"
        else:
            node_type = self._map_item_type(item_type)
        
        if not node_type:
            return None
        
        node_id = str(uuid.uuid4())
        page_num = item.prov[0].page_no if item.prov and len(item.prov) > 0 else 0
        bbox = self._extract_bbox(item) if node_type in ["figure", "table"] else None
        
        if node_type == "section_header":
            self._update_section_stack(section_stack, raw_text)
        
        generator = NodeContentGeneratorFactory.get_generator(node_type)
        
        node = PaperNode(
            node_id=node_id,
            paper_id=paper_id,
            node_type=node_type,
            text="",
            page_num=page_num,
            order=order,
            section_path=section_stack.copy(),
            bbox=bbox
        )
        
        if node_type == "table":
            node.metadata["item"] = item
        
        context = {"raw_text": raw_text, "item": item}
        node.text = generator.generate_text(node, raw_text, context)
        
        return node

    def _map_item_type(self, item_type: str) -> Optional[NodeType]:
        """Map Docling item type to NodeType."""
        mapping = {
            "SectionHeaderItem": "section_header",
            "TextItem": "paragraph",
            "TableItem": "table",
            "PictureItem": "figure",
        }
        return mapping.get(item_type)

    def _is_caption_text(self, text: str) -> bool:
        """Check if text is a caption based on pattern."""
        if not text:
            return False
        text = text.strip()
        return bool(re.match(r'^(Caption:\s*)?(Figure|Table|Fig\.|Tab\.)\s+\d+', text, re.IGNORECASE))

    def _extract_bbox(self, item) -> Optional[tuple[float, float, float, float]]:
        """Extract bounding box from item."""
        if hasattr(item, 'self_ref') and item.self_ref:
            ref = item.self_ref
            if hasattr(ref, 'bbox') and ref.bbox:
                bbox = ref.bbox
                return (bbox.l, bbox.t, bbox.r, bbox.b)
        
        if hasattr(item, 'prov') and item.prov and len(item.prov) > 0:
            prov = item.prov[0]
            if hasattr(prov, 'bbox') and prov.bbox:
                bbox = prov.bbox
                return (bbox.l, bbox.t, bbox.r, bbox.b)
        
        if hasattr(item, 'bbox') and item.bbox:
            bbox = item.bbox
            return (bbox.l, bbox.t, bbox.r, bbox.b)
        
        return None

    def _update_section_stack(self, section_stack: list[str], header_text: str):
        """Update section stack based on header level.
        
        Handles numbered sections (e.g., '3.2.1 Title') by maintaining hierarchy.
        """
        match = re.match(r'^(\d+(?:\.\d+)*)\s+', header_text)
        if match:
            level = max(1, len(match.group(1).split('.')))
            section_stack[:] = section_stack[:level-1] + [header_text]
        else:
            section_stack.append(header_text)

    def _linearize_table(self, item) -> str:
        """Linearize table into key-value format."""
        if not hasattr(item, 'data') or not item.data:
            return ""
        
        data = item.data
        if not hasattr(data, 'table_cells') or not data.table_cells or len(data.table_cells) == 0:
            return ""
        
        try:
            cells = data.table_cells
            rows_dict = {}
            for cell in cells:
                row_idx = cell.start_row_offset_idx if hasattr(cell, 'start_row_offset_idx') else 0
                col_idx = cell.start_col_offset_idx if hasattr(cell, 'start_col_offset_idx') else 0
                text = cell.text if hasattr(cell, 'text') else str(cell)
                
                if row_idx not in rows_dict:
                    rows_dict[row_idx] = {}
                rows_dict[row_idx][col_idx] = text
            
            if not rows_dict:
                return ""
            
            sorted_rows = sorted(rows_dict.items())
            headers = [sorted_rows[0][1].get(i, "") for i in range(max(sorted_rows[0][1].keys()) + 1)] if sorted_rows else []
            rows = [[row_data.get(i, "") for i in range(max(row_data.keys()) + 1)] for _, row_data in sorted_rows[1:]]
            
            if not headers:
                return ""
                
        except (AttributeError, IndexError, TypeError, KeyError) as e:
            return ""
        
        return TableGenerator.linearize_table(headers, rows)

    def _link_captions_to_figures_tables(self, nodes: list[PaperNode]):
        """Link captions to their corresponding figures/tables."""
        for node in nodes:
            if node.node_type in ["figure", "table"]:
                caption_type = "Figure" if node.node_type == "figure" else "Table"
                caption_text = self._find_caption_for_node(node, nodes, caption_type)
                if caption_text:
                    generator = NodeContentGeneratorFactory.get_generator(node.node_type)
                    context = {"caption_text": caption_text}
                    if node.node_type == "table" and 'item' in node.metadata:
                        linearized = self._linearize_table(node.metadata['item'])
                        if linearized:
                            context["linearized_table"] = linearized
                    node.text = generator.generate_text(node, "", context)

    
    def _find_caption_for_node(self, node: PaperNode, nodes: list[PaperNode], caption_type: str) -> str:
        """Find caption for a figure/table node."""
        if not node.bbox:
            return ""
        
        best_caption = ""
        min_distance = float('inf')
        
        for other in nodes:
            if other.node_type == "caption" and other.page_num == node.page_num:
                if caption_type.lower() in other.text.lower():
                    if other.bbox:
                        distance = abs(other.bbox[1] - node.bbox[1])
                        if distance < min_distance and distance < 300:
                            min_distance = distance
                            best_caption = other.text
        
        return best_caption


