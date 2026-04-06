"""MMDocIR benchmark adapter for the scholar-rag retrieval system.

Converts MMDocIR annotations into test cases compatible with eval_retrieval.py.
Ground truth matching is done at page level (page_id overlap) since our system
stores page_num in chunk metadata, while layout-level bbox matching requires
the PDF to be indexed first.
"""

import json
import ast
from pathlib import Path
from typing import Optional

BENCHMARK_DIR = Path(__file__).parent / "benchmark" / "MMDocIR"
ANNOTATIONS_FILE = BENCHMARK_DIR / "MMDocIR_annotations.jsonl"


def _parse_type(raw_type: str) -> list[str]:
    """Parse the type field which may be a string repr of a list."""
    if raw_type.startswith("["):
        try:
            return ast.literal_eval(raw_type)
        except Exception:
            return [raw_type]
    return [raw_type]


def load_test_cases(
    domain_filter: Optional[str] = None,
    type_filter: Optional[list[str]] = None,
    max_docs: Optional[int] = None,
    max_questions_per_doc: Optional[int] = None,
) -> list[dict]:
    """Load MMDocIR annotations as retrieval test cases.

    Each test case has:
        query: str
        relevant_pages: list[int]   - 0-indexed page ids from ground truth
        relevant_layouts: list[dict] - layout_mapping entries for bbox matching
        doc_name: str
        question_type: list[str]

    Args:
        domain_filter: Only include docs whose domain contains this string
                       (e.g. "Academic paper", "Research report").
        type_filter: Only include questions whose type overlaps with this list
                     (e.g. ["Pure-text (Plain-text)", "text-only"]).
        max_docs: Limit number of documents.
        max_questions_per_doc: Limit questions per document.
    """
    test_cases = []
    doc_count = 0

    with open(ANNOTATIONS_FILE) as f:
        for line in f:
            doc = json.loads(line)

            if domain_filter and domain_filter not in doc["domain"]:
                continue

            doc_count += 1
            if max_docs and doc_count > max_docs:
                break

            questions = doc["questions"]
            if max_questions_per_doc:
                questions = questions[:max_questions_per_doc]

            for q in questions:
                q_types = _parse_type(q["type"])

                if type_filter:
                    if not any(t in type_filter for t in q_types):
                        continue

                test_cases.append({
                    "query": q["Q"],
                    "reference_answer": q["A"],
                    "relevant_pages": q["page_id"],
                    "relevant_layouts": q["layout_mapping"],
                    "paper_id": doc["doc_name"].replace(".pdf", ""),
                    "domain": doc["domain"],
                    "question_type": q_types,
                })

    return test_cases


def iou(bbox_a: list, bbox_b: list) -> float:
    """Compute IoU between two [x1,y1,x2,y2] bboxes."""
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    return inter / (area_a + area_b - inter)


def is_hit_page(doc_metadata: dict, case: dict) -> bool:
    """Check if a retrieved document is from the correct paper and covers a relevant page."""
    if doc_metadata.get("paper_id") != case.get("paper_id"):
        return False
    page_num = doc_metadata.get("page_num")
    if page_num is None:
        return False
    return (page_num - 1) in case["relevant_pages"]


def is_hit_layout(doc_metadata: dict, case: dict, iou_threshold: float = 0.3) -> bool:
    """Check if a retrieved document is from the correct paper and bbox overlaps ground truth."""
    if doc_metadata.get("paper_id") != case.get("paper_id"):
        return False
    page_num = doc_metadata.get("page_num")
    bbox_str = doc_metadata.get("bbox")
    if page_num is None or not bbox_str:
        return False

    page_0indexed = page_num - 1

    try:
        bbox = ast.literal_eval(bbox_str) if isinstance(bbox_str, str) else bbox_str
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return False
        bbox = [float(v) for v in bbox[:4]]
    except Exception:
        return False

    for layout in case["relevant_layouts"]:
        if layout["page"] != page_0indexed:
            continue
        if iou(bbox, layout["bbox"]) >= iou_threshold:
            return True

    return False
