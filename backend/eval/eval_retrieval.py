"""Retrieval evaluation: dataset-agnostic core functions.

Test case format:
    {"query": str, "relevant_ids": list[str]}          # chunk_id based (custom datasets)
    {"query": str, "relevant_pages": list[int], ...}   # page based (MMDocIR)

For MMDocIR, pass a hit_fn that maps (doc_metadata, case) -> bool.
"""

from typing import Callable, Optional
from langchain_core.documents import Document
from rag.retrieval import Retriever


def calculate_metrics(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> dict:
    """Recall@k, Precision@k, MRR, MAP from chunk_id lists."""
    if not relevant_ids:
        return {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "ap": 0.0}

    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    hits = len(retrieved_set & relevant_set)

    recall = hits / len(relevant_set)
    precision = hits / k if k > 0 else 0.0

    mrr = 0.0
    for i, rid in enumerate(retrieved_ids[:k], 1):
        if rid in relevant_set:
            mrr = 1.0 / i
            break

    ap, hits_at_k = 0.0, 0
    for i, rid in enumerate(retrieved_ids[:k], 1):
        if rid in relevant_set:
            hits_at_k += 1
            ap += hits_at_k / i
    ap = ap / len(relevant_set) if relevant_set else 0.0

    return {"recall": recall, "precision": precision, "mrr": mrr, "ap": ap}


def calculate_metrics_from_hits(hits: list[bool], num_relevant: int, k: int) -> dict:
    """Recall@k, Precision@k, MRR, MAP from a boolean hit list."""
    if num_relevant == 0:
        return {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "ap": 0.0}

    hits_k = hits[:k]
    total_hits = sum(hits_k)

    recall = min(total_hits / num_relevant, 1.0)
    precision = total_hits / k if k > 0 else 0.0

    mrr = 0.0
    for i, h in enumerate(hits_k, 1):
        if h:
            mrr = 1.0 / i
            break

    ap, running = 0.0, 0
    for i, h in enumerate(hits_k, 1):
        if h:
            running += 1
            ap += running / i
    ap = ap / num_relevant

    return {"recall": recall, "precision": precision, "mrr": mrr, "ap": ap}


def evaluate_retrieval(
    retriever: Retriever,
    test_cases: list[dict],
    k: int = 5,
    fetch_k: int = 20,
    hit_fn: Optional[Callable[[dict, dict], bool]] = None,
    verbose: bool = False,
) -> dict:
    """Evaluate retrieval on test cases.

    Args:
        retriever: Retriever instance.
        test_cases: List of dicts. Must contain "query" and either:
            - "relevant_ids": list[str]  for chunk_id matching
            - any fields consumed by hit_fn
        k: Top-k for metrics.
        fetch_k: Candidates before reranking.
        hit_fn: Optional callable(doc_metadata, case) -> bool.
                When provided, used instead of chunk_id matching.
        verbose: Print per-query details.

    Returns:
        Dict with averaged Recall@k, Precision@k, MRR, MAP.
    """
    all_metrics = []

    for i, case in enumerate(test_cases, 1):
        query = case["query"]
        results: list[Document] = retriever.retrieve(
            query, k=k, fetch_k=fetch_k, rerank=True, expand_parent=True
        )

        if hit_fn is not None:
            hits = [hit_fn(doc.metadata, case) for doc in results]
            num_relevant = len(case.get("relevant_pages", case.get("relevant_layouts", [1])))
            num_relevant = max(num_relevant, 1)
            metrics = calculate_metrics_from_hits(hits, num_relevant, k)
        else:
            relevant_ids = case.get("relevant_ids", [])
            if not relevant_ids:
                continue
            retrieved_ids = [doc.metadata.get("chunk_id", "") for doc in results]
            metrics = calculate_metrics(retrieved_ids, relevant_ids, k)

        all_metrics.append(metrics)

        if verbose:
            print(f"\n[{i}] {query[:80]}")
            if hit_fn is not None:
                print(f"  hits={hits[:k]}  recall={metrics['recall']:.2%}  mrr={metrics['mrr']:.4f}")
            else:
                print(f"  recall={metrics['recall']:.2%}  precision={metrics['precision']:.2%}")

    n = len(all_metrics)
    if n == 0:
        return {"recall@k": 0.0, "precision@k": 0.0, "mrr": 0.0, "map": 0.0, "num_queries": 0}

    from eval.eval_utils import avg
    return {
        f"recall@{k}": avg([m["recall"] for m in all_metrics]),
        f"precision@{k}": avg([m["precision"] for m in all_metrics]),
        "mrr": avg([m["mrr"] for m in all_metrics]),
        "map": avg([m["ap"] for m in all_metrics]),
        "num_queries": n,
    }
