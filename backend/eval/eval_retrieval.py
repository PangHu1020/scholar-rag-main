"""Evaluate retrieval performance with multiple metrics."""

import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.retrieval import Retriever


def calculate_metrics(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> Dict:
    """Calculate Recall@k, Precision@k, MRR, and MAP."""
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
    
    ap = 0.0
    hits_at_k = 0
    for i, rid in enumerate(retrieved_ids[:k], 1):
        if rid in relevant_set:
            hits_at_k += 1
            ap += hits_at_k / i
    ap = ap / len(relevant_set) if relevant_set else 0.0
    
    return {"recall": recall, "precision": precision, "mrr": mrr, "ap": ap}


def evaluate_retrieval(
    retriever: Retriever,
    test_cases: List[Dict],
    k: int = 5,
    fetch_k: int = 20,
    verbose: bool = False,
) -> Dict:
    """Evaluate retrieval on test cases.
    
    Args:
        retriever: Retriever instance
        test_cases: List of {"query": str, "relevant_ids": List[str]}
        k: Number of results to retrieve
        fetch_k: Number of candidates before reranking
        verbose: Print detailed results
    
    Returns:
        Dict with averaged metrics
    """
    all_metrics = []
    
    for i, case in enumerate(test_cases, 1):
        query = case["query"]
        relevant_ids = case["relevant_ids"]
        
        if not relevant_ids:
            continue
        
        results = retriever.retrieve(query, k=k, fetch_k=fetch_k, rerank=True, expand_parent=True)
        retrieved_ids = [doc.metadata.get("chunk_id", "") for doc in results]
        
        metrics = calculate_metrics(retrieved_ids, relevant_ids, k)
        all_metrics.append(metrics)
        
        if verbose:
            print(f"\n[Query {i}] {query}")
            print(f"  Relevant: {relevant_ids}")
            print(f"  Retrieved: {retrieved_ids[:3]}...")
            print(f"  Recall: {metrics['recall']:.2%}, Precision: {metrics['precision']:.2%}")
    
    avg_metrics = {
        "recall@k": sum(m["recall"] for m in all_metrics) / len(all_metrics) if all_metrics else 0.0,
        "precision@k": sum(m["precision"] for m in all_metrics) / len(all_metrics) if all_metrics else 0.0,
        "mrr": sum(m["mrr"] for m in all_metrics) / len(all_metrics) if all_metrics else 0.0,
        "map": sum(m["ap"] for m in all_metrics) / len(all_metrics) if all_metrics else 0.0,
        "num_queries": len(all_metrics),
    }
    
    return avg_metrics


if __name__ == "__main__":
    from dualpath_dataset import DUALPATH_QUERIES
    
    MILVUS_URI = "http://localhost:19530"
    COLLECTION = "test_papers"
    EMBEDDING_MODEL = "/mnt/zh/project/Qwen3-Embedding-0.6B"
    RERANKER_MODEL = "/mnt/zh/project/bge-reranker-v2-m3"
    
    retriever = Retriever(
        embedding_model=EMBEDDING_MODEL,
        reranker_model=RERANKER_MODEL,
        milvus_uri=MILVUS_URI,
        collection_name=COLLECTION,
    )
    
    metrics = evaluate_retrieval(retriever, DUALPATH_QUERIES, k=5, fetch_k=20, verbose=True)
    print(f"\n{'='*50}")
    print(f"Recall@5: {metrics['recall@k']:.2%}")
    print(f"Precision@5: {metrics['precision@k']:.2%}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"MAP: {metrics['map']:.4f}")
    print(f"Queries: {metrics['num_queries']}")
