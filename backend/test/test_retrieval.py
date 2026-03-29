"""Test retrieval pipeline: parse -> index -> retrieve, with structured log output."""

import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.integration import PDFParser, RAGIntegration
from rag.retrieval import Retriever

LOG_DIR = Path(__file__).parent.parent / "log"
LOG_DIR.mkdir(exist_ok=True)

TEST_QUERIES = [
    "What is DualPath and how does it improve LLM inference throughput?",
    "How does the dual-path KV-Cache loading work?",
    "What is the storage bandwidth bottleneck in agentic inference?",
    "How does the scheduler balance load across prefill and decode engines?",
    "What are the experimental results of DualPath on DS 660B?",
]

MILVUS_URI = "http://localhost:19530"
COLLECTION = "test_papers"
EMBEDDING_MODEL = "/mnt/zh/project/Qwen3-Embedding-0.6B"
RERANKER_MODEL = "/mnt/zh/project/bge-reranker-v2-m3"


def fmt_meta(meta: dict) -> str:
    lines = []
    for k, v in sorted(meta.items()):
        val = str(v)
        if len(val) > 120:
            val = val[:120] + "..."
        lines.append(f"    {k}: {val}")
    return "\n".join(lines)


def run_test(pdf_path: str):
    output = []

    def log(msg: str = ""):
        print(msg)
        output.append(msg)

    log(f"{'='*90}")
    log(f"  Retrieval Pipeline Test")
    log(f"  PDF: {pdf_path}")
    log(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{'='*90}")

    # ---- Step 1: Parse ----
    log("\n[STEP 1] Parsing PDF ...")
    t0 = time.time()
    parser = PDFParser()
    nodes = parser.parse(pdf_path, "test_paper")
    log(f"  Parsed {len(nodes)} nodes in {time.time()-t0:.1f}s")

    # ---- Step 2: Convert & Chunk ----
    log("\n[STEP 2] Converting to Documents & Chunking ...")
    rag = RAGIntegration(
        embedding_model=EMBEDDING_MODEL,
        milvus_uri=MILVUS_URI,
        collection_name=COLLECTION,
    )
    docs = rag.nodes_to_documents(nodes)
    parents, children = rag.create_chunks(docs)
    log(f"  Documents: {len(docs)}")
    log(f"  Parent chunks: {len(parents)}")
    log(f"  Child chunks:  {len(children)}")

    split_count = sum(1 for c in children if "_child_1" in c.metadata.get("chunk_id", ""))
    log(f"  Docs that were split: {split_count}")

    # ---- Step 3: Index ----
    log("\n[STEP 3] Indexing into Milvus (hybrid: dense + BM25) ...")
    t0 = time.time()
    ok = rag.store_in_milvus(parents, children)
    log(f"  Index result: {'OK' if ok else 'FAILED'}  ({time.time()-t0:.1f}s)")
    if not ok:
        log("  Aborting.")
        return

    # ---- Step 4: Retrieve ----
    log("\n[STEP 4] Retrieval Tests")
    log(f"  Initializing Retriever ...")
    t0 = time.time()
    retriever = Retriever(
        embedding_model=EMBEDDING_MODEL,
        reranker_model=RERANKER_MODEL,
        milvus_uri=MILVUS_URI,
        collection_name=COLLECTION,
    )
    log(f"  Retriever ready ({time.time()-t0:.1f}s)")

    for qi, query in enumerate(TEST_QUERIES, 1):
        log(f"\n{'='*90}")
        log(f"  Query {qi}: {query}")
        log(f"{'='*90}")

        for mode_name, kwargs in [
            ("hybrid+rerank+parent", dict(rerank=True, expand_parent=True)),
            ("hybrid only (no rerank, no parent)", dict(rerank=False, expand_parent=False)),
            ("hybrid+rerank (no parent)", dict(rerank=True, expand_parent=False)),
        ]:
            log(f"\n  --- Mode: {mode_name} ---")
            t0 = time.time()
            results = retriever.retrieve(query, k=5, fetch_k=20, **kwargs)
            elapsed = time.time() - t0
            log(f"  Results: {len(results)} docs ({elapsed:.2f}s)")

            for ri, doc in enumerate(results, 1):
                log(f"\n  [{ri}] chunk_id: {doc.metadata.get('chunk_id', 'N/A')}")
                log(f"      node_type: {doc.metadata.get('node_type', 'N/A')}")
                log(f"      page_num:  {doc.metadata.get('page_num', 'N/A')}")
                log(f"      section:   {doc.metadata.get('section_path', 'N/A')}")
                if doc.metadata.get("chunk_parent_id"):
                    log(f"      parent_id: {doc.metadata['chunk_parent_id']}")
                text_preview = doc.page_content[:200].replace("\n", " ")
                log(f"      text:      {text_preview}...")
                log(f"      metadata:")
                log(fmt_meta(doc.metadata))

    # ---- Save ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_name = Path(pdf_path).stem
    log_file = LOG_DIR / f"retrieval_{pdf_name}_{timestamp}.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output))
    print(f"\nLog saved to: {log_file}")


if __name__ == "__main__":
    default_pdf = str(Path(__file__).parent.parent.parent / "pdf" / "DualPath.pdf")
    pdf = sys.argv[1] if len(sys.argv) > 1 else default_pdf
    run_test(pdf)
