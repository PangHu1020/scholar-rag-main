"""Evaluation dataset for DualPath paper."""

DUALPATH_QUERIES = [
    {
        "query": "What is DualPath and what problem does it solve?",
        "relevant_ids": ["1e263f92-39ad-4509-b700-7fcc5f40ddb2", "ead81cc4-d050-4733-8bb3-fb666d55299b"],
    },
    {
        "query": "How does DualPath improve LLM inference throughput?",
        "relevant_ids": ["b0f89ccb-a946-4dda-93f3-493d4f1fb9db", "b423f031-90cc-4458-a491-6b66fb65dbe8"],
    },
    {
        "query": "What is the dual-path KV-Cache loading mechanism?",
        "relevant_ids": ["ead81cc4-d050-4733-8bb3-fb666d55299b"],
    },
    {
        "query": "What is the storage bandwidth bottleneck in agentic inference?",
        "relevant_ids": ["1e263f92-39ad-4509-b700-7fcc5f40ddb2"],
    },
    {
        "query": "How does the scheduler balance load across prefill and decode engines?",
        "relevant_ids": [],
    },
    {
        "query": "What are the experimental results of DualPath on DeepSeek 660B?",
        "relevant_ids": ["75dbcb1c-047c-4468-b385-c0729ae7b68d"],
    },
    {
        "query": "How does DualPath handle KV-Cache reuse across requests?",
        "relevant_ids": [],
    },
    {
        "query": "What is the performance comparison between DualPath and baseline systems?",
        "relevant_ids": ["75dbcb1c-047c-4468-b385-c0729ae7b68d"],
    },
    {
        "query": "What are the key components of DualPath architecture?",
        "relevant_ids": ["ead81cc4-d050-4733-8bb3-fb666d55299b"],
    },
    {
        "query": "How does DualPath optimize memory bandwidth usage?",
        "relevant_ids": [],
    },
]
