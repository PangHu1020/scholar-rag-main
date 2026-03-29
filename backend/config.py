"""Configuration management from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration."""
    
    MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "papers")
    
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    
    FETCH_K = int(os.getenv("FETCH_K", "20"))
    TOP_K = int(os.getenv("TOP_K", "5"))
    RRF_K = int(os.getenv("RRF_K", "60"))
    
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))
