"""Configuration management from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration."""
    
    MILVUS_URI = os.getenv("MILVUS_URI", "./data/milvus.db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "papers")
    
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    
    FETCH_K = int(os.getenv("FETCH_K", "20"))
    TOP_K = int(os.getenv("TOP_K", "5"))
    RRF_K = int(os.getenv("RRF_K", "60"))
    
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))
    
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:32b")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))

    VLM_ENABLED = os.getenv("VLM_ENABLED", "false").lower() == "true"
    VLM_BASE_URL = os.getenv("VLM_BASE_URL", "http://localhost:11434/v1")
    VLM_MODEL = os.getenv("VLM_MODEL", "qwen-vl")
    VLM_API_KEY = os.getenv("VLM_API_KEY", "ollama")

    POSTGRES_URI = os.getenv("POSTGRES_URI", "postgresql://postgres:postgres@localhost:5432/scholar_rag")

    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
    MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))

    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
