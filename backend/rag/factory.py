"""Factory for creating reusable RAG components."""

from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder


class EmbeddingService:
    """Singleton for embedding model."""
    
    _instances = {}
    
    @classmethod
    def get_embeddings(cls, model_name: str) -> HuggingFaceEmbeddings:
        """Get or create embedding model instance."""
        if model_name not in cls._instances:
            cls._instances[model_name] = HuggingFaceEmbeddings(model_name=model_name)
        return cls._instances[model_name]


class RerankerService:
    """Singleton for reranker model."""
    
    _instances = {}
    
    @classmethod
    def get_reranker(cls, model_name: str) -> CrossEncoder:
        """Get or create reranker model instance."""
        if model_name not in cls._instances:
            cls._instances[model_name] = CrossEncoder(model_name)
        return cls._instances[model_name]


class MilvusStoreFactory:
    """Factory for creating Milvus stores."""
    
    @staticmethod
    def create_store(
        embeddings: HuggingFaceEmbeddings,
        milvus_uri: str,
        collection_name: str,
        is_child: bool = True,
    ) -> Milvus:
        """Create Milvus store with hybrid search."""
        bm25 = BM25BuiltInFunction(input_field_names="text", output_field_names="sparse")
        suffix = "children" if is_child else "parents"
        
        return Milvus(
            embeddings,
            builtin_function=bm25,
            vector_field=["dense", "sparse"],
            collection_name=f"{collection_name}_{suffix}",
            connection_args={"uri": milvus_uri},
        )
