from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    default_collection: str = "documents"

    # Embedding
    embedding_model: Literal[
        "all-MiniLM-L6-v2",
        "BAAI/bge-base-en-v1.5",
        "text-embedding-3-small",
    ] = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 64

    # Chunking
    chunking_strategy: Literal["fixed", "recursive", "semantic"] = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 50

    # LLM
    ollama_url: str = "http://localhost:11434"
    default_model: str = "qwen2.5-coder:14b"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    max_context_tokens: int = 4096

    # Retrieval
    default_top_k: int = 5
    hybrid_alpha: float = 0.7  # weight toward vector search (1.0 = pure vector)
    use_reranker: bool = False

    # Cache
    cache_enabled: bool = True
    cache_threshold: float = 0.95
    cache_ttl_seconds: int = 3600

    # Eval
    eval_on_query: bool = True
    eval_lightweight: bool = True  # only faithfulness + relevance in query mode

    # Database
    db_path: str = "data/rag_eval.db"

    # Upload
    upload_dir: str = "uploads"
    max_file_size_mb: int = 50
    max_files_per_upload: int = 20

    model_config = {"env_prefix": "RAG_", "env_file": ".env", "extra": "ignore"}


settings = Settings()
