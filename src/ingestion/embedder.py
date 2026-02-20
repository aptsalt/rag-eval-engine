from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from src.config import settings
from src.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

EmbeddingModel = Literal[
    "all-MiniLM-L6-v2",
    "BAAI/bge-base-en-v1.5",
    "text-embedding-3-small",
]

MODEL_DIMENSIONS: dict[str, int] = {
    "all-MiniLM-L6-v2": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "text-embedding-3-small": 1536,
}

_local_model_cache: dict[str, object] = {}


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )


def _get_local_model(model_name: str) -> object:
    if model_name not in _local_model_cache:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        _local_model_cache[model_name] = SentenceTransformer(model_name)
    return _local_model_cache[model_name]


def embed_texts(
    texts: list[str],
    model: EmbeddingModel | None = None,
) -> list[list[float]]:
    model_name = model or settings.embedding_model

    if model_name == "text-embedding-3-small":
        return _embed_openai(texts)

    return _embed_local(texts, model_name)


def _embed_local(texts: list[str], model_name: str) -> list[list[float]]:
    model = _get_local_model(model_name)
    batch_size = settings.embedding_batch_size
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)  # type: ignore[union-attr]
        for emb in embeddings:
            all_embeddings.append(emb.tolist() if isinstance(emb, np.ndarray) else list(emb))  # type: ignore[union-attr]

        logger.info(
            "Embedded batch %d, total: %d/%d",
            i // batch_size + 1,
            len(all_embeddings),
            len(texts),
        )

    return all_embeddings


def _embed_openai(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI

    if not settings.openai_api_key:
        raise ValueError("OpenAI API key required for text-embedding-3-small")

    client = OpenAI(api_key=settings.openai_api_key)
    all_embeddings: list[list[float]] = []
    batch_size = min(settings.embedding_batch_size, 100)

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model="text-embedding-3-small", input=batch)
        for item in response.data:
            all_embeddings.append(item.embedding)

    return all_embeddings


async def ensure_collection(
    collection_name: str,
    model: EmbeddingModel | None = None,
) -> None:
    client = get_qdrant_client()
    model_name = model or settings.embedding_model
    dimension = MODEL_DIMENSIONS[model_name]

    collections = client.get_collections().collections
    existing_names = {c.name for c in collections}

    if collection_name not in existing_names:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )
        logger.info(f"Created collection: {collection_name} (dim={dimension})")


async def store_chunks(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    collection_name: str,
    doc_id: str,
) -> None:
    client = get_qdrant_client()

    points = []
    for _, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
        payload = {
            "text": chunk.text,
            "doc_id": doc_id,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
            **{k: v for k, v in chunk.metadata.items() if k not in ("text",)},
        }
        point_id = abs(hash(f"{doc_id}_{chunk.chunk_index}")) % (2**63)
        points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        logger.info(
            "Stored batch %d, total: %d/%d",
            i // batch_size + 1,
            min(i + batch_size, len(points)),
            len(points),
        )


async def delete_collection(collection_name: str) -> bool:
    client = get_qdrant_client()
    try:
        client.delete_collection(collection_name=collection_name)
        return True
    except Exception:
        return False


def get_collection_info(collection_name: str) -> dict[str, int]:
    client = get_qdrant_client()
    try:
        info = client.get_collection(collection_name=collection_name)
        return {
            "vectors_count": info.vectors_count or 0,
            "points_count": info.points_count or 0,
        }
    except Exception:
        return {"vectors_count": 0, "points_count": 0}
