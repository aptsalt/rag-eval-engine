from __future__ import annotations

from dataclasses import dataclass, field

from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.config import settings
from src.ingestion.embedder import embed_texts, get_qdrant_client


@dataclass
class SearchResult:
    text: str
    score: float
    chunk_index: int
    metadata: dict[str, str | int | float] = field(default_factory=dict)


def vector_search(
    query: str,
    collection_name: str,
    top_k: int | None = None,
    source_filter: str | None = None,
) -> list[SearchResult]:
    k = top_k or settings.default_top_k
    client = get_qdrant_client()

    query_embedding = embed_texts([query])[0]

    search_filter = None
    if source_filter:
        search_filter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
        )

    response = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=k,
        query_filter=search_filter,
        with_payload=True,
    )

    search_results: list[SearchResult] = []
    for hit in response.points:
        payload = hit.payload or {}
        search_results.append(SearchResult(
            text=str(payload.get("text", "")),
            score=hit.score,
            chunk_index=int(payload.get("chunk_index", 0)),
            metadata={
                k: v
                for k, v in payload.items()
                if k != "text" and isinstance(v, (str, int, float))
            },
        ))

    return search_results
