from __future__ import annotations

from dataclasses import dataclass, field

from src.config import settings
from src.retrieval.sparse_search import SparseResult, sparse_search
from src.retrieval.vector_search import SearchResult, vector_search


@dataclass
class RankedResult:
    text: str
    score: float
    vector_score: float
    sparse_score: float
    chunk_index: int
    metadata: dict[str, str | int | float] = field(default_factory=dict)


def hybrid_search(
    query: str,
    collection_name: str,
    top_k: int | None = None,
    alpha: float | None = None,
    source_filter: str | None = None,
) -> list[RankedResult]:
    k = top_k or settings.default_top_k
    weight = alpha if alpha is not None else settings.hybrid_alpha

    fetch_k = k * 3

    vector_results = vector_search(query, collection_name, fetch_k, source_filter)
    sparse_results = sparse_search(query, collection_name, fetch_k)

    fused = reciprocal_rank_fusion(
        vector_results, sparse_results, alpha=weight, top_k=k
    )

    return fused


def reciprocal_rank_fusion(
    vector_results: list[SearchResult],
    sparse_results: list[SparseResult],
    alpha: float = 0.7,
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[RankedResult]:
    scores: dict[str, dict[str, float | str | int | dict[str, str | int | float]]] = {}

    for rank, result in enumerate(vector_results):
        key = _result_key(result.text)
        rrf_score = 1.0 / (rrf_k + rank + 1)
        if key not in scores:
            scores[key] = {
                "text": result.text,
                "vector_rrf": 0.0,
                "sparse_rrf": 0.0,
                "vector_score": result.score,
                "sparse_score": 0.0,
                "chunk_index": result.chunk_index,
                "metadata": result.metadata,
            }
        scores[key]["vector_rrf"] = rrf_score  # type: ignore[assignment]
        scores[key]["vector_score"] = result.score  # type: ignore[assignment]

    for rank, result in enumerate(sparse_results):
        key = _result_key(result.text)
        rrf_score = 1.0 / (rrf_k + rank + 1)
        if key not in scores:
            scores[key] = {
                "text": result.text,
                "vector_rrf": 0.0,
                "sparse_rrf": 0.0,
                "vector_score": 0.0,
                "sparse_score": result.score,
                "chunk_index": result.chunk_index,
                "metadata": result.metadata,
            }
        scores[key]["sparse_rrf"] = rrf_score  # type: ignore[assignment]
        scores[key]["sparse_score"] = result.score  # type: ignore[assignment]

    ranked: list[RankedResult] = []
    for _, data in scores.items():
        vector_rrf = float(data["vector_rrf"])
        sparse_rrf = float(data["sparse_rrf"])
        combined = alpha * vector_rrf + (1 - alpha) * sparse_rrf

        ranked.append(RankedResult(
            text=str(data["text"]),
            score=combined,
            vector_score=float(data["vector_score"]),
            sparse_score=float(data["sparse_score"]),
            chunk_index=int(data["chunk_index"]),
            metadata=data["metadata"] if isinstance(data["metadata"], dict) else {},  # type: ignore[arg-type]
        ))

    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked[:top_k]


def _result_key(text: str) -> str:
    return text[:200].strip().lower()
