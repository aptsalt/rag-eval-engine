from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from src.config import settings
from src.retrieval.hybrid_ranker import hybrid_search

router = APIRouter(prefix="/api", tags=["retrieval"])


class RetrieveRequest(BaseModel):
    query: str
    collection: str = "documents"
    top_k: int = 5
    alpha: float | None = None
    source_filter: str | None = None


class ChunkResult(BaseModel):
    text: str
    score: float
    vector_score: float
    sparse_score: float
    chunk_index: int
    metadata: dict[str, str | int | float]


class RetrieveResponse(BaseModel):
    query: str
    chunks: list[ChunkResult]
    total_results: int
    retrieval_method: str


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_chunks(request: RetrieveRequest) -> RetrieveResponse:
    results = hybrid_search(
        query=request.query,
        collection_name=request.collection,
        top_k=request.top_k,
        alpha=request.alpha,
        source_filter=request.source_filter,
    )

    chunks = [
        ChunkResult(
            text=r.text,
            score=r.score,
            vector_score=r.vector_score,
            sparse_score=r.sparse_score,
            chunk_index=r.chunk_index,
            metadata=r.metadata,
        )
        for r in results
    ]

    return RetrieveResponse(
        query=request.query,
        chunks=chunks,
        total_results=len(chunks),
        retrieval_method=f"hybrid (alpha={request.alpha or settings.hybrid_alpha})",
    )
