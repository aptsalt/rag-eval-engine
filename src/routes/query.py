from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.config import settings
from src.evaluation.eval_pipeline import run_query_pipeline
from src.generation.llm_client import generate_stream
from src.generation.prompt_builder import build_prompt, format_messages
from src.retrieval.hybrid_ranker import hybrid_search

router = APIRouter(prefix="/api", tags=["query"])


class QueryRequest(BaseModel):
    query: str
    collection: str = "documents"
    top_k: int = 5
    model: str | None = None
    stream: bool = False
    evaluate: bool = True


class SourceInfo(BaseModel):
    index: int
    text: str
    source: str
    score: float
    chunk_index: int


class EvalScoresResponse(BaseModel):
    faithfulness: float
    relevance: float
    hallucination_rate: float
    context_precision: float
    context_recall: float | None = None
    latency_retrieval_ms: float
    latency_generation_ms: float


class QueryResponse(BaseModel):
    query_id: str
    answer: str
    sources: list[dict[str, Any]]
    eval_scores: EvalScoresResponse | None = None
    tokens_used: int
    latency_ms: float
    model: str


@router.post("/query", response_model=None)
async def query_rag(request: QueryRequest) -> QueryResponse | StreamingResponse:
    if request.stream:
        return await _stream_query(request)

    result = await run_query_pipeline(
        query=request.query,
        collection=request.collection,
        top_k=request.top_k,
        model=request.model,
        evaluate=request.evaluate,
        lightweight_eval=settings.eval_lightweight,
    )

    eval_response = None
    if result.eval_scores:
        eval_response = EvalScoresResponse(
            faithfulness=result.eval_scores.faithfulness,
            relevance=result.eval_scores.relevance,
            hallucination_rate=result.eval_scores.hallucination_rate,
            context_precision=result.eval_scores.context_precision,
            context_recall=result.eval_scores.context_recall,
            latency_retrieval_ms=result.eval_scores.latency_retrieval_ms,
            latency_generation_ms=result.eval_scores.latency_generation_ms,
        )

    return QueryResponse(
        query_id=result.query_id,
        answer=result.answer,
        sources=result.sources,
        eval_scores=eval_response,
        tokens_used=result.tokens_used,
        latency_ms=result.latency_ms,
        model=result.model,
    )


async def _stream_query(request: QueryRequest) -> StreamingResponse:
    async def event_generator():  # type: ignore[no-untyped-def]
        results = hybrid_search(
            query=request.query,
            collection_name=request.collection,
            top_k=request.top_k,
        )

        system_prompt, user_prompt, sources = build_prompt(request.query, results)
        messages = format_messages(system_prompt, user_prompt)

        sources_event = json.dumps({"type": "sources", "data": sources})
        yield f"data: {sources_event}\n\n"

        model = request.model or settings.default_model
        full_answer = ""
        async for chunk in generate_stream(messages, model):
            full_answer += chunk
            token_event = json.dumps({"type": "token", "data": chunk})
            yield f"data: {token_event}\n\n"

        done_event = json.dumps({"type": "done", "data": {"answer": full_answer}})
        yield f"data: {done_event}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
