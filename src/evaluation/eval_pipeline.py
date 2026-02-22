from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from src.config import settings
from src.db.models import (
    get_db,
    insert_eval_result,
    insert_query_log,
)
from src.evaluation.metrics import EvalScores, evaluate_query
from src.generation.llm_client import generate
from src.generation.prompt_builder import build_prompt, format_messages
from src.retrieval.hybrid_ranker import hybrid_search

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    query_id: str
    answer: str
    sources: list[dict[str, Any]]
    eval_scores: EvalScores | None
    tokens_used: int
    latency_ms: float
    latency_retrieval_ms: float
    latency_generation_ms: float
    model: str
    cache_hit: bool = False
    cost_usd: float = 0.0


async def run_query_pipeline(
    query: str,
    collection: str,
    top_k: int | None = None,
    model: str | None = None,
    evaluate: bool = True,
    lightweight_eval: bool = True,
    alpha: float | None = None,
    auto_tune: bool = False,
) -> QueryResult:
    from src.caching.query_cache import CachedResult, cache_lookup, cache_store

    query_id = str(uuid.uuid4())
    model_name = model or settings.default_model
    start = time.perf_counter()

    # Check cache first
    cached = await cache_lookup(query, collection)
    if cached is not None:
        total_latency = (time.perf_counter() - start) * 1000
        return QueryResult(
            query_id=query_id,
            answer=cached.answer,
            sources=cached.sources,
            eval_scores=None,
            tokens_used=cached.tokens_used,
            latency_ms=total_latency,
            latency_retrieval_ms=0,
            latency_generation_ms=0,
            model=cached.model,
            cache_hit=True,
        )

    # Auto-tune retrieval params if requested
    effective_alpha = alpha
    effective_top_k = top_k
    if auto_tune:
        try:
            from src.retrieval.auto_tune import get_optimal_params

            opt_alpha, opt_top_k = await get_optimal_params(collection)
            if opt_alpha is not None:
                effective_alpha = opt_alpha
            if opt_top_k is not None:
                effective_top_k = opt_top_k
        except Exception:
            pass

    retrieval_start = time.perf_counter()
    results = hybrid_search(query, collection, effective_top_k, effective_alpha)
    latency_retrieval = (time.perf_counter() - retrieval_start) * 1000

    system_prompt, user_prompt, sources = build_prompt(query, results)
    messages = format_messages(system_prompt, user_prompt)

    generation_start = time.perf_counter()
    llm_response = await generate(messages, model_name)
    latency_generation = (time.perf_counter() - generation_start) * 1000

    total_latency = (time.perf_counter() - start) * 1000

    eval_scores = None
    if evaluate:
        context_chunks = [r.text for r in results]
        eval_scores = await evaluate_query(
            query=query,
            answer=llm_response.content,
            context_chunks=context_chunks,
            model=model_name,
            lightweight=lightweight_eval,
            latency_retrieval_ms=latency_retrieval,
            latency_generation_ms=latency_generation,
        )

    source_dicts: list[dict[str, Any]] = []
    for s in sources:
        source_dicts.append(dict(s))

    # Calculate cost
    from src.generation.cost_tracker import calculate_cost

    cost = calculate_cost(model_name, llm_response.tokens_used, 0)

    used_alpha = effective_alpha if effective_alpha is not None else settings.hybrid_alpha
    used_top_k = effective_top_k or settings.default_top_k

    await insert_query_log(
        query_id=query_id,
        collection=collection,
        query=query,
        answer=llm_response.content,
        sources=source_dicts,
        model=model_name,
        tokens_used=llm_response.tokens_used,
        latency_ms=total_latency,
        latency_retrieval_ms=latency_retrieval,
        latency_generation_ms=latency_generation,
        cost_usd=cost,
        alpha=used_alpha,
        top_k=used_top_k,
    )

    if eval_scores:
        eval_id = str(uuid.uuid4())
        await insert_eval_result(
            eval_id=eval_id,
            query_id=query_id,
            faithfulness=eval_scores.faithfulness,
            relevance=eval_scores.relevance,
            hallucination_rate=eval_scores.hallucination_rate,
            context_precision=eval_scores.context_precision,
            context_recall=eval_scores.context_recall,
        )

    # Store in cache
    eval_dict = None
    if eval_scores:
        eval_dict = {
            "faithfulness": eval_scores.faithfulness,
            "relevance": eval_scores.relevance,
            "hallucination_rate": eval_scores.hallucination_rate,
        }

    await cache_store(
        query,
        collection,
        CachedResult(
            answer=llm_response.content,
            sources=source_dicts,
            eval_scores=eval_dict,
            model=model_name,
            created_at=time.time(),
            tokens_used=llm_response.tokens_used,
            latency_ms=total_latency,
        ),
    )

    return QueryResult(
        query_id=query_id,
        answer=llm_response.content,
        sources=source_dicts,
        eval_scores=eval_scores,
        tokens_used=llm_response.tokens_used,
        latency_ms=total_latency,
        latency_retrieval_ms=latency_retrieval,
        latency_generation_ms=latency_generation,
        model=model_name,
        cost_usd=cost,
    )


async def run_batch_eval(
    test_set_id: str,
    model: str | None = None,
) -> dict[str, Any]:
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM test_sets WHERE id = ?", (test_set_id,)
        )
        row = await cursor.fetchone()
        if not row:
            raise ValueError(f"Test set not found: {test_set_id}")
        test_set = dict(row)  # type: ignore[arg-type]
    finally:
        await db.close()

    questions = json.loads(test_set["questions"])
    collection = test_set["collection"]
    run_id = str(uuid.uuid4())

    db = await get_db()
    try:
        await db.execute(
            """INSERT INTO eval_runs (id, test_set_id, status, created_at)
            VALUES (?, ?, 'running', ?)""",
            (run_id, test_set_id, time.time()),
        )
        await db.commit()
    finally:
        await db.close()

    results: list[dict[str, Any]] = []
    total_faithfulness = 0.0
    total_relevance = 0.0
    total_hallucination = 0.0
    total_precision = 0.0
    count = 0

    for q_item in questions:
        question = q_item["question"]
        ground_truth = q_item.get("ground_truth")

        try:
            query_result = await run_query_pipeline(
                query=question,
                collection=collection,
                model=model,
                evaluate=True,
                lightweight_eval=False,
            )

            if query_result.eval_scores:
                if ground_truth:
                    from src.evaluation.metrics import compute_context_recall
                    context_chunks = [s.get("text", "") for s in query_result.sources]
                    recall = await compute_context_recall(
                        query_result.answer, ground_truth, context_chunks, model
                    )
                    query_result.eval_scores.context_recall = recall

                result_data = {
                    "question": question,
                    "answer": query_result.answer,
                    "ground_truth": ground_truth,
                    "faithfulness": query_result.eval_scores.faithfulness,
                    "relevance": query_result.eval_scores.relevance,
                    "hallucination_rate": query_result.eval_scores.hallucination_rate,
                    "context_precision": query_result.eval_scores.context_precision,
                    "context_recall": query_result.eval_scores.context_recall,
                }
                results.append(result_data)

                total_faithfulness += query_result.eval_scores.faithfulness
                total_relevance += query_result.eval_scores.relevance
                total_hallucination += query_result.eval_scores.hallucination_rate
                total_precision += query_result.eval_scores.context_precision
                count += 1

        except Exception as exc:
            logger.error(f"Eval failed for question '{question}': {exc}")
            results.append({
                "question": question,
                "error": str(exc),
            })

    avg_faithfulness = total_faithfulness / count if count > 0 else None
    avg_relevance = total_relevance / count if count > 0 else None
    avg_hallucination = total_hallucination / count if count > 0 else None
    avg_precision = total_precision / count if count > 0 else None

    db = await get_db()
    try:
        await db.execute(
            """UPDATE eval_runs
            SET status = 'completed', results = ?, completed_at = ?,
                avg_faithfulness = ?, avg_relevance = ?,
                avg_hallucination_rate = ?, avg_context_precision = ?
            WHERE id = ?""",
            (
                json.dumps(results),
                time.time(),
                avg_faithfulness,
                avg_relevance,
                avg_hallucination,
                avg_precision,
                run_id,
            ),
        )
        await db.commit()
    finally:
        await db.close()

    return {
        "run_id": run_id,
        "test_set_id": test_set_id,
        "status": "completed",
        "total_questions": len(questions),
        "evaluated": count,
        "avg_faithfulness": avg_faithfulness,
        "avg_relevance": avg_relevance,
        "avg_hallucination_rate": avg_hallucination,
        "avg_context_precision": avg_precision,
        "results": results,
    }
