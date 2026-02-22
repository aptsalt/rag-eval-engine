from __future__ import annotations

from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from src.db.models import get_db, get_metrics
from src.evaluation.eval_pipeline import run_batch_eval
from src.evaluation.test_sets import (
    auto_generate_questions,
    create_test_set,
    delete_test_set,
    get_eval_runs,
    get_test_set,
    list_test_sets,
)

router = APIRouter(prefix="/api", tags=["evaluation"])


class CreateTestSetRequest(BaseModel):
    name: str
    collection: str
    questions: list[dict[str, str | None]]


class AutoGenerateRequest(BaseModel):
    collection: str
    num_questions: int = 10
    model: str | None = None
    test_set_name: str | None = None


class BatchEvalRequest(BaseModel):
    test_set_id: str
    model: str | None = None


class MetricsQuery(BaseModel):
    collection: str | None = None
    limit: int = 100


@router.post("/test-sets")
async def create_test_set_endpoint(request: CreateTestSetRequest) -> dict[str, Any]:
    return await create_test_set(
        name=request.name,
        collection=request.collection,
        questions=request.questions,
    )


@router.get("/test-sets")
async def list_test_sets_endpoint() -> list[dict[str, Any]]:
    return await list_test_sets()


@router.get("/test-sets/{test_set_id}")
async def get_test_set_endpoint(test_set_id: str) -> dict[str, Any]:
    result = await get_test_set(test_set_id)
    if not result:
        raise HTTPException(status_code=404, detail="Test set not found")
    return result


@router.delete("/test-sets/{test_set_id}")
async def delete_test_set_endpoint(test_set_id: str) -> dict[str, str]:
    deleted = await delete_test_set(test_set_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Test set not found")
    return {"status": "deleted"}


@router.post("/test-sets/auto-generate")
async def auto_generate_endpoint(request: AutoGenerateRequest) -> dict[str, Any]:
    questions = await auto_generate_questions(
        collection=request.collection,
        num_questions=request.num_questions,
        model=request.model,
    )

    if request.test_set_name and questions:
        result = await create_test_set(
            name=request.test_set_name,
            collection=request.collection,
            questions=questions,
        )
        return {**result, "questions": questions}

    return {"questions": questions, "count": len(questions)}


@router.post("/evaluate/batch")
async def batch_evaluate(
    request: BatchEvalRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    test_set = await get_test_set(request.test_set_id)
    if not test_set:
        raise HTTPException(status_code=404, detail="Test set not found")

    background_tasks.add_task(run_batch_eval, request.test_set_id, request.model)
    return {"status": "started", "test_set_id": request.test_set_id}


@router.get("/evaluate/runs")
async def list_eval_runs(test_set_id: str | None = None) -> list[dict[str, Any]]:
    return await get_eval_runs(test_set_id)


@router.get("/metrics")
async def get_metrics_endpoint(
    collection: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    metrics = await get_metrics(collection, limit)

    if not metrics:
        return {
            "total_queries": 0,
            "avg_faithfulness": None,
            "avg_relevance": None,
            "avg_hallucination_rate": None,
            "avg_latency_ms": None,
            "p50_latency_ms": None,
            "p95_latency_ms": None,
            "time_series": [],
        }

    faithfulness_vals = [m["faithfulness"] for m in metrics if m.get("faithfulness") is not None]
    relevance_vals = [m["relevance"] for m in metrics if m.get("relevance") is not None]
    hallucination_vals = [
        m["hallucination_rate"]
        for m in metrics
        if m.get("hallucination_rate") is not None
    ]
    latency_vals = [m["latency_ms"] for m in metrics if m.get("latency_ms") is not None]

    sorted_latencies = sorted(latency_vals) if latency_vals else []
    p50 = sorted_latencies[len(sorted_latencies) // 2] if sorted_latencies else None
    p95_idx = int(len(sorted_latencies) * 0.95)
    p95 = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)] if sorted_latencies else None

    cost_vals = [m.get("cost_usd", 0) for m in metrics if m.get("cost_usd") is not None]
    total_cost = sum(cost_vals) if cost_vals else 0

    time_series = [
        {
            "query_id": m["id"],
            "timestamp": m["created_at"],
            "faithfulness": m.get("faithfulness"),
            "relevance": m.get("relevance"),
            "hallucination_rate": m.get("hallucination_rate"),
            "latency_ms": m.get("latency_ms"),
            "tokens_used": m.get("tokens_used"),
            "cost_usd": m.get("cost_usd", 0),
        }
        for m in reversed(metrics)
    ]

    return {
        "total_queries": len(metrics),
        "avg_faithfulness": (
            sum(faithfulness_vals) / len(faithfulness_vals)
            if faithfulness_vals
            else None
        ),
        "avg_relevance": (
            sum(relevance_vals) / len(relevance_vals)
            if relevance_vals
            else None
        ),
        "avg_hallucination_rate": (
            sum(hallucination_vals) / len(hallucination_vals)
            if hallucination_vals
            else None
        ),
        "avg_latency_ms": (
            sum(latency_vals) / len(latency_vals)
            if latency_vals
            else None
        ),
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "total_cost_usd": round(total_cost, 4),
        "avg_cost_per_query": round(total_cost / len(metrics), 6) if metrics else 0,
        "time_series": time_series,
    }


@router.get("/metrics/{query_id}")
async def get_query_metrics(query_id: str) -> dict[str, Any]:
    db = await get_db()
    try:
        cursor = await db.execute(
            """SELECT q.*, e.faithfulness, e.relevance, e.hallucination_rate,
                      e.context_precision, e.context_recall
               FROM query_log q
               LEFT JOIN eval_results e ON e.query_id = q.id
               WHERE q.id = ?""",
            (query_id,),
        )
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Query not found")
        return dict(row)  # type: ignore[arg-type]
    finally:
        await db.close()
