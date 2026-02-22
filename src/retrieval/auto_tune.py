from __future__ import annotations

import logging
from typing import Any

from src.db.models import get_db

logger = logging.getLogger(__name__)

MIN_QUERIES_FOR_TUNING = 10
ALPHA_BINS = [round(i * 0.1, 1) for i in range(11)]  # 0.0 to 1.0


async def get_optimal_params(
    collection: str,
) -> tuple[float | None, int | None]:
    """Analyze query_log + eval_results to find retrieval params that produce the highest
    average faithfulness + relevance scores. Returns (alpha, top_k) or (None, None) if
    insufficient data."""
    db = await get_db()
    try:
        cursor = await db.execute(
            """SELECT q.alpha, q.top_k, e.faithfulness, e.relevance
               FROM query_log q
               JOIN eval_results e ON e.query_id = q.id
               WHERE q.collection = ?
                 AND q.alpha IS NOT NULL
                 AND e.faithfulness IS NOT NULL
                 AND e.relevance IS NOT NULL
               ORDER BY q.created_at DESC
               LIMIT 500""",
            (collection,),
        )
        rows = await cursor.fetchall()

        if len(rows) < MIN_QUERIES_FOR_TUNING:
            return None, None

        # Bin alpha values and compute avg quality per bin
        alpha_scores: dict[float, list[float]] = {}
        top_k_scores: dict[int, list[float]] = {}

        for row in rows:
            alpha_val = row[0]
            top_k_val = row[1]
            quality = (row[2] + row[3]) / 2  # avg of faithfulness + relevance

            # Snap alpha to nearest 0.1 bin
            binned_alpha = round(round(alpha_val / 0.1) * 0.1, 1)
            alpha_scores.setdefault(binned_alpha, []).append(quality)

            if top_k_val is not None:
                top_k_scores.setdefault(int(top_k_val), []).append(quality)

        # Find best alpha
        best_alpha: float | None = None
        best_alpha_score = -1.0
        for alpha_bin, scores in alpha_scores.items():
            if len(scores) >= 3:  # Need at least 3 data points per bin
                avg = sum(scores) / len(scores)
                if avg > best_alpha_score:
                    best_alpha_score = avg
                    best_alpha = alpha_bin

        # Find best top_k
        best_top_k: int | None = None
        best_top_k_score = -1.0
        for top_k_val, scores in top_k_scores.items():
            if len(scores) >= 3:
                avg = sum(scores) / len(scores)
                if avg > best_top_k_score:
                    best_top_k_score = avg
                    best_top_k = top_k_val

        return best_alpha, best_top_k
    except Exception as exc:
        logger.warning("Auto-tune failed: %s", exc)
        return None, None
    finally:
        await db.close()


async def get_param_analysis(collection: str) -> dict[str, Any]:
    """Return detailed analysis of retrieval param performance for the dashboard."""
    db = await get_db()
    try:
        cursor = await db.execute(
            """SELECT q.alpha, q.top_k, e.faithfulness, e.relevance
               FROM query_log q
               JOIN eval_results e ON e.query_id = q.id
               WHERE q.collection = ?
                 AND q.alpha IS NOT NULL
                 AND e.faithfulness IS NOT NULL
               ORDER BY q.created_at DESC
               LIMIT 500""",
            (collection,),
        )
        rows = await cursor.fetchall()

        total = len(rows)
        if total < MIN_QUERIES_FOR_TUNING:
            return {
                "sufficient_data": False,
                "total_queries": total,
                "min_required": MIN_QUERIES_FOR_TUNING,
            }

        optimal_alpha, optimal_top_k = await get_optimal_params(collection)

        return {
            "sufficient_data": True,
            "total_queries": total,
            "optimal_alpha": optimal_alpha,
            "optimal_top_k": optimal_top_k,
        }
    except Exception as exc:
        logger.warning("Param analysis failed: %s", exc)
        return {
            "sufficient_data": False,
            "total_queries": 0,
            "min_required": MIN_QUERIES_FOR_TUNING,
        }
    finally:
        await db.close()
