from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from src.config import settings
from src.db.models import get_db
from src.ingestion.embedder import embed_texts, get_qdrant_client

logger = logging.getLogger(__name__)

CACHE_COLLECTION = "_query_cache"


@dataclass
class CachedResult:
    answer: str
    sources: list[dict[str, Any]]
    eval_scores: dict[str, Any] | None
    model: str
    created_at: float
    tokens_used: int
    latency_ms: float


async def ensure_cache_collection() -> None:
    client = get_qdrant_client()
    try:
        collections = client.get_collections().collections
        existing = {c.name for c in collections}
        if CACHE_COLLECTION not in existing:
            from src.ingestion.embedder import MODEL_DIMENSIONS

            dim = MODEL_DIMENSIONS[settings.embedding_model]
            client.create_collection(
                collection_name=CACHE_COLLECTION,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            logger.info("Created cache collection: %s", CACHE_COLLECTION)
    except Exception as exc:
        logger.warning("Failed to create cache collection: %s", exc)


def _query_hash(query: str, collection: str) -> str:
    return hashlib.sha256(f"{collection}:{query.strip().lower()}".encode()).hexdigest()


async def cache_lookup(
    query: str,
    collection: str,
) -> CachedResult | None:
    if not settings.cache_enabled:
        return None

    try:
        embedding = embed_texts([query])[0]
        client = get_qdrant_client()

        results = client.query_points(
            collection_name=CACHE_COLLECTION,
            query=embedding,
            limit=1,
            with_payload=True,
        )

        if not results.points:
            await _record_cache_stat(query, collection, hit=False)
            return None

        point = results.points[0]
        if point.score < settings.cache_threshold:
            await _record_cache_stat(query, collection, hit=False)
            return None

        payload = point.payload or {}

        # Check if same collection
        if payload.get("collection") != collection:
            await _record_cache_stat(query, collection, hit=False)
            return None

        # Check TTL
        created_at = float(payload.get("created_at", 0))
        if time.time() - created_at > settings.cache_ttl_seconds:
            await _record_cache_stat(query, collection, hit=False)
            return None

        saved_latency = float(payload.get("latency_ms", 0))
        await _record_cache_stat(query, collection, hit=True, saved_latency_ms=saved_latency)

        return CachedResult(
            answer=str(payload.get("answer", "")),
            sources=json.loads(str(payload.get("sources", "[]"))),
            eval_scores=json.loads(str(payload.get("eval_scores", "null"))),
            model=str(payload.get("model", "")),
            created_at=created_at,
            tokens_used=int(payload.get("tokens_used", 0)),
            latency_ms=float(payload.get("latency_ms", 0)),
        )
    except Exception as exc:
        logger.warning("Cache lookup failed: %s", exc)
        return None


async def cache_store(
    query: str,
    collection: str,
    result: CachedResult,
) -> None:
    if not settings.cache_enabled:
        return

    try:
        embedding = embed_texts([query])[0]
        client = get_qdrant_client()

        await ensure_cache_collection()

        point_id = abs(hash(_query_hash(query, collection))) % (2**63)
        payload = {
            "query": query,
            "collection": collection,
            "answer": result.answer,
            "sources": json.dumps(result.sources),
            "eval_scores": json.dumps(result.eval_scores),
            "model": result.model,
            "created_at": result.created_at,
            "tokens_used": result.tokens_used,
            "latency_ms": result.latency_ms,
        }

        client.upsert(
            collection_name=CACHE_COLLECTION,
            points=[PointStruct(id=point_id, vector=embedding, payload=payload)],
        )
    except Exception as exc:
        logger.warning("Cache store failed: %s", exc)


async def clear_cache() -> int:
    try:
        client = get_qdrant_client()
        info = client.get_collection(CACHE_COLLECTION)
        count = info.points_count or 0
        client.delete_collection(CACHE_COLLECTION)
        return count
    except Exception:
        return 0


async def get_cache_stats() -> dict[str, Any]:
    db = await get_db()
    try:
        cursor = await db.execute("SELECT COUNT(*) FROM cache_stats WHERE hit_or_miss = 'hit'")
        row = await cursor.fetchone()
        hits = row[0] if row else 0

        cursor = await db.execute("SELECT COUNT(*) FROM cache_stats WHERE hit_or_miss = 'miss'")
        row = await cursor.fetchone()
        misses = row[0] if row else 0

        cursor = await db.execute(
            "SELECT AVG(saved_latency_ms) FROM cache_stats WHERE hit_or_miss = 'hit'"
        )
        row = await cursor.fetchone()
        avg_saved = row[0] if row and row[0] else 0

        total = hits + misses
        hit_rate = (hits / total * 100) if total > 0 else 0

        # Get cache size from Qdrant
        cache_size = 0
        try:
            client = get_qdrant_client()
            info = client.get_collection(CACHE_COLLECTION)
            cache_size = info.points_count or 0
        except Exception:
            pass

        return {
            "cache_enabled": settings.cache_enabled,
            "cache_size": cache_size,
            "total_lookups": total,
            "hits": hits,
            "misses": misses,
            "hit_rate_percent": round(hit_rate, 1),
            "avg_saved_latency_ms": round(avg_saved, 1),
            "threshold": settings.cache_threshold,
            "ttl_seconds": settings.cache_ttl_seconds,
        }
    finally:
        await db.close()


async def _record_cache_stat(
    query: str,
    collection: str,
    *,
    hit: bool,
    saved_latency_ms: float = 0,
) -> None:
    try:
        db = await get_db()
        try:
            query_hash = _query_hash(query, collection)
            await db.execute(
                """INSERT INTO cache_stats (id, query_hash, hit_or_miss, saved_latency_ms, created_at)
                VALUES (?, ?, ?, ?, ?)""",
                (
                    hashlib.sha256(f"{time.time()}:{query_hash}".encode()).hexdigest()[:16],
                    query_hash,
                    "hit" if hit else "miss",
                    saved_latency_ms,
                    time.time(),
                ),
            )
            await db.commit()
        finally:
            await db.close()
    except Exception as exc:
        logger.warning("Failed to record cache stat: %s", exc)
