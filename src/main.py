from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.db.models import init_db
from src.generation.llm_client import check_ollama_health, list_ollama_models
from src.routes import evaluate, ingest, query, retrieve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Starting RAG Eval Engine...")
    await init_db()
    logger.info("Database initialized")

    if settings.cache_enabled:
        from src.caching.query_cache import ensure_cache_collection

        await ensure_cache_collection()
        logger.info("Query cache initialized")

    ollama_ok = await check_ollama_health()
    if ollama_ok:
        models = await list_ollama_models()
        model_names = [m.get("name", "unknown") for m in models]
        logger.info(f"Ollama connected. Models: {model_names}")
    else:
        logger.warning("Ollama not available. LLM features will fail until connected.")

    yield
    logger.info("Shutting down RAG Eval Engine")


app = FastAPI(
    title="RAG Eval Engine",
    description=(
        "Production RAG system with built-in evaluation"
        " — hybrid retrieval, multi-model support,"
        " quality metrics dashboard"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3400", "http://localhost:3500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(retrieve.router)
app.include_router(query.router)
app.include_router(evaluate.router)


@app.get("/health")
async def health_check() -> dict[str, Any]:
    ollama_ok = await check_ollama_health()
    return {
        "status": "healthy",
        "ollama": "connected" if ollama_ok else "disconnected",
        "embedding_model": settings.embedding_model,
        "default_llm": settings.default_model,
        "eval_enabled": settings.eval_on_query,
    }


@app.get("/api/models")
async def get_models() -> list[dict[str, Any]]:
    models = await list_ollama_models()
    return [
        {
            "name": m.get("name", "unknown"),
            "size": m.get("size"),
            "modified_at": m.get("modified_at"),
        }
        for m in models
    ]


@app.middleware("http")
async def add_response_time(request: Request, call_next: Any) -> Response:
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    response.headers["X-Response-Time"] = f"{elapsed:.1f}ms"
    return response


@app.get("/api/cache/stats")
async def cache_stats() -> dict[str, Any]:
    from src.caching.query_cache import get_cache_stats

    return await get_cache_stats()


@app.delete("/api/cache")
async def clear_cache_endpoint() -> dict[str, Any]:
    from src.caching.query_cache import clear_cache

    deleted = await clear_cache()
    return {"status": "cleared", "entries_removed": deleted}


@app.get("/api/retrieval/optimal-params")
async def optimal_params(collection: str = "documents") -> dict[str, Any]:
    from src.retrieval.auto_tune import get_param_analysis

    return await get_param_analysis(collection)


@app.get("/api/settings")
async def get_settings() -> dict[str, Any]:
    return {
        "embedding_model": settings.embedding_model,
        "chunking_strategy": settings.chunking_strategy,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "default_model": settings.default_model,
        "hybrid_alpha": settings.hybrid_alpha,
        "default_top_k": settings.default_top_k,
        "eval_on_query": settings.eval_on_query,
        "eval_lightweight": settings.eval_lightweight,
        "use_reranker": settings.use_reranker,
        "cache_enabled": settings.cache_enabled,
        "cache_threshold": settings.cache_threshold,
        "cache_ttl_seconds": settings.cache_ttl_seconds,
    }
