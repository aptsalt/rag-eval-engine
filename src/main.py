from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
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
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3500"],
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
    }
