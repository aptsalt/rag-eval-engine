from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import aiosqlite

from src.config import settings

DB_PATH = Path(settings.db_path)


async def get_db() -> aiosqlite.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(DB_PATH))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    return db


async def init_db() -> None:
    db = await get_db()
    try:
        await db.executescript(SCHEMA)
        await db.commit()
    finally:
        await db.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    collection TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    token_count INTEGER NOT NULL DEFAULT 0,
    ingested_at REAL NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection);

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id TEXT PRIMARY KEY,
    collection TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    total_files INTEGER NOT NULL DEFAULT 0,
    processed_files INTEGER NOT NULL DEFAULT 0,
    total_chunks INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at REAL NOT NULL,
    completed_at REAL
);

CREATE TABLE IF NOT EXISTS query_log (
    id TEXT PRIMARY KEY,
    collection TEXT NOT NULL,
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources TEXT NOT NULL DEFAULT '[]',
    model TEXT NOT NULL,
    tokens_used INTEGER NOT NULL DEFAULT 0,
    latency_ms REAL NOT NULL DEFAULT 0,
    latency_retrieval_ms REAL NOT NULL DEFAULT 0,
    latency_generation_ms REAL NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_query_log_collection ON query_log(collection);
CREATE INDEX IF NOT EXISTS idx_query_log_created ON query_log(created_at);

CREATE TABLE IF NOT EXISTS eval_results (
    id TEXT PRIMARY KEY,
    query_id TEXT NOT NULL REFERENCES query_log(id),
    faithfulness REAL,
    relevance REAL,
    hallucination_rate REAL,
    context_precision REAL,
    context_recall REAL,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_eval_results_query ON eval_results(query_id);
CREATE INDEX IF NOT EXISTS idx_eval_results_created ON eval_results(created_at);

CREATE TABLE IF NOT EXISTS test_sets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    collection TEXT NOT NULL,
    questions TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id TEXT PRIMARY KEY,
    test_set_id TEXT NOT NULL REFERENCES test_sets(id),
    status TEXT NOT NULL DEFAULT 'pending',
    results TEXT NOT NULL DEFAULT '[]',
    avg_faithfulness REAL,
    avg_relevance REAL,
    avg_hallucination_rate REAL,
    avg_context_precision REAL,
    created_at REAL NOT NULL,
    completed_at REAL
);
"""


async def insert_document(
    doc_id: str,
    collection: str,
    filename: str,
    file_type: str,
    chunk_count: int,
    token_count: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    db = await get_db()
    try:
        await db.execute(
            """INSERT OR REPLACE INTO documents
            (id, collection, filename, file_type, chunk_count, token_count, ingested_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                doc_id,
                collection,
                filename,
                file_type,
                chunk_count,
                token_count,
                time.time(),
                json.dumps(metadata or {}),
            ),
        )
        await db.commit()
    finally:
        await db.close()


async def insert_ingestion_job(
    job_id: str, collection: str, total_files: int
) -> None:
    db = await get_db()
    try:
        await db.execute(
            """INSERT INTO ingestion_jobs
            (id, collection, status, total_files, created_at)
            VALUES (?, ?, 'processing', ?, ?)""",
            (job_id, collection, total_files, time.time()),
        )
        await db.commit()
    finally:
        await db.close()


async def update_ingestion_job(
    job_id: str,
    *,
    status: str | None = None,
    processed_files: int | None = None,
    total_chunks: int | None = None,
    error: str | None = None,
) -> None:
    db = await get_db()
    try:
        updates: list[str] = []
        values: list[Any] = []
        if status is not None:
            updates.append("status = ?")
            values.append(status)
            if status in ("completed", "failed"):
                updates.append("completed_at = ?")
                values.append(time.time())
        if processed_files is not None:
            updates.append("processed_files = ?")
            values.append(processed_files)
        if total_chunks is not None:
            updates.append("total_chunks = ?")
            values.append(total_chunks)
        if error is not None:
            updates.append("error = ?")
            values.append(error)
        if updates:
            values.append(job_id)
            await db.execute(
                f"UPDATE ingestion_jobs SET {', '.join(updates)} WHERE id = ?",
                values,
            )
            await db.commit()
    finally:
        await db.close()


async def get_ingestion_job(job_id: str) -> dict[str, Any] | None:
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM ingestion_jobs WHERE id = ?", (job_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return dict(row)  # type: ignore[arg-type]
    finally:
        await db.close()


async def insert_query_log(
    query_id: str,
    collection: str,
    query: str,
    answer: str,
    sources: list[dict[str, Any]],
    model: str,
    tokens_used: int,
    latency_ms: float,
    latency_retrieval_ms: float,
    latency_generation_ms: float,
) -> None:
    db = await get_db()
    try:
        await db.execute(
            """INSERT INTO query_log
            (id, collection, query, answer, sources, model, tokens_used,
             latency_ms, latency_retrieval_ms, latency_generation_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                query_id,
                collection,
                query,
                answer,
                json.dumps(sources),
                model,
                tokens_used,
                latency_ms,
                latency_retrieval_ms,
                latency_generation_ms,
                time.time(),
            ),
        )
        await db.commit()
    finally:
        await db.close()


async def insert_eval_result(
    eval_id: str,
    query_id: str,
    faithfulness: float | None,
    relevance: float | None,
    hallucination_rate: float | None,
    context_precision: float | None,
    context_recall: float | None = None,
) -> None:
    db = await get_db()
    try:
        await db.execute(
            """INSERT INTO eval_results
            (id, query_id, faithfulness, relevance, hallucination_rate,
             context_precision, context_recall, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                eval_id,
                query_id,
                faithfulness,
                relevance,
                hallucination_rate,
                context_precision,
                context_recall,
                time.time(),
            ),
        )
        await db.commit()
    finally:
        await db.close()


async def get_metrics(
    collection: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    db = await get_db()
    try:
        query = """
            SELECT q.id, q.collection, q.query, q.latency_ms,
                   q.latency_retrieval_ms, q.latency_generation_ms,
                   q.tokens_used, q.created_at,
                   e.faithfulness, e.relevance, e.hallucination_rate,
                   e.context_precision, e.context_recall
            FROM query_log q
            LEFT JOIN eval_results e ON e.query_id = q.id
        """
        params: list[Any] = []
        if collection:
            query += " WHERE q.collection = ?"
            params.append(collection)
        query += " ORDER BY q.created_at DESC LIMIT ?"
        params.append(limit)
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]  # type: ignore[arg-type]
    finally:
        await db.close()


async def get_collections() -> list[dict[str, Any]]:
    db = await get_db()
    try:
        cursor = await db.execute(
            """SELECT collection, COUNT(*) as doc_count,
                      SUM(chunk_count) as total_chunks,
                      SUM(token_count) as total_tokens
               FROM documents
               GROUP BY collection
               ORDER BY collection"""
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]  # type: ignore[arg-type]
    finally:
        await db.close()


async def delete_collection_docs(collection: str) -> int:
    db = await get_db()
    try:
        cursor = await db.execute(
            "DELETE FROM documents WHERE collection = ?", (collection,)
        )
        await db.commit()
        return cursor.rowcount
    finally:
        await db.close()
