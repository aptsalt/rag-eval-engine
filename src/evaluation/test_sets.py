from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from src.db.models import get_db
from src.generation.llm_client import generate

logger = logging.getLogger(__name__)


@dataclass
class TestQuestion:
    question: str
    ground_truth: str | None = None


async def create_test_set(
    name: str,
    collection: str,
    questions: list[dict[str, str | None]],
) -> dict[str, Any]:
    test_set_id = str(uuid.uuid4())
    now = time.time()

    db = await get_db()
    try:
        await db.execute(
            """INSERT INTO test_sets (id, name, collection, questions, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (test_set_id, name, collection, json.dumps(questions), now, now),
        )
        await db.commit()
    finally:
        await db.close()

    return {
        "id": test_set_id,
        "name": name,
        "collection": collection,
        "question_count": len(questions),
    }


async def get_test_set(test_set_id: str) -> dict[str, Any] | None:
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM test_sets WHERE id = ?", (test_set_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        data = dict(row)  # type: ignore[arg-type]
        data["questions"] = json.loads(data["questions"])
        return data
    finally:
        await db.close()


async def list_test_sets() -> list[dict[str, Any]]:
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id, name, collection, created_at, updated_at"
            " FROM test_sets ORDER BY created_at DESC"
        )
        rows = await cursor.fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            data = dict(row)  # type: ignore[arg-type]
            result.append(data)
        return result
    finally:
        await db.close()


async def delete_test_set(test_set_id: str) -> bool:
    db = await get_db()
    try:
        cursor = await db.execute(
            "DELETE FROM test_sets WHERE id = ?", (test_set_id,)
        )
        await db.commit()
        return cursor.rowcount > 0
    finally:
        await db.close()


async def auto_generate_questions(
    collection: str,
    num_questions: int = 10,
    model: str | None = None,
) -> list[dict[str, str | None]]:
    from src.ingestion.embedder import get_qdrant_client

    client = get_qdrant_client()

    try:
        points = client.scroll(
            collection_name=collection,
            limit=min(20, num_questions * 2),
            with_payload=True,
        )
    except Exception:
        return []

    chunks = []
    for point in points[0]:
        payload = point.payload or {}
        text = str(payload.get("text", ""))
        if text:
            chunks.append(text)

    if not chunks:
        return []

    context = "\n\n---\n\n".join(chunks[:10])
    prompt = (
        f"Based on the following document excerpts, generate"
        f" {num_questions} diverse questions that could be"
        f" answered using this content.\n\n"
        f"Document Excerpts:\n{context}\n\n"
        f"Generate exactly {num_questions} questions. For each"
        f" question, also provide the expected answer based on"
        f" the content.\n\n"
        "Format your response as a JSON array like this:\n"
        "[\n"
        '  {"question": "What is ...", '
        '"ground_truth": "The answer is ..."},\n'
        "  ...\n"
        "]\n\n"
        "Respond with ONLY the JSON array, no other text."
    )

    try:
        response = await generate(
            [{"role": "user", "content": prompt}],
            model=model,
        )

        content = response.content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])

        questions: list[dict[str, str | None]] = json.loads(content)
        return questions[:num_questions]
    except Exception as exc:
        logger.error(f"Auto-generate questions failed: {exc}")
        return []


async def get_eval_runs(test_set_id: str | None = None) -> list[dict[str, Any]]:
    db = await get_db()
    try:
        if test_set_id:
            cursor = await db.execute(
                """SELECT id, test_set_id, status, avg_faithfulness, avg_relevance,
                          avg_hallucination_rate, avg_context_precision,
                          created_at, completed_at
                   FROM eval_runs WHERE test_set_id = ?
                   ORDER BY created_at DESC""",
                (test_set_id,),
            )
        else:
            cursor = await db.execute(
                """SELECT id, test_set_id, status, avg_faithfulness, avg_relevance,
                          avg_hallucination_rate, avg_context_precision,
                          created_at, completed_at
                   FROM eval_runs ORDER BY created_at DESC"""
            )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]  # type: ignore[arg-type]
    finally:
        await db.close()
