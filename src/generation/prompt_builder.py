from __future__ import annotations

from src.config import settings
from src.ingestion.chunker import count_tokens
from src.retrieval.hybrid_ranker import RankedResult

SYSTEM_PROMPT = (
    "You are a precise, helpful assistant that answers"
    " questions based ONLY on the provided context.\n\n"
    "Rules:\n"
    "1. Only use information from the provided context"
    " to answer.\n"
    "2. If the context doesn't contain enough information,"
    ' say "I don\'t have enough information to answer this'
    ' question based on the provided documents."\n'
    "3. Cite your sources by referencing [Source N] where N"
    " corresponds to the context chunk number.\n"
    "4. Never make up or hallucinate information.\n"
    "5. Be concise and direct in your answers.\n"
    "6. If multiple sources support your answer, cite all"
    " relevant ones."
)


def build_prompt(
    query: str,
    results: list[RankedResult],
    max_context_tokens: int | None = None,
) -> tuple[str, str, list[dict[str, str | int | float]]]:
    max_tokens = max_context_tokens or settings.max_context_tokens
    token_budget = max_tokens - count_tokens(SYSTEM_PROMPT) - count_tokens(query) - 200

    context_parts: list[str] = []
    sources: list[dict[str, str | int | float]] = []
    used_tokens = 0

    for i, result in enumerate(results):
        chunk_tokens = count_tokens(result.text)
        if used_tokens + chunk_tokens > token_budget:
            break

        source_label = f"[Source {i + 1}]"
        source_info = str(result.metadata.get("source", f"chunk_{result.chunk_index}"))
        page_info = result.metadata.get("page", "")
        source_detail = f"{source_info}"
        if page_info:
            source_detail += f" (page {page_info})"

        context_parts.append(f"{source_label} ({source_detail}):\n{result.text}")
        sources.append({
            "index": i + 1,
            "text": result.text[:200] + "..." if len(result.text) > 200 else result.text,
            "source": source_info,
            "score": result.score,
            "chunk_index": result.chunk_index,
        })
        used_tokens += chunk_tokens

    context_block = "\n\n---\n\n".join(context_parts)

    user_prompt = f"""Context:
{context_block}

Question: {query}

Answer the question based only on the context above. Cite sources using [Source N] notation."""

    return SYSTEM_PROMPT, user_prompt, sources


def format_messages(
    system: str, user: str
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
