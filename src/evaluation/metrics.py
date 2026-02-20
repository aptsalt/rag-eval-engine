from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.generation.llm_client import generate

logger = logging.getLogger(__name__)


@dataclass
class EvalScores:
    faithfulness: float
    relevance: float
    hallucination_rate: float
    context_precision: float
    context_recall: float | None = None
    latency_retrieval_ms: float = 0.0
    latency_generation_ms: float = 0.0


async def compute_faithfulness(
    query: str,
    answer: str,
    context_chunks: list[str],
    model: str | None = None,
) -> float:
    if not answer.strip() or not context_chunks:
        return 0.0

    context = "\n\n".join(context_chunks)
    prompt = (
        "You are an evaluation judge. Assess whether the answer"
        " is faithful to the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer: {answer}\n\n"
        "Rate the faithfulness of the answer on a scale of 0.0 to 1.0:\n"
        "- 1.0 = Every claim in the answer is directly supported"
        " by the context\n"
        "- 0.5 = Some claims are supported, some are not"
        " verifiable from context\n"
        "- 0.0 = The answer contradicts or fabricates information"
        " not in the context\n\n"
        "Respond with ONLY a number between 0.0 and 1.0."
    )

    try:
        response = await generate(
            [{"role": "user", "content": prompt}],
            model=model,
        )
        return _parse_score(response.content)
    except Exception as exc:
        logger.warning(f"Faithfulness eval failed: {exc}")
        return _heuristic_faithfulness(answer, context_chunks)


async def compute_relevance(
    query: str,
    answer: str,
    model: str | None = None,
) -> float:
    if not answer.strip():
        return 0.0

    prompt = f"""You are an evaluation judge. Assess whether the answer is relevant to the question.

Question: {query}

Answer: {answer}

Rate the relevance of the answer on a scale of 0.0 to 1.0:
- 1.0 = The answer directly and completely addresses the question
- 0.5 = The answer partially addresses the question
- 0.0 = The answer is completely irrelevant to the question

Respond with ONLY a number between 0.0 and 1.0."""

    try:
        response = await generate(
            [{"role": "user", "content": prompt}],
            model=model,
        )
        return _parse_score(response.content)
    except Exception as exc:
        logger.warning(f"Relevance eval failed: {exc}")
        return _heuristic_relevance(query, answer)


async def compute_hallucination_rate(
    answer: str,
    context_chunks: list[str],
    model: str | None = None,
) -> float:
    if not answer.strip():
        return 0.0

    context = "\n\n".join(context_chunks)
    prompt = (
        "You are an evaluation judge. Identify sentences in the"
        " answer that are NOT supported by the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Answer: {answer}\n\n"
        "For each sentence in the answer, determine if it is"
        " grounded in the context.\n"
        "Count the total number of factual claim sentences and"
        " how many are NOT grounded.\n\n"
        "Respond with ONLY a number between 0.0 and 1.0"
        " representing the hallucination rate:\n"
        "- 0.0 = No hallucination (all claims grounded in"
        " context)\n"
        "- 1.0 = Complete hallucination (no claims grounded in"
        " context)"
    )

    try:
        response = await generate(
            [{"role": "user", "content": prompt}],
            model=model,
        )
        return _parse_score(response.content)
    except Exception as exc:
        logger.warning(f"Hallucination eval failed: {exc}")
        return _heuristic_hallucination(answer, context_chunks)


def compute_context_precision(
    query: str,
    context_chunks: list[str],
    relevant_indices: list[int] | None = None,
) -> float:
    if not context_chunks:
        return 0.0

    if relevant_indices is not None:
        relevant_count = len(relevant_indices)
        return relevant_count / len(context_chunks) if context_chunks else 0.0

    query_terms = set(query.lower().split())
    relevant = 0
    for chunk in context_chunks:
        chunk_terms = set(chunk.lower().split())
        overlap = len(query_terms & chunk_terms)
        if overlap >= max(1, len(query_terms) * 0.2):
            relevant += 1

    return relevant / len(context_chunks)


async def compute_context_recall(
    answer: str,
    ground_truth: str,
    context_chunks: list[str],
    model: str | None = None,
) -> float:
    if not ground_truth:
        return 0.0

    context = "\n\n".join(context_chunks)
    prompt = (
        "You are an evaluation judge. Determine what fraction"
        " of the ground truth answer can be attributed to the"
        " retrieved context.\n\n"
        f"Ground Truth Answer: {ground_truth}\n\n"
        f"Retrieved Context:\n{context}\n\n"
        "Rate the context recall on a scale of 0.0 to 1.0:\n"
        "- 1.0 = All information in the ground truth is present"
        " in the context\n"
        "- 0.5 = About half the ground truth information is in"
        " the context\n"
        "- 0.0 = None of the ground truth information is in the"
        " context\n\n"
        "Respond with ONLY a number between 0.0 and 1.0."
    )

    try:
        response = await generate(
            [{"role": "user", "content": prompt}],
            model=model,
        )
        return _parse_score(response.content)
    except Exception as exc:
        logger.warning(f"Context recall eval failed: {exc}")
        return 0.5


async def evaluate_query(
    query: str,
    answer: str,
    context_chunks: list[str],
    ground_truth: str | None = None,
    model: str | None = None,
    lightweight: bool = True,
    latency_retrieval_ms: float = 0.0,
    latency_generation_ms: float = 0.0,
) -> EvalScores:
    faithfulness = await compute_faithfulness(query, answer, context_chunks, model)
    relevance = await compute_relevance(query, answer, model)

    hallucination_rate = 0.0
    context_precision = 0.0
    context_recall = None

    if not lightweight:
        hallucination_rate = await compute_hallucination_rate(answer, context_chunks, model)
        context_precision = compute_context_precision(query, context_chunks)

        if ground_truth:
            context_recall = await compute_context_recall(
                answer, ground_truth, context_chunks, model
            )

    return EvalScores(
        faithfulness=faithfulness,
        relevance=relevance,
        hallucination_rate=hallucination_rate,
        context_precision=context_precision,
        context_recall=context_recall,
        latency_retrieval_ms=latency_retrieval_ms,
        latency_generation_ms=latency_generation_ms,
    )


def _parse_score(text: str) -> float:
    numbers = re.findall(r"(\d+\.?\d*)", text.strip())
    if not numbers:
        return 0.5
    score = float(numbers[0])
    return max(0.0, min(1.0, score))


def _heuristic_faithfulness(answer: str, context_chunks: list[str]) -> float:
    if not context_chunks:
        return 0.0
    context_text = " ".join(context_chunks).lower()
    answer_words = set(answer.lower().split())
    context_words = set(context_text.split())
    if not answer_words:
        return 0.0
    overlap = len(answer_words & context_words)
    return min(1.0, overlap / len(answer_words))


def _heuristic_relevance(query: str, answer: str) -> float:
    query_words = set(query.lower().split())
    answer_words = set(answer.lower().split())
    if not query_words:
        return 0.0
    overlap = len(query_words & answer_words)
    return min(1.0, overlap / len(query_words))


def _heuristic_hallucination(answer: str, context_chunks: list[str]) -> float:
    faithfulness = _heuristic_faithfulness(answer, context_chunks)
    return max(0.0, 1.0 - faithfulness)
