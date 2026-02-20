from __future__ import annotations

import pytest

from src.evaluation.metrics import (
    _heuristic_faithfulness,
    _heuristic_hallucination,
    _heuristic_relevance,
    _parse_score,
    compute_context_precision,
)


class TestParseScore:
    def test_simple_float(self) -> None:
        assert _parse_score("0.85") == 0.85

    def test_float_in_text(self) -> None:
        score = _parse_score("The faithfulness score is 0.75 based on the analysis.")
        assert score == 0.75

    def test_integer(self) -> None:
        assert _parse_score("1") == 1.0

    def test_clamps_above_one(self) -> None:
        assert _parse_score("5.0") == 1.0

    def test_no_number_returns_default(self) -> None:
        assert _parse_score("no number here") == 0.5

    def test_empty_string(self) -> None:
        assert _parse_score("") == 0.5

    def test_zero(self) -> None:
        assert _parse_score("0.0") == 0.0

    def test_multiple_numbers_takes_first(self) -> None:
        score = _parse_score("Score: 0.8 out of 1.0")
        assert score == 0.8


class TestHeuristicFaithfulness:
    def test_high_overlap(self) -> None:
        context = ["Machine learning is a subset of artificial intelligence"]
        answer = "Machine learning is a subset of artificial intelligence"
        score = _heuristic_faithfulness(answer, context)
        assert score > 0.8

    def test_no_overlap(self) -> None:
        context = ["The weather is sunny today"]
        answer = "Quantum computing uses qubits for processing"
        score = _heuristic_faithfulness(answer, context)
        assert score < 0.5

    def test_empty_context(self) -> None:
        assert _heuristic_faithfulness("some answer", []) == 0.0

    def test_empty_answer(self) -> None:
        assert _heuristic_faithfulness("", ["some context"]) == 0.0


class TestHeuristicRelevance:
    def test_relevant_answer(self) -> None:
        query = "What is machine learning?"
        answer = "Machine learning is a field of AI that enables systems to learn from data"
        score = _heuristic_relevance(query, answer)
        assert score >= 0.5

    def test_irrelevant_answer(self) -> None:
        query = "What is machine learning?"
        answer = "The sun rises in the east and sets in the west"
        score = _heuristic_relevance(query, answer)
        assert score < 0.5

    def test_empty_query(self) -> None:
        assert _heuristic_relevance("", "some answer") == 0.0


class TestHeuristicHallucination:
    def test_no_hallucination(self) -> None:
        context = ["Python is a programming language used for web development"]
        answer = "Python is a programming language"
        score = _heuristic_hallucination(answer, context)
        assert score < 0.5

    def test_high_hallucination(self) -> None:
        context = ["The weather is sunny"]
        answer = "Quantum computing uses superposition and entanglement principles"
        score = _heuristic_hallucination(answer, context)
        assert score > 0.5


class TestContextPrecision:
    def test_all_relevant(self) -> None:
        query = "machine learning AI"
        chunks = [
            "Machine learning is a type of AI technology",
            "AI and machine learning are transforming industries",
        ]
        score = compute_context_precision(query, chunks)
        assert score > 0.5

    def test_none_relevant(self) -> None:
        query = "quantum physics"
        chunks = [
            "The recipe calls for two cups of flour",
            "Basketball is played with five players per team",
        ]
        score = compute_context_precision(query, chunks)
        assert score < 0.5

    def test_empty_chunks(self) -> None:
        assert compute_context_precision("test query", []) == 0.0

    def test_with_explicit_indices(self) -> None:
        query = "test"
        chunks = ["a", "b", "c", "d"]
        score = compute_context_precision(query, chunks, relevant_indices=[0, 2])
        assert score == 0.5
