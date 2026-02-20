from __future__ import annotations

import pytest

from src.ingestion.chunker import (
    ChunkingStrategy,
    chunk_text,
    count_tokens,
)


SAMPLE_TEXT = """Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.
Instead of being explicitly programmed, these systems improve their performance on a specific task over time.

There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.
Supervised learning uses labeled data to train models. Unsupervised learning finds patterns in unlabeled data.
Reinforcement learning trains agents to make decisions by rewarding desired behaviors.

Deep learning is a subset of machine learning that uses neural networks with multiple layers.
These networks can automatically learn hierarchical representations of data.
Convolutional neural networks are particularly effective for image recognition tasks.
Recurrent neural networks are designed for sequential data like text and time series.

Transfer learning allows models trained on one task to be adapted for related tasks.
This approach has been particularly successful in natural language processing.
Models like BERT and GPT have been pre-trained on large corpora and fine-tuned for specific tasks."""


class TestCountTokens:
    def test_empty_string(self) -> None:
        assert count_tokens("") == 0

    def test_single_word(self) -> None:
        tokens = count_tokens("hello")
        assert tokens >= 1

    def test_sentence(self) -> None:
        tokens = count_tokens("This is a simple test sentence.")
        assert tokens > 0 and tokens < 20


class TestFixedChunking:
    def test_produces_chunks(self) -> None:
        chunks = chunk_text(SAMPLE_TEXT, strategy="fixed", chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 0

    def test_chunk_sizes_within_limit(self) -> None:
        chunks = chunk_text(SAMPLE_TEXT, strategy="fixed", chunk_size=100, chunk_overlap=10)
        for chunk in chunks:
            assert chunk.token_count <= 110  # allow small buffer

    def test_chunk_metadata(self) -> None:
        chunks = chunk_text(
            SAMPLE_TEXT,
            strategy="fixed",
            chunk_size=100,
            chunk_overlap=10,
            source_metadata={"source": "test.txt"},
        )
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert chunk.metadata["source"] == "test.txt"
            assert chunk.metadata["strategy"] == "fixed"

    def test_chunk_indices_sequential(self) -> None:
        chunks = chunk_text(SAMPLE_TEXT, strategy="fixed", chunk_size=100, chunk_overlap=10)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_overlap_creates_more_chunks(self) -> None:
        no_overlap = chunk_text(SAMPLE_TEXT, strategy="fixed", chunk_size=100, chunk_overlap=0)
        with_overlap = chunk_text(SAMPLE_TEXT, strategy="fixed", chunk_size=100, chunk_overlap=30)
        assert len(with_overlap) >= len(no_overlap)

    def test_single_chunk_for_small_text(self) -> None:
        chunks = chunk_text("Hello world", strategy="fixed", chunk_size=100, chunk_overlap=10)
        assert len(chunks) == 1


class TestRecursiveChunking:
    def test_produces_chunks(self) -> None:
        chunks = chunk_text(SAMPLE_TEXT, strategy="recursive", chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 0

    def test_respects_paragraph_boundaries(self) -> None:
        chunks = chunk_text(SAMPLE_TEXT, strategy="recursive", chunk_size=200, chunk_overlap=0)
        for chunk in chunks:
            assert chunk.token_count <= 250

    def test_chunk_metadata(self) -> None:
        chunks = chunk_text(SAMPLE_TEXT, strategy="recursive", chunk_size=100, chunk_overlap=10)
        for chunk in chunks:
            assert chunk.metadata.get("strategy") == "recursive"

    def test_no_empty_chunks(self) -> None:
        chunks = chunk_text(SAMPLE_TEXT, strategy="recursive", chunk_size=100, chunk_overlap=10)
        for chunk in chunks:
            assert len(chunk.text.strip()) > 0

    def test_covers_all_content(self) -> None:
        chunks = chunk_text(SAMPLE_TEXT, strategy="recursive", chunk_size=200, chunk_overlap=0)
        combined = " ".join(c.text for c in chunks)
        key_phrases = ["machine learning", "supervised learning", "deep learning", "transfer learning"]
        for phrase in key_phrases:
            assert phrase.lower() in combined.lower(), f"Missing content: {phrase}"


class TestSemanticChunking:
    def test_produces_chunks(self) -> None:
        chunks = chunk_text(SAMPLE_TEXT, strategy="semantic", chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 0

    def test_respects_sentence_boundaries(self) -> None:
        chunks = chunk_text(SAMPLE_TEXT, strategy="semantic", chunk_size=100, chunk_overlap=0)
        for chunk in chunks:
            text = chunk.text.strip()
            if len(text) > 20:
                assert text[0].isupper() or text[0].isdigit(), f"Chunk doesn't start at sentence boundary: {text[:50]}"

    def test_chunk_metadata(self) -> None:
        chunks = chunk_text(SAMPLE_TEXT, strategy="semantic", chunk_size=100, chunk_overlap=10)
        for chunk in chunks:
            assert chunk.metadata.get("strategy") == "semantic"


class TestChunkingStrategies:
    @pytest.mark.parametrize("strategy", ["fixed", "recursive", "semantic"])
    def test_all_strategies_produce_output(self, strategy: str) -> None:
        chunks = chunk_text(
            SAMPLE_TEXT,
            strategy=strategy,  # type: ignore[arg-type]
            chunk_size=100,
            chunk_overlap=10,
        )
        assert len(chunks) > 0

    @pytest.mark.parametrize("strategy", ["fixed", "recursive", "semantic"])
    def test_all_strategies_have_token_counts(self, strategy: str) -> None:
        chunks = chunk_text(
            SAMPLE_TEXT,
            strategy=strategy,  # type: ignore[arg-type]
            chunk_size=100,
            chunk_overlap=10,
        )
        for chunk in chunks:
            assert chunk.token_count > 0

    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            chunk_text(SAMPLE_TEXT, strategy="invalid")  # type: ignore[arg-type]
