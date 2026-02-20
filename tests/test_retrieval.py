from __future__ import annotations

import pytest

from src.retrieval.hybrid_ranker import RankedResult, reciprocal_rank_fusion
from src.retrieval.sparse_search import BM25Index, SparseResult, _tokenize
from src.retrieval.vector_search import SearchResult


class TestBM25Index:
    def test_build_and_search(self) -> None:
        index = BM25Index(
            collection_name="test",
            documents=[
                "Machine learning is a subset of artificial intelligence",
                "Deep learning uses neural networks",
                "Natural language processing handles text data",
                "Computer vision processes images and videos",
            ],
            metadata_list=[
                {"chunk_index": 0},
                {"chunk_index": 1},
                {"chunk_index": 2},
                {"chunk_index": 3},
            ],
        )
        index.build()

        results = index.search("neural networks deep learning", top_k=2)
        assert len(results) > 0
        assert results[0].score > 0

    def test_empty_index_search(self) -> None:
        index = BM25Index(
            collection_name="test",
            documents=[],
            metadata_list=[],
        )
        index.build()
        results = index.search("test query")
        assert len(results) == 0

    def test_add_documents(self) -> None:
        index = BM25Index(
            collection_name="test",
            documents=["Initial document about AI and technology"],
            metadata_list=[{"chunk_index": 0}],
        )
        index.build()

        index.add_documents(
            ["New document about machine learning and neural networks for AI"],
            [{"chunk_index": 1}],
        )

        assert len(index.documents) == 2
        assert index.bm25 is not None

    def test_top_k_limits_results(self) -> None:
        docs = [f"Document number {i} about topic {i}" for i in range(10)]
        meta = [{"chunk_index": i} for i in range(10)]
        index = BM25Index(
            collection_name="test", documents=docs, metadata_list=meta
        )
        index.build()

        results = index.search("document topic", top_k=3)
        assert len(results) <= 3


class TestTokenize:
    def test_basic_tokenization(self) -> None:
        tokens = _tokenize("Hello World Test")
        assert tokens == ["hello", "world", "test"]

    def test_removes_punctuation(self) -> None:
        tokens = _tokenize("Hello, World! How are you?")
        assert "hello" in tokens
        assert "," not in tokens

    def test_removes_single_chars(self) -> None:
        tokens = _tokenize("I am a big fan")
        assert "i" not in tokens
        assert "a" not in tokens
        assert "am" in tokens
        assert "big" in tokens

    def test_lowercases(self) -> None:
        tokens = _tokenize("Machine Learning AI")
        assert all(t == t.lower() for t in tokens)


class TestReciprocalRankFusion:
    def test_basic_fusion(self) -> None:
        vector_results = [
            SearchResult(text="doc1", score=0.95, chunk_index=0, metadata={}),
            SearchResult(text="doc2", score=0.85, chunk_index=1, metadata={}),
            SearchResult(text="doc3", score=0.75, chunk_index=2, metadata={}),
        ]
        sparse_results = [
            SparseResult(text="doc2", score=5.0, chunk_index=1, metadata={}),
            SparseResult(text="doc1", score=4.0, chunk_index=0, metadata={}),
            SparseResult(text="doc4", score=3.0, chunk_index=3, metadata={}),
        ]

        fused = reciprocal_rank_fusion(
            vector_results, sparse_results, alpha=0.5, top_k=5
        )
        assert len(fused) > 0
        assert all(isinstance(r, RankedResult) for r in fused)

    def test_alpha_weighting(self) -> None:
        vector_results = [
            SearchResult(text="vector_top", score=0.99, chunk_index=0, metadata={}),
        ]
        sparse_results = [
            SparseResult(text="sparse_top", score=10.0, chunk_index=1, metadata={}),
        ]

        vector_heavy = reciprocal_rank_fusion(
            vector_results, sparse_results, alpha=1.0, top_k=2
        )
        sparse_heavy = reciprocal_rank_fusion(
            vector_results, sparse_results, alpha=0.0, top_k=2
        )

        assert vector_heavy[0].text == "vector_top"
        assert sparse_heavy[0].text == "sparse_top"

    def test_deduplication(self) -> None:
        vector_results = [
            SearchResult(text="same document content", score=0.9, chunk_index=0, metadata={}),
        ]
        sparse_results = [
            SparseResult(text="same document content", score=5.0, chunk_index=0, metadata={}),
        ]

        fused = reciprocal_rank_fusion(
            vector_results, sparse_results, alpha=0.5, top_k=5
        )
        assert len(fused) == 1
        assert fused[0].vector_score > 0
        assert fused[0].sparse_score > 0

    def test_top_k_limits(self) -> None:
        vector_results = [
            SearchResult(text=f"doc{i}", score=0.9 - i * 0.1, chunk_index=i, metadata={})
            for i in range(10)
        ]
        fused = reciprocal_rank_fusion(
            vector_results, [], alpha=0.7, top_k=3
        )
        assert len(fused) == 3

    def test_scores_are_descending(self) -> None:
        vector_results = [
            SearchResult(text=f"doc{i}", score=0.9 - i * 0.1, chunk_index=i, metadata={})
            for i in range(5)
        ]
        sparse_results = [
            SparseResult(text=f"sdoc{i}", score=5.0 - i, chunk_index=i + 10, metadata={})
            for i in range(5)
        ]

        fused = reciprocal_rank_fusion(
            vector_results, sparse_results, alpha=0.5, top_k=10
        )
        for i in range(len(fused) - 1):
            assert fused[i].score >= fused[i + 1].score

    def test_empty_inputs(self) -> None:
        fused = reciprocal_rank_fusion([], [], alpha=0.5, top_k=5)
        assert len(fused) == 0

    def test_preserves_metadata(self) -> None:
        vector_results = [
            SearchResult(
                text="doc1", score=0.9, chunk_index=0,
                metadata={"source": "test.pdf", "page": 1},
            ),
        ]

        fused = reciprocal_rank_fusion(
            vector_results, [], alpha=0.7, top_k=5
        )
        assert len(fused) == 1
        assert fused[0].metadata.get("source") == "test.pdf"
