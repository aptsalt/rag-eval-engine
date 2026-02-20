from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from src.config import settings

logger = logging.getLogger(__name__)

INDEX_DIR = Path("data/bm25_indices")


@dataclass
class SparseResult:
    text: str
    score: float
    chunk_index: int
    metadata: dict[str, str | int | float] = field(default_factory=dict)


@dataclass
class BM25Index:
    collection_name: str
    documents: list[str]
    metadata_list: list[dict[str, str | int | float]]
    bm25: BM25Okapi | None = None

    def build(self) -> None:
        if not self.documents:
            self.bm25 = None
            return
        tokenized = [_tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int | None = None) -> list[SparseResult]:
        if not self.documents:
            return []
        if self.bm25 is None:
            self.build()
        if self.bm25 is None:
            return []

        k = top_k or settings.default_top_k
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        scored_indices = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:k]

        results: list[SparseResult] = []
        for idx, score in scored_indices:
            if score <= 0:
                continue
            meta = self.metadata_list[idx] if idx < len(self.metadata_list) else {}
            results.append(SparseResult(
                text=self.documents[idx],
                score=float(score),
                chunk_index=int(meta.get("chunk_index", idx)),
                metadata=meta,
            ))

        return results

    def save(self) -> None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        index_path = INDEX_DIR / f"{self.collection_name}.json"
        data = {
            "collection_name": self.collection_name,
            "documents": self.documents,
            "metadata_list": self.metadata_list,
        }
        index_path.write_text(json.dumps(data), encoding="utf-8")
        logger.info(f"Saved BM25 index: {index_path} ({len(self.documents)} docs)")

    @classmethod
    def load(cls, collection_name: str) -> BM25Index | None:
        index_path = INDEX_DIR / f"{collection_name}.json"
        if not index_path.exists():
            return None
        data = json.loads(index_path.read_text(encoding="utf-8"))
        index = cls(
            collection_name=data["collection_name"],
            documents=data["documents"],
            metadata_list=data["metadata_list"],
        )
        index.build()
        return index

    def add_documents(
        self,
        documents: list[str],
        metadata_list: list[dict[str, str | int | float]],
    ) -> None:
        self.documents.extend(documents)
        self.metadata_list.extend(metadata_list)
        self.build()


_bm25_cache: dict[str, BM25Index] = {}


def get_or_create_index(collection_name: str) -> BM25Index:
    if collection_name in _bm25_cache:
        return _bm25_cache[collection_name]

    loaded = BM25Index.load(collection_name)
    if loaded:
        _bm25_cache[collection_name] = loaded
        return loaded

    index = BM25Index(
        collection_name=collection_name,
        documents=[],
        metadata_list=[],
    )
    _bm25_cache[collection_name] = index
    return index


def add_to_index(
    collection_name: str,
    texts: list[str],
    metadata_list: list[dict[str, str | int | float]],
) -> None:
    index = get_or_create_index(collection_name)
    index.add_documents(texts, metadata_list)
    index.save()


def sparse_search(
    query: str,
    collection_name: str,
    top_k: int | None = None,
) -> list[SparseResult]:
    index = get_or_create_index(collection_name)
    if not index.documents:
        return []
    return index.search(query, top_k)


def delete_index(collection_name: str) -> None:
    index_path = INDEX_DIR / f"{collection_name}.json"
    if index_path.exists():
        index_path.unlink()
    _bm25_cache.pop(collection_name, None)


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]
