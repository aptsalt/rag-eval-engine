from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import tiktoken


@dataclass
class Chunk:
    text: str
    chunk_index: int
    token_count: int
    metadata: dict[str, str | int | float] = field(default_factory=dict)


ChunkingStrategy = Literal["fixed", "recursive", "semantic"]

_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


def chunk_text(
    text: str,
    strategy: ChunkingStrategy = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    source_metadata: dict[str, str | int | float] | None = None,
) -> list[Chunk]:
    if strategy == "fixed":
        return _fixed_chunk(text, chunk_size, chunk_overlap, source_metadata)
    elif strategy == "recursive":
        return _recursive_chunk(text, chunk_size, chunk_overlap, source_metadata)
    elif strategy == "semantic":
        return _semantic_chunk(text, chunk_size, chunk_overlap, source_metadata)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")


def _fixed_chunk(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    source_metadata: dict[str, str | int | float] | None,
) -> list[Chunk]:
    tokens = _encoder.encode(text)
    chunks: list[Chunk] = []
    start = 0
    idx = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = _encoder.decode(chunk_tokens)

        meta = dict(source_metadata) if source_metadata else {}
        meta["chunk_index"] = idx
        meta["strategy"] = "fixed"

        chunks.append(Chunk(
            text=chunk_text.strip(),
            chunk_index=idx,
            token_count=len(chunk_tokens),
            metadata=meta,
        ))

        start += chunk_size - chunk_overlap
        idx += 1

    return chunks


def _recursive_chunk(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    source_metadata: dict[str, str | int | float] | None,
) -> list[Chunk]:
    separators = ["\n\n", "\n", ". ", " ", ""]
    raw_chunks = _recursive_split(text, separators, chunk_size)

    chunks: list[Chunk] = []
    for idx, raw in enumerate(raw_chunks):
        raw = raw.strip()
        if not raw:
            continue
        token_count = count_tokens(raw)
        meta = dict(source_metadata) if source_metadata else {}
        meta["chunk_index"] = idx
        meta["strategy"] = "recursive"
        chunks.append(Chunk(text=raw, chunk_index=idx, token_count=token_count, metadata=meta))

    if chunk_overlap > 0 and len(chunks) > 1:
        chunks = _apply_overlap(chunks, chunk_overlap, source_metadata)

    return chunks


def _recursive_split(
    text: str,
    separators: list[str],
    chunk_size: int,
) -> list[str]:
    if not text.strip():
        return []

    if count_tokens(text) <= chunk_size:
        return [text]

    if not separators:
        tokens = _encoder.encode(text)
        return [_encoder.decode(tokens[:chunk_size]), _encoder.decode(tokens[chunk_size:])]

    separator = separators[0]
    remaining_separators = separators[1:]

    if separator == "":
        tokens = _encoder.encode(text)
        result: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            result.append(_encoder.decode(tokens[start:end]))
            start = end
        return result

    parts = text.split(separator)
    result = []
    current = ""

    for part in parts:
        candidate = current + separator + part if current else part
        if count_tokens(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                result.append(current)
            if count_tokens(part) > chunk_size:
                result.extend(_recursive_split(part, remaining_separators, chunk_size))
                current = ""
            else:
                current = part

    if current:
        result.append(current)

    return result


def _semantic_chunk(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    source_metadata: dict[str, str | int | float] | None,
) -> list[Chunk]:
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_tokens = 0
    idx = 0

    for sentence in sentences:
        sent_tokens = count_tokens(sentence)

        if sent_tokens > chunk_size:
            if current_sentences:
                chunk_text = " ".join(current_sentences)
                meta = dict(source_metadata) if source_metadata else {}
                meta["chunk_index"] = idx
                meta["strategy"] = "semantic"
                chunks.append(Chunk(
                    text=chunk_text.strip(),
                    chunk_index=idx,
                    token_count=count_tokens(chunk_text),
                    metadata=meta,
                ))
                idx += 1
                current_sentences = []
                current_tokens = 0

            sub_chunks = _fixed_chunk(sentence, chunk_size, chunk_overlap, source_metadata)
            for sub in sub_chunks:
                sub.chunk_index = idx
                sub.metadata["strategy"] = "semantic"
                chunks.append(sub)
                idx += 1
            continue

        if current_tokens + sent_tokens > chunk_size and current_sentences:
            chunk_text = " ".join(current_sentences)
            meta = dict(source_metadata) if source_metadata else {}
            meta["chunk_index"] = idx
            meta["strategy"] = "semantic"
            chunks.append(Chunk(
                text=chunk_text.strip(),
                chunk_index=idx,
                token_count=current_tokens,
                metadata=meta,
            ))
            idx += 1

            if chunk_overlap > 0:
                overlap_sentences: list[str] = []
                overlap_tokens = 0
                for sent in reversed(current_sentences):
                    st = count_tokens(sent)
                    if overlap_tokens + st > chunk_overlap:
                        break
                    overlap_sentences.insert(0, sent)
                    overlap_tokens += st
                current_sentences = overlap_sentences
                current_tokens = overlap_tokens
            else:
                current_sentences = []
                current_tokens = 0

        current_sentences.append(sentence)
        current_tokens += sent_tokens

    if current_sentences:
        chunk_text = " ".join(current_sentences)
        meta = dict(source_metadata) if source_metadata else {}
        meta["chunk_index"] = idx
        meta["strategy"] = "semantic"
        chunks.append(Chunk(
            text=chunk_text.strip(),
            chunk_index=idx,
            token_count=count_tokens(chunk_text),
            metadata=meta,
        ))

    return chunks


def _split_sentences(text: str) -> list[str]:
    pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    raw = re.split(pattern, text)
    sentences: list[str] = []
    for part in raw:
        stripped = part.strip()
        if stripped:
            sentences.append(stripped)
    return sentences


def _apply_overlap(
    chunks: list[Chunk],
    overlap_tokens: int,
    source_metadata: dict[str, str | int | float] | None,
) -> list[Chunk]:
    if len(chunks) <= 1:
        return chunks

    result: list[Chunk] = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tokens = _encoder.encode(chunks[i - 1].text)
        overlap_text = (
            _encoder.decode(prev_tokens[-overlap_tokens:])
            if len(prev_tokens) > overlap_tokens
            else chunks[i - 1].text
        )
        combined = overlap_text.strip() + " " + chunks[i].text
        token_count = count_tokens(combined)
        meta = dict(source_metadata) if source_metadata else {}
        meta["chunk_index"] = i
        meta["strategy"] = chunks[i].metadata.get("strategy", "recursive")
        result.append(Chunk(
            text=combined.strip(),
            chunk_index=i,
            token_count=token_count,
            metadata=meta,
        ))
    return result


def chunk_document_pages(
    pages: list[str],
    strategy: ChunkingStrategy = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    source_metadata: dict[str, str | int | float] | None = None,
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    global_idx = 0

    for page_num, page_text in enumerate(pages):
        meta = dict(source_metadata) if source_metadata else {}
        meta["page"] = page_num + 1
        page_chunks = chunk_text(page_text, strategy, chunk_size, chunk_overlap, meta)
        for chunk in page_chunks:
            chunk.chunk_index = global_idx
            chunk.metadata["chunk_index"] = global_idx
            all_chunks.append(chunk)
            global_idx += 1

    return all_chunks
