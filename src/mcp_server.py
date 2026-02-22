"""MCP Server for RAG Eval Engine.

Exposes RAG capabilities as MCP tools over JSON-RPC 2.0 via stdio.
Tools: rag_query, rag_retrieve, rag_ingest_text, rag_collections, rag_metrics
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from typing import Any


async def handle_request(request: dict[str, Any]) -> dict[str, Any]:
    method = request.get("method", "")
    params = request.get("params", {})
    req_id = request.get("id")

    if method == "initialize":
        return _ok(req_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "rag-eval-engine", "version": "1.0.0"},
        })

    if method == "notifications/initialized":
        return {}

    if method == "tools/list":
        return _ok(req_id, {"tools": TOOLS})

    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        result = await _call_tool(tool_name, arguments)
        return _ok(req_id, {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]})

    return _error(req_id, -32601, f"Method not found: {method}")


async def _call_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    # Lazy imports to avoid loading heavy modules at startup
    if name == "rag_query":
        return await _tool_query(args)
    if name == "rag_retrieve":
        return await _tool_retrieve(args)
    if name == "rag_ingest_text":
        return await _tool_ingest_text(args)
    if name == "rag_collections":
        return await _tool_collections()
    if name == "rag_metrics":
        return await _tool_metrics(args)
    return {"error": f"Unknown tool: {name}"}


async def _tool_query(args: dict[str, Any]) -> dict[str, Any]:
    from src.db.models import init_db
    from src.evaluation.eval_pipeline import run_query_pipeline

    await init_db()
    result = await run_query_pipeline(
        query=args["query"],
        collection=args.get("collection", "documents"),
        top_k=args.get("top_k", 5),
        model=args.get("model"),
        evaluate=args.get("evaluate", False),
    )
    return {
        "answer": result.answer,
        "sources": result.sources,
        "model": result.model,
        "tokens_used": result.tokens_used,
        "latency_ms": round(result.latency_ms, 1),
        "cache_hit": result.cache_hit,
    }


async def _tool_retrieve(args: dict[str, Any]) -> dict[str, Any]:
    from src.retrieval.hybrid_ranker import hybrid_search

    results = hybrid_search(
        query=args["query"],
        collection_name=args.get("collection", "documents"),
        top_k=args.get("top_k", 5),
        alpha=args.get("alpha"),
    )
    return {
        "chunks": [
            {
                "text": r.text,
                "score": round(r.score, 4),
                "source": r.metadata.get("source", ""),
                "chunk_index": r.chunk_index,
            }
            for r in results
        ],
        "count": len(results),
    }


async def _tool_ingest_text(args: dict[str, Any]) -> dict[str, Any]:
    from src.db.models import init_db, insert_document
    from src.ingestion.chunker import chunk_text
    from src.ingestion.embedder import embed_texts, ensure_collection, store_chunks
    from src.retrieval.sparse_search import add_to_index

    await init_db()
    text = args["text"]
    collection = args.get("collection", "documents")
    source_name = args.get("source", "mcp_input")

    await ensure_collection(collection)
    chunks = chunk_text(text, strategy="recursive")

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    doc_id = str(uuid.uuid4())
    for c in chunks:
        c.metadata["source"] = source_name

    await store_chunks(chunks, embeddings, collection, doc_id)
    add_to_index(collection, texts, [c.metadata for c in chunks])

    total_tokens = sum(c.token_count for c in chunks)
    await insert_document(doc_id, collection, source_name, "text", len(chunks), total_tokens)

    return {
        "doc_id": doc_id,
        "chunks_created": len(chunks),
        "total_tokens": total_tokens,
        "collection": collection,
    }


async def _tool_collections() -> dict[str, Any]:
    from src.db.models import get_collections, init_db
    from src.ingestion.embedder import get_collection_info

    await init_db()
    collections = await get_collections()
    result = []
    for col in collections:
        name = col["collection"]
        info = get_collection_info(name)
        result.append({
            "name": name,
            "doc_count": col["doc_count"],
            "total_chunks": col["total_chunks"],
            "total_tokens": col["total_tokens"],
            "vectors_count": info.get("vectors_count", 0),
        })
    return {"collections": result, "count": len(result)}


async def _tool_metrics(args: dict[str, Any]) -> dict[str, Any]:
    from src.db.models import get_metrics, init_db

    await init_db()
    metrics = await get_metrics(args.get("collection"), args.get("limit", 50))
    if not metrics:
        return {"total_queries": 0, "message": "No metrics data yet"}

    faithfulness_vals = [m["faithfulness"] for m in metrics if m.get("faithfulness") is not None]
    relevance_vals = [m["relevance"] for m in metrics if m.get("relevance") is not None]

    return {
        "total_queries": len(metrics),
        "avg_faithfulness": round(sum(faithfulness_vals) / len(faithfulness_vals), 3) if faithfulness_vals else None,
        "avg_relevance": round(sum(relevance_vals) / len(relevance_vals), 3) if relevance_vals else None,
    }


def _ok(req_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _error(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


TOOLS = [
    {
        "name": "rag_query",
        "description": "Query documents using RAG with optional evaluation. Returns an answer grounded in your document collection with source citations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The question to ask"},
                "collection": {"type": "string", "description": "Document collection name", "default": "documents"},
                "top_k": {"type": "integer", "description": "Number of chunks to retrieve", "default": 5},
                "model": {"type": "string", "description": "LLM model to use (optional)"},
                "evaluate": {"type": "boolean", "description": "Run quality evaluation on the response", "default": False},
            },
            "required": ["query"],
        },
    },
    {
        "name": "rag_retrieve",
        "description": "Retrieve ranked document chunks using hybrid search (vector + BM25). Returns chunks sorted by relevance score.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "collection": {"type": "string", "description": "Collection to search", "default": "documents"},
                "top_k": {"type": "integer", "description": "Number of results", "default": 5},
                "alpha": {"type": "number", "description": "Vector vs keyword weight (0=BM25, 1=vector)", "default": 0.7},
            },
            "required": ["query"],
        },
    },
    {
        "name": "rag_ingest_text",
        "description": "Ingest raw text into a document collection. Chunks, embeds, and indexes the text for later retrieval.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text content to ingest"},
                "collection": {"type": "string", "description": "Target collection", "default": "documents"},
                "source": {"type": "string", "description": "Source name for the text", "default": "mcp_input"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "rag_collections",
        "description": "List all document collections with their statistics (doc count, chunks, tokens, vectors).",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "rag_metrics",
        "description": "Get evaluation metrics summary including average faithfulness and relevance scores.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "collection": {"type": "string", "description": "Filter by collection (optional)"},
                "limit": {"type": "integer", "description": "Max queries to aggregate", "default": 50},
            },
        },
    },
]


async def main_async() -> None:
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin.buffer)

    while True:
        line = await reader.readline()
        if not line:
            break

        line_str = line.decode("utf-8").strip()
        if not line_str:
            continue

        try:
            request = json.loads(line_str)
            response = await handle_request(request)
            if response:
                out = json.dumps(response) + "\n"
                sys.stdout.write(out)
                sys.stdout.flush()
        except json.JSONDecodeError:
            err = _error(None, -32700, "Parse error")
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
