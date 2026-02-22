# RAG Eval Engine — Capabilities & Technical Deep Dive

> A portfolio walkthrough of every system in the RAG Eval Engine, what it does technically, and the engineering skills it demonstrates.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Hybrid Retrieval Engine](#1-hybrid-retrieval-engine)
3. [Document Ingestion Pipeline](#2-document-ingestion-pipeline)
4. [Multi-Provider LLM Routing](#3-multi-provider-llm-routing)
5. [Semantic Query Cache](#4-semantic-query-cache-fact-pattern)
6. [Evaluation Engine](#5-evaluation-engine)
7. [Adaptive Retrieval (Self-Learning)](#6-adaptive-retrieval-self-learning)
8. [MCP Server](#7-mcp-server)
9. [Dashboard (6 Pages)](#8-dashboard-6-pages)
10. [Infrastructure & DevOps](#9-infrastructure--devops)
11. [Skills Showcased](#skills-showcased)

---

## System Overview

```
┌─────────────────────────────────────────────────────┐
│                   Next.js Dashboard                  │
│  Query | Documents | Retrieval | Eval | Test Sets    │
└──────────────────────┬──────────────────────────────┘
                       │ REST API + SSE
┌──────────────────────▼──────────────────────────────┐
│                  FastAPI Gateway                      │
│  /ingest  /query  /retrieve  /evaluate  /metrics     │
│  Middleware: CORS, X-Response-Time, Cache Init        │
└───┬──────────┬──────────┬──────────┬────────────────┘
    │          │          │          │
┌───▼───┐ ┌───▼───┐ ┌───▼────┐ ┌───▼─────┐
│Ingest │ │Hybrid │ │Generate│ │Evaluate │
│Pipeline│ │Ranker │ │Engine  │ │Engine   │
└───┬───┘ └───┬───┘ └───┬────┘ └───┬─────┘
    │    ┌────┴────┐    │          │
    ▼    ▼         ▼    ▼          ▼
┌────────┐┌──────┐┌──────┐┌──────┐┌────────┐
│Qdrant  ││Vector││BM25  ││Ollama││SQLite  │
│VectorDB││Search││Index ││/LLM  ││Metrics │
└────────┘└──────┘└──────┘└──────┘└────────┘
```

**What it is:** A complete RAG system where every query is scored for quality in real-time. Not just retrieval and generation — continuous measurement of faithfulness, relevance, and hallucination rate.

**What makes it different:** Most RAG systems are black boxes. This one has evaluation baked into the pipeline. You can see exactly when quality degrades, which retrieval parameters produce the best results, and how much each query costs.

---

## 1. Hybrid Retrieval Engine

### What it does

Combines two fundamentally different search approaches — **vector similarity** and **keyword matching** — then merges them using Reciprocal Rank Fusion to get the best of both.

### How it works technically

**Vector Search** (`src/retrieval/vector_search.py`)
- Embeds the query into a 384-dimensional vector using `all-MiniLM-L6-v2`
- Searches Qdrant using **cosine similarity**
- Returns chunks ranked by semantic closeness to the query
- Supports source file filtering via Qdrant `FieldCondition`

**BM25 Sparse Search** (`src/retrieval/sparse_search.py`)
- Classic **BM25Okapi** (Best Match 25) algorithm from information retrieval
- Custom tokenizer: lowercase, strip punctuation, remove single-char tokens
- Index persisted as JSON in `data/bm25_indices/{collection}.json`
- In-memory cache per collection for fast repeated lookups

**Reciprocal Rank Fusion** (`src/retrieval/hybrid_ranker.py`)
```
For each document:
  vector_rrf  = 1 / (60 + rank_in_vector_results + 1)
  sparse_rrf  = 1 / (60 + rank_in_sparse_results + 1)
  final_score = alpha * vector_rrf + (1 - alpha) * sparse_rrf
```

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `alpha` | 0.7 | 70% vector, 30% keyword. Tunable 0.0–1.0 |
| `top_k` | 5 | Final chunks returned |
| `fetch_k` | `top_k * 3` | Oversample before fusion for better coverage |
| `rrf_k` | 60 | Standard RRF constant (dampens rank influence) |

**Deduplication:** Chunks are keyed by first 200 characters (normalized lowercase) to prevent duplicates from both pipelines appearing in results.

### Why hybrid matters

Pure vector search misses exact keyword matches ("error code 404" won't match "HTTP 404 error"). Pure BM25 misses semantic connections ("car" won't match "automobile"). RRF lets you tune the balance per use case.

### Skills demonstrated

- **Information retrieval theory** — BM25, cosine similarity, rank fusion
- **Algorithm design** — RRF implementation with configurable weighting
- **System design** — Two parallel search paths merged at retrieval time
- **Performance optimization** — In-memory caching, oversample-then-filter

---

## 2. Document Ingestion Pipeline

### What it does

Takes raw files (PDF, DOCX, TXT, Markdown, 15+ code formats), splits them into semantically meaningful chunks, embeds each chunk, and stores everything in Qdrant for retrieval.

### Three chunking strategies (`src/ingestion/chunker.py`)

**Fixed-size chunking**
- Sliding window over token sequence (512 tokens default, 50 overlap)
- Token counting via `tiktoken` (`cl100k_base` encoding — same as GPT-4)
- Fast and predictable, but may split mid-sentence

**Recursive chunking** (default)
- Splits by separators in order: `\n\n` → `\n` → `. ` → ` ` → `""`
- If a chunk exceeds the size limit, recurse with the next separator
- Respects document structure (paragraphs > sentences > words)
- Token-based overlap reconstruction between adjacent chunks

**Semantic chunking**
- Splits by sentence boundaries using regex: `(?<=[.!?])\s+(?=[A-Z])`
- Accumulates sentences until chunk size exceeded
- Falls back to fixed chunking for oversized single sentences
- Most coherent chunks, but slowest

### File loading (`src/ingestion/loader.py`)

| Format | Library | Metadata |
|--------|---------|----------|
| PDF | PyMuPDF (fitz) | page_count, per-page text |
| DOCX | python-docx | paragraph_count |
| Text/MD | Built-in | encoding detection |
| Code (.py, .ts, .rs, .go, etc.) | Built-in | language, source path |

- **Encoding detection:** Tries utf-8 → utf-8-sig → latin-1 → cp1252, then falls back to utf-8 with replacement
- **Text cleaning:** Normalizes whitespace while preserving newlines

### Embedding & storage (`src/ingestion/embedder.py`)

| Model | Dimensions | Source |
|-------|-----------|--------|
| `all-MiniLM-L6-v2` | 384 | Local (sentence-transformers) |
| `BAAI/bge-base-en-v1.5` | 768 | Local (sentence-transformers) |
| `text-embedding-3-small` | 1536 | OpenAI API |

- **Batch processing:** 64 embeddings per batch
- **L2 normalization** on local models
- **Qdrant upsert:** 100 points per batch, cosine distance metric
- **Point IDs:** Deterministic hash of `{doc_id}_{chunk_index}`

### Skills demonstrated

- **NLP fundamentals** — Tokenization, sentence segmentation, chunking strategies
- **ETL pipeline design** — Load → transform → embed → store
- **Multi-format parsing** — PDF rendering, DOCX XML extraction, encoding detection
- **Batch processing** — Efficient embedding with configurable batch sizes

---

## 3. Multi-Provider LLM Routing

### What it does

Routes queries to the right LLM provider automatically, tracks costs per query, and supports real-time streaming from all three providers.

### Routing logic (`src/generation/llm_client.py`)

```
claude-*          → Anthropic API (via httpx, no SDK dependency)
gpt-* / o1 / o3  → OpenAI API (via official SDK)
everything else   → Ollama (local, free)
```

### Provider implementations

**Ollama** (default, local)
- Endpoint: `http://localhost:11434/api/chat`
- Streaming: Newline-delimited JSON, each line has `message.content`
- Health check: GET `/api/tags` with 5s timeout
- Cost: Always $0.00

**OpenAI**
- SDK: `AsyncOpenAI` with async streaming
- Streaming: AsyncIterator on `chunk.choices[0].delta.content`
- Token counting from `response.usage`

**Anthropic** (no SDK — raw httpx)
- Endpoint: `https://api.anthropic.com/v1/messages`
- Headers: `x-api-key`, `anthropic-version: 2023-06-01`
- Streaming: SSE format, filters for `content_block_delta` events
- Max tokens: 4096

### Cost tracking (`src/generation/cost_tracker.py`)

| Model | Input $/M tokens | Output $/M tokens |
|-------|-------------------|---------------------|
| gpt-4o | $2.50 | $10.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| gpt-4-turbo | $10.00 | $30.00 |
| o1 | $15.00 | $60.00 |
| o3-mini | $1.10 | $4.40 |
| claude-3.5-sonnet | $3.00 | $15.00 |
| claude-3-opus | $15.00 | $75.00 |
| Ollama (any) | $0.00 | $0.00 |

```python
cost = (input_tokens / 1_000_000 * input_rate) + (output_tokens / 1_000_000 * output_rate)
```

Model matching uses case-insensitive substring matching against the cost table.

### Skills demonstrated

- **API integration** — Three different API patterns (REST, SDK, SSE) unified under one interface
- **Streaming architecture** — Server-Sent Events with incremental token delivery
- **Cost engineering** — Per-query cost tracking with model-specific pricing
- **Zero-dependency design** — Anthropic integration uses httpx instead of a vendor SDK

---

## 4. Semantic Query Cache (FACT Pattern)

### What it does

Caches query results using **embedding similarity** (not just exact string matching). If someone asks "What is machine learning?" and the cache has "Explain ML", it recognizes they're semantically identical and returns the cached result in <100ms instead of running the full pipeline.

### How it works (`src/caching/query_cache.py`)

```
1. Embed the incoming query
2. Search Qdrant _query_cache collection (cosine similarity)
3. If best match score >= 0.95 AND same collection AND not expired (TTL):
     → Return cached answer (cache HIT)
4. Otherwise:
     → Run full pipeline, then store result in cache (cache MISS)
```

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `cache_threshold` | 0.95 | Minimum similarity to consider a match |
| `cache_ttl_seconds` | 3600 | Entries expire after 1 hour |
| `cache_enabled` | true | Global toggle |

### What gets cached

- Query text and embedding
- Full answer text
- Source citations (JSON)
- Eval scores (JSON)
- Model used, token count, latency
- Timestamp for TTL enforcement

### Cache key

```python
hash = SHA256(f"{collection}:{query.strip().lower()}")
point_id = abs(hash(hash_string)) % (2^63)
```

### Statistics tracking

Every lookup records a hit or miss in SQLite's `cache_stats` table with the saved latency (for hits). The dashboard shows:
- Hit rate percentage
- Total cached entries
- Queries saved (avoided full pipeline)
- Average latency saved per hit

### Skills demonstrated

- **Caching architecture** — Semantic similarity cache (not just hash-based)
- **The FACT pattern** — Fetch-And-Cache-Together, used in production RAG systems
- **Performance optimization** — Sub-100ms responses on cache hits vs 5-15s full pipeline
- **Observability** — Hit/miss tracking with latency savings metrics

---

## 5. Evaluation Engine

### What it does

Every query gets scored for quality. Not as an afterthought — evaluation runs inline as part of the query pipeline. Five metrics measure whether the answer is grounded, relevant, and free of hallucinations.

### Five metrics (`src/evaluation/metrics.py`)

| Metric | What it measures | Primary method | Fallback |
|--------|-----------------|----------------|----------|
| **Faithfulness** | Is the answer grounded in the retrieved context? | LLM-as-judge (0.0–1.0) | Word overlap ratio |
| **Relevance** | Does the answer address the question asked? | LLM-as-judge (0.0–1.0) | Query term overlap |
| **Hallucination Rate** | What fraction of claims are ungrounded? | LLM counts grounded vs ungrounded sentences | `1 - faithfulness_heuristic` |
| **Context Precision** | Are the retrieved chunks actually relevant? | Word overlap threshold (20% of query terms) | Deterministic |
| **Context Recall** | Did we retrieve everything needed? | LLM compares context to ground truth | Requires ground truth |

### LLM-as-judge approach

Each metric sends a targeted prompt to the LLM asking it to rate the output on a 0.0–1.0 scale with specific guidance:

```
"Rate faithfulness from 0.0 to 1.0:
 0.0 = answer contradicts the context
 0.5 = partially supported
 1.0 = fully grounded in provided context"
```

Score parsing uses regex to extract the first float from the response, clamped to [0.0, 1.0].

### Heuristic fallbacks

When the LLM judge is unavailable or too slow, word-overlap heuristics provide approximate scores:

```python
# Faithfulness heuristic
answer_words = set(answer.lower().split())
context_words = set(context.lower().split())
overlap = answer_words & context_words
score = len(overlap) / len(answer_words)
```

### Lightweight mode vs full evaluation

| Mode | Metrics computed | Use case |
|------|-----------------|----------|
| **Lightweight** (default) | Faithfulness + Relevance only | Every query in real-time |
| **Full** | All 5 metrics | Batch evaluation on test sets |

### Batch evaluation (`src/evaluation/eval_pipeline.py`)

1. Load test set (question + optional ground truth pairs)
2. Run each question through the full pipeline with all 5 metrics
3. Aggregate averages across the test set
4. Store results with status tracking (pending → running → completed)

### Test set auto-generation (`src/evaluation/test_sets.py`)

- Fetches 10 random chunks from the collection via Qdrant
- Asks the LLM to generate diverse Q&A pairs from those chunks
- Parses JSON output (handles markdown code blocks)
- Stores as reusable test set for repeated evaluation

### Skills demonstrated

- **LLM-as-judge evaluation** — The standard approach for RAG quality measurement
- **Graceful degradation** — Heuristic fallbacks when LLM is unavailable
- **RAG evaluation theory** — Faithfulness, relevance, hallucination, context precision/recall
- **Batch processing** — Async evaluation runs with status tracking
- **Test data generation** — Automated Q&A creation from documents

---

## 6. Adaptive Retrieval (Self-Learning)

### What it does

The system learns from its own performance. It correlates retrieval parameters (alpha, top_k) with evaluation scores to automatically recommend the optimal configuration for each collection.

### How it works (`src/retrieval/auto_tune.py`)

```
1. Query the last 500 evaluated queries for a collection
2. For each query, record: (alpha_used, top_k_used, quality_score)
     quality = (faithfulness + relevance) / 2
3. Bin alpha values into 11 buckets: [0.0, 0.1, 0.2, ..., 1.0]
4. For each bin with >= 3 data points, compute average quality
5. Return the alpha bin with highest average quality
6. Same process for top_k values
```

| Constraint | Value | Purpose |
|------------|-------|---------|
| Minimum queries before tuning | 10 | Prevent premature optimization |
| Minimum data points per bin | 3 | Statistical significance |
| Query lookback window | 500 | Focus on recent performance |

### Integration

When `auto_tune: true` is passed to the query endpoint:
1. `get_optimal_params(collection)` runs before retrieval
2. If sufficient data exists, recommended alpha/top_k override defaults
3. The parameters used are logged in `query_log` for future tuning cycles

This creates a **feedback loop** — the system continuously improves its retrieval as more queries are evaluated.

### Skills demonstrated

- **Self-learning systems** — Feedback loop between evaluation and retrieval
- **Statistical analysis** — Binned histogram approach for parameter optimization
- **Production ML patterns** — Minimum sample requirements, lookback windows
- **Incremental improvement** — System gets better with usage, no manual tuning

---

## 7. MCP Server

### What it does

Exposes the entire RAG system as **Model Context Protocol** tools, letting Claude Code (or any MCP client) query your knowledge base directly from the terminal.

### Protocol (`src/mcp_server.py`)

- **Transport:** JSON-RPC 2.0 over stdio (standard MCP)
- **Protocol version:** 2024-11-05
- **Server info:** `rag-eval-engine v1.0.0`

### Five tools

| Tool | Purpose | Key params |
|------|---------|------------|
| `rag_query` | Full RAG pipeline — retrieve + generate + eval | query, collection, top_k, model, evaluate |
| `rag_retrieve` | Hybrid search only — returns ranked chunks | query, collection, top_k, alpha |
| `rag_ingest_text` | Ingest raw text into a collection | text, collection, source |
| `rag_collections` | List collections with doc/chunk/vector counts | — |
| `rag_metrics` | Get evaluation metrics summary | collection, limit |

### Registration

```json
{
  "mcpServers": {
    "rag-eval-engine": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "/path/to/rag-eval-engine"
    }
  }
}
```

### Skills demonstrated

- **Protocol implementation** — JSON-RPC 2.0 from scratch (no framework)
- **MCP specification** — Correct tool schema, initialize handshake, capabilities
- **Stdio I/O** — Async stdin/stdout communication with proper line buffering
- **API design** — Clean tool interfaces that mirror the REST API

---

## 8. Dashboard (6 Pages)

### Query Playground

![Query Playground](screenshots/01-query-playground.png)

- Chat interface with real-time **SSE streaming** (tokens appear as generated)
- Model selector dropdown (all Ollama models + cloud providers)
- Collection selector with top-k control
- **Eval score badges** on each response (faithfulness %, relevance %)
- **Cache hit badge** (amber) when response served from cache
- **Cost badge** ($0.00 for Ollama, actual cost for cloud)
- Suggested starter queries for empty state
- **Ctrl+K / Cmd+K** keyboard shortcut to focus input

### Document Management

![Documents](screenshots/02-documents.png)

- **Drag-and-drop** file upload area
- Collection name input + chunking strategy picker (recursive/fixed/semantic)
- Advanced options: chunk size, overlap, embedding model
- Stats cards: collections count, documents, chunks, vectors
- Collection cards with document count, chunk count, token count
- Delete collection with **toast confirmation**
- **Skeleton loading** states while data loads

### Retrieval Explorer

![Retrieval](screenshots/03-retrieval.png)

- Test hybrid retrieval independently from generation
- **Alpha slider** (0.0–1.0) with real-time label: "keyword-heavy" / "balanced" / "vector-heavy"
- **Top-K slider** (1–20)
- Collection selector with source file filter
- Auto-tune recommendation banner: "Recommended: alpha=0.8, top_k=7" with Apply button
- Results show: chunk text, fused score, vector score, sparse score, source file
- **Expand/collapse** for long chunk text (>300 chars)

### Evaluation Dashboard

![Evaluation](screenshots/04-eval.png)

- **5 metric cards:** Avg Faithfulness, Avg Relevance, Avg Hallucination, P95 Latency, Total Cost
- **Query Cache section:** Hit rate, cached entries, queries saved, avg latency saved
- **Quality Over Time** area chart (faithfulness + relevance trend lines)
- **3-column chart grid:** Latency distribution, Token usage, Cost per query (bar charts)
- **Performance Summary:** Total queries, P50 latency, avg latency
- Collection filter dropdown
- **CSV export** button (downloads timestamped metrics file)
- **Auto-refresh** toggle (30-second interval)

### Test Sets

![Test Sets](screenshots/05-test-sets.png)

- Create test sets with Q&A pairs
- Auto-generate questions from documents (LLM-powered)
- Run batch evaluations against test sets
- View evaluation run history with aggregate scores

### Settings

![Settings](screenshots/06-settings.png)

- System status: API server, Ollama, embedding model, default LLM (green dots)
- Available models grid with sizes (all Ollama models)
- Retrieval configuration display
- Evaluation configuration display
- Environment variables reference

### Cross-cutting UI features

| Feature | Implementation |
|---------|---------------|
| Dark mode | System preference detection + manual toggle |
| Responsive layout | Collapsible sidebar on mobile |
| Skeleton loading | Shimmer animation components (pulse with staggered delays) |
| Toast notifications | Context-based system: success/error/info, auto-dismiss 4s, stack 3 max |
| Keyboard shortcuts | K (Query), D (Documents), R (Retrieval), E (Evaluation) |
| Streaming | SSE with incremental token rendering |

### Skills demonstrated

- **Full-stack development** — Next.js 14 App Router + TypeScript strict
- **Real-time UI** — SSE streaming with incremental rendering
- **Data visualization** — Recharts (area, bar, line charts)
- **Component architecture** — Reusable toast, skeleton, sidebar components
- **UX design** — Empty states, loading states, keyboard shortcuts, responsive layout
- **State management** — React hooks (useState, useEffect, useCallback, useRef)

---

## 9. Infrastructure & DevOps

### Database (`src/db/models.py`)

- **Engine:** SQLite with WAL (Write-Ahead Logging) mode
- **Async:** aiosqlite for non-blocking queries
- **8 tables:** documents, ingestion_jobs, query_log, eval_results, test_sets, cache_stats, eval_runs
- **7 indexes** on frequently queried columns
- **Migrations:** `init_db()` handles ALTER TABLE for schema evolution on existing databases
- **Foreign keys** enabled for referential integrity

### Docker Compose (`docker-compose.yml`)

```yaml
3 services:
  qdrant     → 2GB memory limit, health check on /healthz
  api        → 4GB memory limit, health check on /health, depends_on: qdrant (healthy)
  dashboard  → 512MB memory limit, depends_on: api (healthy)

3 volumes:
  qdrant_data, api_data, api_uploads
```

- Health checks with proper intervals and retries
- `service_healthy` dependency conditions (not just `service_started`)
- Resource limits prevent container runaway
- Ollama runs on host, accessed via `host.docker.internal`

### CI Pipeline (`.github/workflows/ci.yml`)

```
Backend:  ruff lint → pyright type check → pytest (68 tests)
Dashboard: next lint → tsc --noEmit → npm run build
```

Triggers on push to main and all pull requests.

### API Middleware (`src/main.py`)

- **CORS:** Configurable allowed origins
- **Response timing:** `X-Response-Time` header on every response
- **Lifespan management:** DB init, cache collection init, Ollama health check

### Configuration (`src/config.py`)

- **Pydantic BaseSettings** with env prefix `RAG_`
- **19 settings** covering: Qdrant, embedding, chunking, LLM, retrieval, cache, evaluation, database, upload limits
- **Type-safe** with Literal types for enums (strategy, model names)
- `.env` file support

### Skills demonstrated

- **Database design** — Normalized schema, WAL mode, async queries, migrations
- **Containerization** — Multi-service Docker Compose with health checks
- **CI/CD** — Automated linting, type checking, testing, building
- **Configuration management** — Environment-based settings with Pydantic validation
- **Production readiness** — Resource limits, health checks, timing middleware

---

## Skills Showcased

### Backend Engineering

| Skill | Where it's demonstrated |
|-------|------------------------|
| **Python async/await** | All DB queries, HTTP clients, embedding, streaming |
| **FastAPI** | REST API, SSE streaming, middleware, lifespan management |
| **API design** | 20 endpoints, consistent response formats, error handling |
| **Database design** | 8-table SQLite schema with indexes, WAL mode, migrations |
| **Caching** | Semantic cache with Qdrant, TTL, similarity threshold |
| **Algorithm implementation** | RRF, BM25, LLM-as-judge, auto-tuning |

### AI/ML Engineering

| Skill | Where it's demonstrated |
|-------|------------------------|
| **RAG architecture** | Full pipeline: ingest → chunk → embed → retrieve → generate → evaluate |
| **Vector databases** | Qdrant integration for embeddings and semantic cache |
| **Information retrieval** | Hybrid search, BM25, cosine similarity, rank fusion |
| **LLM integration** | 3 providers, streaming, cost tracking, prompt engineering |
| **Evaluation methodology** | 5 metrics, LLM-as-judge, heuristic fallbacks |
| **Self-learning systems** | Adaptive retrieval parameter optimization |
| **Embedding models** | Local (sentence-transformers) + cloud (OpenAI) |

### Frontend Engineering

| Skill | Where it's demonstrated |
|-------|------------------------|
| **Next.js 14** | App Router, server components, client components |
| **TypeScript strict** | Type-safe API client, interfaces, generics |
| **Real-time streaming** | SSE handling with incremental DOM updates |
| **Data visualization** | Recharts: area, bar, line charts with time series |
| **Component design** | Toast system, skeleton loaders, sidebar navigation |
| **UX polish** | Keyboard shortcuts, loading states, empty states, responsive |

### Systems & DevOps

| Skill | Where it's demonstrated |
|-------|------------------------|
| **Docker Compose** | 3-service stack with health checks and resource limits |
| **CI/CD** | GitHub Actions: lint, type check, test, build |
| **Protocol implementation** | MCP server (JSON-RPC 2.0 over stdio) |
| **Multi-provider integration** | Ollama + OpenAI + Anthropic unified interface |
| **Observability** | Response timing, cost tracking, cache stats, eval metrics |
| **Configuration management** | Pydantic settings with env vars and .env support |

### Architecture Patterns

| Pattern | Application |
|---------|-------------|
| **FACT caching** | Semantic query cache with embedding similarity |
| **Reciprocal Rank Fusion** | Hybrid retrieval merging vector + keyword results |
| **LLM-as-judge** | Automated quality scoring of RAG outputs |
| **Feedback loops** | Eval scores feed back into retrieval parameter optimization |
| **Graceful degradation** | Heuristic fallbacks when LLM judge is unavailable |
| **Provider abstraction** | Single interface over 3 LLM providers |
| **Event streaming** | SSE for real-time token delivery to the UI |

---

## By the Numbers

| Metric | Value |
|--------|-------|
| Python source files | 15 modules across 7 packages |
| TypeScript source files | 10 components + pages |
| API endpoints | 20 |
| MCP tools | 5 |
| Eval metrics | 5 |
| Chunking strategies | 3 |
| LLM providers | 3 (Ollama, OpenAI, Anthropic) |
| Embedding models | 3 (MiniLM, BGE, OpenAI) |
| Dashboard pages | 6 |
| Database tables | 8 |
| Tests | 68 |
| Supported file types | 20+ (PDF, DOCX, TXT, MD, 15+ code formats) |
| Docker services | 3 (Qdrant, API, Dashboard) |
| CI jobs | 2 (backend + dashboard) |
