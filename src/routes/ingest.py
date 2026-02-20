from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile
from pydantic import BaseModel
from qdrant_client import QdrantClient

from src.config import settings
from src.db.models import (
    delete_collection_docs,
    get_collections,
    get_ingestion_job,
    insert_document,
    insert_ingestion_job,
    update_ingestion_job,
)
from src.ingestion.chunker import ChunkingStrategy, chunk_document_pages, chunk_text
from src.ingestion.embedder import (
    delete_collection,
    embed_texts,
    ensure_collection,
    get_collection_info,
    store_chunks,
)
from src.ingestion.loader import SUPPORTED_EXTENSIONS, load_document
from src.retrieval.sparse_search import add_to_index, delete_index

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["ingestion"])


class IngestRequest(BaseModel):
    collection: str = "documents"
    chunking_strategy: Literal["fixed", "recursive", "semantic"] = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 50


class IngestResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    total_files: int
    processed_files: int
    total_chunks: int
    error: str | None = None


class CollectionInfo(BaseModel):
    name: str
    doc_count: int
    total_chunks: int
    total_tokens: int
    vectors_count: int


def _check_qdrant_health() -> None:
    """Verify Qdrant is reachable before accepting an upload."""
    try:
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=5)
        client.get_collections()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Qdrant is not available at {settings.qdrant_url}. Start Qdrant before uploading. ({exc})",
        )


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    background_tasks: BackgroundTasks,
    files: list[UploadFile],
    collection: str = "documents",
    chunking_strategy: str = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> IngestResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    if len(files) > settings.max_files_per_upload:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum {settings.max_files_per_upload} files per upload, got {len(files)}.",
        )

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    for file in files:
        if file.filename:
            ext = Path(file.filename).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unsupported file type: {ext}."
                        f" Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
                    ),
                )
        if file.size is not None and file.size > max_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' exceeds {settings.max_file_size_mb}MB limit.",
            )

    _check_qdrant_health()

    job_id = str(uuid.uuid4())
    await insert_ingestion_job(job_id, collection, len(files))

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for file in files:
        if not file.filename:
            continue
        file_path = upload_dir / f"{job_id}_{file.filename}"
        content = await file.read()
        file_path.write_bytes(content)
        saved_paths.append(file_path)

    background_tasks.add_task(
        _process_ingestion,
        job_id=job_id,
        file_paths=saved_paths,
        collection=collection,
        strategy=chunking_strategy,  # type: ignore[arg-type]
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return IngestResponse(
        job_id=job_id,
        status="processing",
        message=f"Ingesting {len(files)} file(s) into collection '{collection}'",
    )


@router.get("/ingest/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str) -> JobStatus:
    job = await get_ingestion_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(
        job_id=job["id"],
        status=job["status"],
        total_files=job["total_files"],
        processed_files=job["processed_files"],
        total_chunks=job["total_chunks"],
        error=job.get("error"),
    )


@router.get("/collections", response_model=list[CollectionInfo])
async def list_collections() -> list[CollectionInfo]:
    db_collections = await get_collections()
    result: list[CollectionInfo] = []

    for col in db_collections:
        vec_info = get_collection_info(col["collection"])
        result.append(CollectionInfo(
            name=col["collection"],
            doc_count=col["doc_count"],
            total_chunks=col["total_chunks"] or 0,
            total_tokens=col["total_tokens"] or 0,
            vectors_count=vec_info.get("vectors_count", 0),
        ))

    return result


@router.delete("/collections/{name}")
async def remove_collection(name: str) -> dict[str, str]:
    await delete_collection(name)
    await delete_collection_docs(name)
    delete_index(name)
    return {"status": "deleted", "collection": name}


async def _process_ingestion(
    job_id: str,
    file_paths: list[Path],
    collection: str,
    strategy: ChunkingStrategy,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    try:
        await ensure_collection(collection)
        total_chunks = 0

        for i, file_path in enumerate(file_paths):
            try:
                doc = load_document(file_path)
                doc_id = str(uuid.uuid4())

                source_meta: dict[str, str | int | float] = {
                    "source": doc.filename,
                    "file_type": doc.file_type,
                    "doc_id": doc_id,
                }

                if doc.pages:
                    chunks = chunk_document_pages(
                        doc.pages, strategy, chunk_size, chunk_overlap, source_meta
                    )
                else:
                    chunks = chunk_text(
                        doc.text, strategy, chunk_size, chunk_overlap, source_meta
                    )

                if not chunks:
                    continue

                texts = [c.text for c in chunks]
                embeddings = embed_texts(texts)
                await store_chunks(chunks, embeddings, collection, doc_id)

                add_to_index(
                    collection,
                    texts,
                    [c.metadata for c in chunks],
                )

                token_count = sum(c.token_count for c in chunks)
                await insert_document(
                    doc_id=doc_id,
                    collection=collection,
                    filename=doc.filename,
                    file_type=doc.file_type,
                    chunk_count=len(chunks),
                    token_count=token_count,
                    metadata=dict(doc.metadata),
                )

                total_chunks += len(chunks)
                await update_ingestion_job(
                    job_id,
                    processed_files=i + 1,
                    total_chunks=total_chunks,
                )

            except Exception as exc:
                logger.error(f"Failed to process {file_path.name}: {exc}")
                continue
            finally:
                if file_path.exists():
                    file_path.unlink()

        await update_ingestion_job(
            job_id,
            status="completed",
            processed_files=len(file_paths),
            total_chunks=total_chunks,
        )

    except Exception as exc:
        logger.error(f"Ingestion job {job_id} failed: {exc}")
        await update_ingestion_job(job_id, status="failed", error=str(exc))
