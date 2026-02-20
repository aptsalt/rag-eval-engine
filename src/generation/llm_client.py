from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from src.config import settings
from src.ingestion.chunker import count_tokens

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    content: str
    model: str
    tokens_used: int
    latency_ms: float


async def generate(
    messages: list[dict[str, str]],
    model: str | None = None,
    stream: bool = False,
) -> LLMResponse:
    model_name = model or settings.default_model

    if settings.openai_api_key and model_name.startswith("gpt"):
        return await _generate_openai(messages, model_name)

    return await _generate_ollama(messages, model_name)


async def generate_stream(
    messages: list[dict[str, str]],
    model: str | None = None,
) -> AsyncIterator[str]:
    model_name = model or settings.default_model

    async for chunk in _stream_ollama(messages, model_name):
        yield chunk


async def _generate_ollama(
    messages: list[dict[str, str]],
    model: str,
) -> LLMResponse:
    start = time.perf_counter()

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{settings.ollama_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()

    elapsed = (time.perf_counter() - start) * 1000
    content = data.get("message", {}).get("content", "")
    tokens = count_tokens(content)

    return LLMResponse(
        content=content,
        model=model,
        tokens_used=tokens,
        latency_ms=elapsed,
    )


async def _stream_ollama(
    messages: list[dict[str, str]],
    model: str,
) -> AsyncIterator[str]:
    async with httpx.AsyncClient(timeout=120.0) as client, client.stream(
        "POST",
        f"{settings.ollama_url}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": True,
        },
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line:
                continue
            import json
            data = json.loads(line)
            content = data.get("message", {}).get("content", "")
            if content:
                yield content
            if data.get("done", False):
                break


async def _generate_openai(
    messages: list[dict[str, str]],
    model: str,
) -> LLMResponse:
    from openai import AsyncOpenAI

    if not settings.openai_api_key:
        raise ValueError("OpenAI API key required")

    start = time.perf_counter()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    response = await client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
    )

    elapsed = (time.perf_counter() - start) * 1000
    content = response.choices[0].message.content or ""
    tokens = response.usage.total_tokens if response.usage else count_tokens(content)

    return LLMResponse(
        content=content,
        model=model,
        tokens_used=tokens,
        latency_ms=elapsed,
    )


async def check_ollama_health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.ollama_url}/api/tags")
            return response.status_code == 200
    except Exception:
        return False


async def list_ollama_models() -> list[dict[str, Any]]:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.ollama_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])  # type: ignore[no-any-return]
    except Exception:
        return []
