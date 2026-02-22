from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from src.config import settings
from src.generation.cost_tracker import calculate_cost
from src.ingestion.chunker import count_tokens

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    content: str
    model: str
    tokens_used: int
    latency_ms: float
    cost_usd: float = 0.0


async def generate(
    messages: list[dict[str, str]],
    model: str | None = None,
    stream: bool = False,
) -> LLMResponse:
    model_name = model or settings.default_model

    # Route: claude-* → Anthropic, gpt-* → OpenAI, else → Ollama
    if settings.anthropic_api_key and model_name.startswith("claude"):
        return await _generate_anthropic(messages, model_name)

    if settings.openai_api_key and model_name.startswith(("gpt", "o1", "o3")):
        return await _generate_openai(messages, model_name)

    return await _generate_ollama(messages, model_name)


async def generate_stream(
    messages: list[dict[str, str]],
    model: str | None = None,
) -> AsyncIterator[str]:
    model_name = model or settings.default_model

    if settings.anthropic_api_key and model_name.startswith("claude"):
        async for chunk in _stream_anthropic(messages, model_name):
            yield chunk
        return

    if settings.openai_api_key and model_name.startswith(("gpt", "o1", "o3")):
        async for chunk in _stream_openai(messages, model_name):
            yield chunk
        return

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
        cost_usd=0.0,
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

    input_tokens = response.usage.prompt_tokens if response.usage else count_tokens(str(messages))
    output_tokens = response.usage.completion_tokens if response.usage else count_tokens(content)
    total_tokens = input_tokens + output_tokens
    cost = calculate_cost(model, input_tokens, output_tokens)

    return LLMResponse(
        content=content,
        model=model,
        tokens_used=total_tokens,
        latency_ms=elapsed,
        cost_usd=cost,
    )


async def _stream_openai(
    messages: list[dict[str, str]],
    model: str,
) -> AsyncIterator[str]:
    from openai import AsyncOpenAI

    if not settings.openai_api_key:
        raise ValueError("OpenAI API key required")

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    stream = await client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        stream=True,
    )

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


async def _generate_anthropic(
    messages: list[dict[str, str]],
    model: str,
) -> LLMResponse:
    if not settings.anthropic_api_key:
        raise ValueError("Anthropic API key required")

    start = time.perf_counter()

    # Extract system message
    system_content = ""
    api_messages: list[dict[str, str]] = []
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        else:
            api_messages.append(msg)

    body: dict[str, Any] = {
        "model": model,
        "max_tokens": 4096,
        "messages": api_messages,
    }
    if system_content:
        body["system"] = system_content

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": settings.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=body,
        )
        response.raise_for_status()
        data = response.json()

    elapsed = (time.perf_counter() - start) * 1000
    content = ""
    for block in data.get("content", []):
        if block.get("type") == "text":
            content += block.get("text", "")

    usage = data.get("usage", {})
    input_tokens = usage.get("input_tokens", count_tokens(str(messages)))
    output_tokens = usage.get("output_tokens", count_tokens(content))
    total_tokens = input_tokens + output_tokens
    cost = calculate_cost(model, input_tokens, output_tokens)

    return LLMResponse(
        content=content,
        model=model,
        tokens_used=total_tokens,
        latency_ms=elapsed,
        cost_usd=cost,
    )


async def _stream_anthropic(
    messages: list[dict[str, str]],
    model: str,
) -> AsyncIterator[str]:
    if not settings.anthropic_api_key:
        raise ValueError("Anthropic API key required")

    system_content = ""
    api_messages: list[dict[str, str]] = []
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        else:
            api_messages.append(msg)

    body: dict[str, Any] = {
        "model": model,
        "max_tokens": 4096,
        "messages": api_messages,
        "stream": True,
    }
    if system_content:
        body["system"] = system_content

    async with httpx.AsyncClient(timeout=120.0) as client, client.stream(
        "POST",
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=body,
    ) as response:
        response.raise_for_status()
        import json as json_mod
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            try:
                data = json_mod.loads(line[6:])
                if data.get("type") == "content_block_delta":
                    text = data.get("delta", {}).get("text", "")
                    if text:
                        yield text
            except Exception:
                continue


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
