from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.db.models import init_db
from src.main import app


@pytest.fixture(autouse=True)
async def setup_db(tmp_path: object) -> None:
    import src.db.models as db_mod

    original = db_mod.DB_PATH
    db_mod.DB_PATH = original.parent / "test_rag_eval.db"
    db_mod.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    await init_db()
    yield  # type: ignore[misc]
    if db_mod.DB_PATH.exists():
        db_mod.DB_PATH.unlink()
    db_mod.DB_PATH = original


@pytest.fixture
async def client() -> AsyncClient:
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac  # type: ignore[misc]


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "ollama" in data
        assert "embedding_model" in data

    @pytest.mark.asyncio
    async def test_health_contains_config(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        data = response.json()
        assert "default_llm" in data
        assert "eval_enabled" in data


class TestSettingsEndpoint:
    @pytest.mark.asyncio
    async def test_settings_returns_200(self, client: AsyncClient) -> None:
        response = await client.get("/api/settings")
        assert response.status_code == 200
        data = response.json()
        assert "embedding_model" in data
        assert "chunking_strategy" in data
        assert "default_model" in data
        assert "hybrid_alpha" in data


class TestCollectionsEndpoint:
    @pytest.mark.asyncio
    async def test_list_collections(self, client: AsyncClient) -> None:
        response = await client.get("/api/collections")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestIngestEndpoint:
    @pytest.mark.asyncio
    async def test_ingest_no_files_returns_400(self, client: AsyncClient) -> None:
        response = await client.post("/api/ingest", files=[])
        assert response.status_code in (400, 422)


class TestRetrieveEndpoint:
    @pytest.mark.asyncio
    async def test_retrieve_validation(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/retrieve",
            json={"collection": "nonexistent"},
        )
        assert response.status_code == 422


class TestMetricsEndpoint:
    @pytest.mark.asyncio
    async def test_metrics_returns_200(self, client: AsyncClient) -> None:
        response = await client.get("/api/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_queries" in data

    @pytest.mark.asyncio
    async def test_metrics_with_collection_filter(self, client: AsyncClient) -> None:
        response = await client.get("/api/metrics?collection=test")
        assert response.status_code == 200
