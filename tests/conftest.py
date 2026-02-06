import os
import asyncpg
from typing import Generator
import pytest
from fastapi.testclient import TestClient
import httpx
from asgi_lifespan import LifespanManager

from main import app
from routers.moderation import get_db_pool

def pytest_runtest_setup(item):
    if "db" in item.keywords:
        dsn = os.getenv("DATABASE_URL", "")
        if "hw_test" not in dsn:
            pytest.skip(
                "DB tests are skipped: DATABASE_URL does not point to test database"
            )

@pytest.fixture
async def db_pool():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        pytest.skip("DATABASE_URL is not set")

    pool = await asyncpg.create_pool(dsn)
    try:
        yield pool
    finally:
        await pool.close()


@pytest.fixture
async def app_async_client(db_pool):
    async def _override_get_db_pool():
        return db_pool

    app.dependency_overrides[get_db_pool] = _override_get_db_pool

    transport = httpx.ASGITransport(app=app)
    async with LifespanManager(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

    app.dependency_overrides.clear()


@pytest.fixture
def app_client() -> Generator[TestClient, None, None]:
    return TestClient(app)
