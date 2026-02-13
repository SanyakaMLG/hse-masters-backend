import os
import asyncpg
from typing import Generator
import pytest
from fastapi.testclient import TestClient
import httpx
from asgi_lifespan import LifespanManager
from pathlib import Path

from main import app
from routers.moderation import get_db_pool

import subprocess
from testcontainers.postgres import PostgresContainer


@pytest.fixture(scope="session")
def postgres_container():
    BASE_DIR = Path(__file__).parent.parent
    DB_PATH = BASE_DIR / "db"

    with PostgresContainer("postgres:15.3") as postgres:
        host = postgres.get_container_host_ip()
        port = postgres.get_exposed_port(postgres.port)
        user = postgres.username
        password = postgres.password
        dbname = postgres.dbname
        
        pg_dsn = f"host={host} port={port} user={user} password={password} dbname={dbname}"
        
        try:
            result = subprocess.run(
                ["pgmigrate", "-d", str(DB_PATH), "-c", pg_dsn, "-t", "latest", "migrate"],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            pytest.fail(f"PGMIGRATE FAILED: {e.stderr}")
            
        yield postgres


@pytest.fixture
async def db_pool(postgres_container):
    p = postgres_container
    dsn = f"postgresql://{p.username}:{p.password}@{p.get_container_host_ip()}:{p.get_exposed_port(p.port)}/{p.dbname}"
    
    pool = await asyncpg.create_pool(dsn)
    
    async with pool.acquire() as conn:
        await conn.execute("TRUNCATE users, items, moderation_results RESTART IDENTITY CASCADE")
        
    try:
        yield pool
    finally:
        await pool.close()


def pytest_runtest_setup(item):
    if "db" in item.keywords:
        dsn = os.getenv("DATABASE_URL", "")
        if "hw_test" not in dsn:
            pytest.skip(
                "DB tests are skipped: DATABASE_URL does not point to test database"
            )


@pytest.fixture(scope="session")
def trained_model():
    from model import train_model
    return train_model()


@pytest.fixture(autouse=True)
def setup_moderation_model(trained_model):
    from services.moderation_service import ModerationService
    ModerationService.model = trained_model
    yield


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
