import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock

import asyncpg
import httpx
import pytest
import redis.asyncio as redis
from asgi_lifespan import LifespanManager
from docker.errors import DockerException
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

from clients.db import get_db_pool_dependency
from clients.kafka import get_kafka_client_dependency
from clients.redis import get_redis_dependency
from dependencies import get_current_account
from main import app
from models.moderation import Account


@pytest.fixture(scope="session")
def postgres_container():
    BASE_DIR = Path(__file__).parent.parent
    MIGRATIONS_PATH = BASE_DIR / "db" / "migrations"

    try:
        with PostgresContainer("postgres:15.3") as postgres:
            host = postgres.get_container_host_ip()
            port = postgres.get_exposed_port(postgres.port)
            user = postgres.username
            password = postgres.password
            dbname = postgres.dbname

            pg_dsn = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

            async def apply_migrations() -> None:
                conn = await asyncpg.connect(pg_dsn)
                try:
                    for migration_path in sorted(MIGRATIONS_PATH.glob("V*.sql")):
                        migration_sql = migration_path.read_text(encoding="utf-8")
                        await conn.execute(migration_sql)
                finally:
                    await conn.close()

            asyncio.run(apply_migrations())

            yield postgres
    except DockerException as exc:
        pytest.skip(f"Docker is unavailable for integration tests: {exc}")


@pytest.fixture(scope="session")
def redis_container():
    try:
        with RedisContainer("redis:7-alpine") as redis_server:
            os.environ["REDIS_URL"] = (
                f"redis://{redis_server.get_container_host_ip()}:{redis_server.get_exposed_port(6379)}/0"
            )
            yield redis_server
    except DockerException as exc:
        pytest.skip(f"Docker is unavailable for integration tests: {exc}")


@pytest.fixture
async def redis_client(redis_container):
    client = redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
    await client.flushdb()
    yield client
    await client.aclose()


@pytest.fixture
async def db_pool(postgres_container):
    p = postgres_container
    dsn = f"postgresql://{p.username}:{p.password}@{p.get_container_host_ip()}:{p.get_exposed_port(p.port)}/{p.dbname}"

    pool = await asyncpg.create_pool(dsn)

    async with pool.acquire() as conn:
        await conn.execute(
            "TRUNCATE users, items, "
            "moderation_results, account RESTART IDENTITY CASCADE"
        )

    try:
        yield pool
    finally:
        await pool.close()


@pytest.fixture
def test_account() -> Account:
    return Account(id=1, login="test", password="hashed", is_blocked=False)


@pytest.fixture
async def app_async_client(test_account):
    mock_pool = AsyncMock()
    mock_redis = AsyncMock()
    mock_kafka = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock(return_value=True)
    mock_redis.delete = AsyncMock(return_value=1)

    app.dependency_overrides[get_db_pool_dependency] = lambda: mock_pool
    app.dependency_overrides[get_redis_dependency] = lambda: mock_redis
    app.dependency_overrides[get_kafka_client_dependency] = lambda: mock_kafka
    app.dependency_overrides[get_current_account] = lambda: test_account

    transport = httpx.ASGITransport(app=app)
    async with LifespanManager(app):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            yield client

    app.dependency_overrides.clear()


@pytest.fixture
async def integration_app_async_client(db_pool, redis_client, test_account):
    app.dependency_overrides[get_db_pool_dependency] = lambda: db_pool
    app.dependency_overrides[get_redis_dependency] = lambda: redis_client
    app.dependency_overrides[get_current_account] = lambda: test_account

    transport = httpx.ASGITransport(app=app)
    async with LifespanManager(app):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            yield client

    app.dependency_overrides.clear()
