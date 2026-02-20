import os
import subprocess
from pathlib import Path

import asyncpg
import httpx
import pytest
import redis.asyncio as redis
from asgi_lifespan import LifespanManager
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

from main import app
from routers.moderation import get_db_pool_dependency


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

        pg_dsn = (
            f"host={host} port={port} user={user} password={password} dbname={dbname}"
        )

        subprocess.run(
            ["pgmigrate", "-d", str(DB_PATH), "-c", pg_dsn, "-t", "latest", "migrate"],
            check=True,
            capture_output=True,
            text=True,
        )

        yield postgres


@pytest.fixture(scope="session")
def redis_container():
    with RedisContainer("redis:7-alpine") as redis_server:
        os.environ["REDIS_URL"] = (
            f"redis://{redis_server.get_container_host_ip()}:{redis_server.get_exposed_port(6379)}/0"
        )
        yield redis_server


@pytest.fixture
async def redis_client(redis_container):
    client = redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
    await client.flushdb()
    yield client
    await client.close()


@pytest.fixture
async def db_pool(postgres_container):
    p = postgres_container
    dsn = f"postgresql://{p.username}:{p.password}@{p.get_container_host_ip()}:{p.get_exposed_port(p.port)}/{p.dbname}"

    pool = await asyncpg.create_pool(dsn)

    async with pool.acquire() as conn:
        await conn.execute(
            "TRUNCATE users, items, moderation_results RESTART IDENTITY CASCADE"
        )

    try:
        yield pool
    finally:
        await pool.close()


@pytest.fixture
async def app_async_client(db_pool, redis_container):
    async def _override_get_db_pool():
        return db_pool

    app.dependency_overrides[get_db_pool_dependency] = _override_get_db_pool

    transport = httpx.ASGITransport(app=app)
    async with LifespanManager(app):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            yield client

    app.dependency_overrides.clear()
