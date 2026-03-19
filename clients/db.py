import asyncio
import os
import socket
from typing import Optional

import asyncpg
from fastapi import FastAPI, HTTPException, Request


async def create_db_pool(app: FastAPI) -> None:
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        return
    last_error: Exception | None = None
    for _ in range(10):
        try:
            app.state.db_pool = await asyncpg.create_pool(dsn)
            return
        except (OSError, socket.gaierror, asyncpg.PostgresError) as exc:
            last_error = exc
            await asyncio.sleep(2)
    if last_error is not None:
        raise last_error


async def close_db_pool(app: FastAPI) -> None:
    pool: Optional[asyncpg.Pool] = getattr(app.state, "db_pool", None)
    if pool is not None:
        await pool.close()
        app.state.db_pool = None


def get_db_pool_dependency(request: Request) -> asyncpg.Pool:
    pool = getattr(request.app.state, "db_pool", None)
    if pool is None:
        raise HTTPException(
            status_code=503,
            detail="База данных не настроена",
        )
    return pool


async def create_standalone_db_pool() -> asyncpg.Pool:
    dsn = os.getenv(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/hw"
    )
    last_error: Exception | None = None
    for _ in range(10):
        try:
            return await asyncpg.create_pool(dsn)
        except (OSError, socket.gaierror, asyncpg.PostgresError) as exc:
            last_error = exc
            await asyncio.sleep(2)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to initialize standalone DB pool")
