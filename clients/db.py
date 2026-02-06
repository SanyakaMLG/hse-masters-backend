import os
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI


async def create_db_pool(app: FastAPI) -> None:
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        return

    app.state.db_pool = await asyncpg.create_pool(dsn)


async def close_db_pool(app: FastAPI) -> None:
    pool: Optional[asyncpg.Pool] = getattr(app.state, "db_pool", None)
    if pool is not None:
        await pool.close()
        app.state.db_pool = None


@asynccontextmanager
async def get_pg_connection() -> AsyncGenerator[asyncpg.Connection, None]:

    connection: asyncpg.Connection = await asyncpg.connect(
        user="postgres",
        password="postgres",
        database="hw",
        host="localhost",
        port=5432,
    )

    try:
        yield connection
    finally:
        await connection.close()

