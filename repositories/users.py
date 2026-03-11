import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import asyncpg

from models.moderation import User
from utils.metrics import DB_QUERY_DURATION


@dataclass(frozen=True)
class UserPostgresStorage:
    pool: asyncpg.Pool

    async def create(self, is_verified_seller: bool) -> Mapping[str, Any]:
        start_time = time.time()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO users (is_verified_seller)
                VALUES ($1)
                RETURNING id, is_verified_seller
                """,
                is_verified_seller,
            )

        DB_QUERY_DURATION.labels(query_type="insert").observe(time.time() - start_time)
        return dict(row)

    async def select_by_id(self, user_id: int) -> Optional[Mapping[str, Any]]:
        start_time = time.time()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, is_verified_seller
                FROM users
                WHERE id = $1
                """,
                user_id,
            )

        DB_QUERY_DURATION.labels(query_type="select").observe(time.time() - start_time)
        return dict(row) if row is not None else None


@dataclass(frozen=True)
class UserRepository:
    storage: UserPostgresStorage

    def __init__(self, pool: asyncpg.Pool) -> None:
        object.__setattr__(self, "storage", UserPostgresStorage(pool))

    async def create_user(self, is_verified_seller: bool) -> User:
        row = await self.storage.create(is_verified_seller)
        return User(id=row["id"], is_verified_seller=row["is_verified_seller"])

    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        row = await self.storage.select_by_id(user_id)
        if row is None:
            return None
        return User(id=row["id"], is_verified_seller=row["is_verified_seller"])