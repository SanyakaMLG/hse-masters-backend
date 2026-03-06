import time
from typing import Optional

import asyncpg

from models.moderation import User
from utils.metrics import DB_QUERY_DURATION


class UserRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def create_user(
        self,
        is_verified_seller: bool,
    ) -> User:
        start_time = time.time()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO users (is_verified_seller)
                VALUES ($1)
                RETURNING id, is_verified_seller
                """,
                is_verified_seller,
            )

        DB_QUERY_DURATION.labels(query_type="insert").observe(time.time() - start_time)
        return User(id=row["id"], is_verified_seller=row["is_verified_seller"])

    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        start_time = time.time()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, is_verified_seller
                FROM users
                WHERE id = $1
                """,
                user_id,
            )

        DB_QUERY_DURATION.labels(query_type="select").observe(time.time() - start_time)
        if row is None:
            return None
        return User(id=row["id"], is_verified_seller=row["is_verified_seller"])
