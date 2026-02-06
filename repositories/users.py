from dataclasses import dataclass
from typing import Optional

import asyncpg


@dataclass
class User:
    id: int
    is_verified_seller: bool


class UserRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def create_user(
        self,
        is_verified_seller: bool,
    ) -> User:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO users (is_verified_seller)
                VALUES ($1)
                RETURNING id, is_verified_seller
                """,
                is_verified_seller,
            )
        return User(id=row["id"], is_verified_seller=row["is_verified_seller"])

    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, is_verified_seller
                FROM users
                WHERE id = $1
                """,
                user_id,
            )
        if row is None:
            return None
        return User(id=row["id"], is_verified_seller=row["is_verified_seller"])

