import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import asyncpg
from redis.asyncio import Redis

from errors import UserNotFoundError
from models.moderation import User
from utils.metrics import (
    observe_cache_hit,
    observe_cache_miss,
    observe_db_insert,
    observe_db_select,
)


@dataclass(frozen=True)
class UserPostgresStorage:
    pool: asyncpg.Pool

    async def create(self, is_verified_seller: bool) -> Mapping[str, Any]:
        async with observe_db_insert():
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO users (is_verified_seller)
                    VALUES ($1)
                    RETURNING id, is_verified_seller
                    """,
                    is_verified_seller,
                )
        return dict(row)

    async def select_by_id(self, user_id: int) -> Optional[Mapping[str, Any]]:
        async with observe_db_select():
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT id, is_verified_seller
                    FROM users
                    WHERE id = $1
                    """,
                    user_id,
                )
        return dict(row) if row is not None else None


@dataclass(frozen=True)
class UserRedisStorage:
    redis: Redis

    TTL_SECONDS = 60 * 60 * 24

    @staticmethod
    def _key(user_id: int) -> str:
        return f"user:{user_id}"

    async def get(self, user_id: int) -> Optional[Mapping[str, Any]]:
        data = await self.redis.get(self._key(user_id))
        return json.loads(data) if data else None

    async def set(self, user: User) -> None:
        await self.redis.set(
            self._key(user.id),
            json.dumps(
                {
                    "id": user.id,
                    "is_verified_seller": user.is_verified_seller,
                }
            ),
            ex=self.TTL_SECONDS,
        )


@dataclass(frozen=True)
class UserRepository:
    storage: UserPostgresStorage
    cache_storage: Optional[UserRedisStorage]
    USER_BY_ID_CACHE = "user_by_id"

    def __init__(self, pool: asyncpg.Pool, redis: Optional[Redis] = None) -> None:
        object.__setattr__(self, "storage", UserPostgresStorage(pool))
        object.__setattr__(
            self,
            "cache_storage",
            UserRedisStorage(redis) if redis is not None else None,
        )

    async def create_user(self, is_verified_seller: bool) -> User:
        row = await self.storage.create(is_verified_seller)
        user = User(id=row["id"], is_verified_seller=row["is_verified_seller"])
        if self.cache_storage is not None:
            await self.cache_storage.set(user)
        return user

    async def get_user_by_id(self, user_id: int) -> User:
        if self.cache_storage is not None:
            cached_user = await self.cache_storage.get(user_id)
            if cached_user is not None:
                observe_cache_hit(self.USER_BY_ID_CACHE)
                return User(
                    id=cached_user["id"],
                    is_verified_seller=cached_user["is_verified_seller"],
                )
            observe_cache_miss(self.USER_BY_ID_CACHE)

        row = await self.storage.select_by_id(user_id)
        if row is None:
            raise UserNotFoundError("Пользователь не найден")
        user = User(id=row["id"], is_verified_seller=row["is_verified_seller"])
        if self.cache_storage is not None:
            await self.cache_storage.set(user)
        return user
