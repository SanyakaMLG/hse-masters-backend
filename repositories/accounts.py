import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import asyncpg
from redis.asyncio import Redis

from errors import AccountNotFoundError
from models.moderation import Account
from utils.metrics import (
    observe_cache_hit,
    observe_cache_miss,
    observe_db_delete,
    observe_db_insert,
    observe_db_select,
    observe_db_update,
)


@dataclass(frozen=True)
class AccountPostgresStorage:
    pool: asyncpg.Pool

    async def create(self, login: str, hashed_password: str) -> Mapping[str, Any]:
        async with observe_db_insert():
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO account (login, password)
                    VALUES ($1, $2)
                    RETURNING id, login, password, is_blocked
                    """,
                    login,
                    hashed_password,
                )
        return dict(row)

    async def select_by_id(self, account_id: int) -> Optional[Mapping[str, Any]]:
        async with observe_db_select():
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT id, login, password, is_blocked
                    FROM account
                    WHERE id = $1
                    """,
                    account_id,
                )
        return dict(row) if row is not None else None

    async def delete(self, account_id: int) -> bool:
        async with observe_db_delete():
            async with self.pool.acquire() as conn:
                status = await conn.execute(
                    "DELETE FROM account WHERE id = $1", account_id
                )
        return status == "DELETE 1"

    async def block(self, account_id: int) -> bool:
        async with observe_db_update():
            async with self.pool.acquire() as conn:
                status = await conn.execute(
                    "UPDATE account SET is_blocked = TRUE WHERE id = $1", account_id
                )
        return status == "UPDATE 1"

    async def select_by_login_password(
        self, login: str, hashed_password: str
    ) -> Optional[Mapping[str, Any]]:
        async with observe_db_select():
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT id, login, password, is_blocked
                    FROM account
                    WHERE login = $1 AND password = $2
                    """,
                    login,
                    hashed_password,
                )
        return dict(row) if row is not None else None


@dataclass(frozen=True)
class AccountRedisStorage:
    redis: Redis

    TTL_SECONDS = 60 * 60 * 24

    @staticmethod
    def _key(account_id: int) -> str:
        return f"account:{account_id}"

    async def get(self, account_id: int) -> Optional[Mapping[str, Any]]:
        data = await self.redis.get(self._key(account_id))
        return json.loads(data) if data else None

    async def set(self, account: Account) -> None:
        await self.redis.set(
            self._key(account.id),
            json.dumps(
                {
                    "id": account.id,
                    "login": account.login,
                    "password": account.password,
                    "is_blocked": account.is_blocked,
                }
            ),
            ex=self.TTL_SECONDS,
        )

    async def delete(self, account_id: int) -> None:
        await self.redis.delete(self._key(account_id))


@dataclass(frozen=True)
class AccountRepository:
    storage: AccountPostgresStorage
    cache_storage: Optional[AccountRedisStorage]
    ACCOUNT_BY_ID_CACHE = "account_by_id"

    def __init__(self, pool: asyncpg.Pool, redis: Optional[Redis] = None) -> None:
        object.__setattr__(self, "storage", AccountPostgresStorage(pool))
        object.__setattr__(
            self,
            "cache_storage",
            AccountRedisStorage(redis) if redis is not None else None,
        )

    @staticmethod
    def _hash_password(password: str) -> str:
        return hashlib.md5(password.encode("utf-8")).hexdigest()

    async def create_account(self, login: str, password: str) -> Account:
        hashed_password = self._hash_password(password)
        row = await self.storage.create(login, hashed_password)
        account = Account(
            id=row["id"],
            login=row["login"],
            password=row["password"],
            is_blocked=row["is_blocked"],
        )
        if self.cache_storage is not None:
            await self.cache_storage.set(account)
        return account

    async def get_by_id(self, account_id: int) -> Account:
        if self.cache_storage is not None:
            cached_account = await self.cache_storage.get(account_id)
            if cached_account is not None:
                observe_cache_hit(self.ACCOUNT_BY_ID_CACHE)
                return Account(
                    id=cached_account["id"],
                    login=cached_account["login"],
                    password=cached_account["password"],
                    is_blocked=cached_account["is_blocked"],
                )
            observe_cache_miss(self.ACCOUNT_BY_ID_CACHE)

        row = await self.storage.select_by_id(account_id)
        if row is None:
            raise AccountNotFoundError("Пользователь не найден")

        account = Account(
            id=row["id"],
            login=row["login"],
            password=row["password"],
            is_blocked=row["is_blocked"],
        )
        if self.cache_storage is not None:
            await self.cache_storage.set(account)
        return account

    async def delete(self, account_id: int) -> bool:
        deleted = await self.storage.delete(account_id)
        if deleted and self.cache_storage is not None:
            await self.cache_storage.delete(account_id)
        return deleted

    async def block(self, account_id: int) -> bool:
        blocked = await self.storage.block(account_id)
        if blocked and self.cache_storage is not None:
            await self.cache_storage.delete(account_id)
        return blocked

    async def get_by_login_password(self, login: str, password: str) -> Account:
        hashed_password = self._hash_password(password)
        row = await self.storage.select_by_login_password(login, hashed_password)
        if row is None:
            raise AccountNotFoundError("Пользователь не найден")

        account = Account(
            id=row["id"],
            login=row["login"],
            password=row["password"],
            is_blocked=row["is_blocked"],
        )
        if self.cache_storage is not None:
            await self.cache_storage.set(account)
        return account
