import hashlib
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import asyncpg

from models.moderation import Account
from utils.metrics import DB_QUERY_DURATION


@dataclass(frozen=True)
class AccountPostgresStorage:
    pool: asyncpg.Pool

    async def create(self, login: str, hashed_password: str) -> Mapping[str, Any]:
        start_time = time.time()
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

        DB_QUERY_DURATION.labels(query_type="insert").observe(time.time() - start_time)
        return dict(row)

    async def select_by_id(self, account_id: int) -> Optional[Mapping[str, Any]]:
        start_time = time.time()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, login, password, is_blocked
                FROM account
                WHERE id = $1
                """,
                account_id,
            )

        DB_QUERY_DURATION.labels(query_type="select").observe(time.time() - start_time)
        return dict(row) if row is not None else None

    async def delete(self, account_id: int) -> bool:
        start_time = time.time()
        async with self.pool.acquire() as conn:
            status = await conn.execute("DELETE FROM account WHERE id = $1", account_id)

        DB_QUERY_DURATION.labels(query_type="delete").observe(time.time() - start_time)
        return status == "DELETE 1"

    async def block(self, account_id: int) -> bool:
        start_time = time.time()
        async with self.pool.acquire() as conn:
            status = await conn.execute(
                "UPDATE account SET is_blocked = TRUE WHERE id = $1", account_id
            )

        DB_QUERY_DURATION.labels(query_type="update").observe(time.time() - start_time)
        return status == "UPDATE 1"

    async def select_by_login_password(
        self, login: str, hashed_password: str
    ) -> Optional[Mapping[str, Any]]:
        start_time = time.time()

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

        DB_QUERY_DURATION.labels(query_type="select").observe(time.time() - start_time)
        return dict(row) if row is not None else None


@dataclass(frozen=True)
class AccountRepository:
    storage: AccountPostgresStorage

    def __init__(self, pool: asyncpg.Pool) -> None:
        object.__setattr__(self, "storage", AccountPostgresStorage(pool))

    @staticmethod
    def _hash_password(password: str) -> str:
        return hashlib.md5(password.encode("utf-8")).hexdigest()

    async def create_account(self, login: str, password: str) -> Account:
        hashed_password = self._hash_password(password)
        row = await self.storage.create(login, hashed_password)
        return Account(
            id=row["id"],
            login=row["login"],
            password=row["password"],
            is_blocked=row["is_blocked"],
        )

    async def get_by_id(self, account_id: int) -> Optional[Account]:
        row = await self.storage.select_by_id(account_id)
        if row is None:
            return None

        return Account(
            id=row["id"],
            login=row["login"],
            password=row["password"],
            is_blocked=row["is_blocked"],
        )

    async def delete(self, account_id: int) -> bool:
        return await self.storage.delete(account_id)

    async def block(self, account_id: int) -> bool:
        return await self.storage.block(account_id)

    async def get_by_login_password(
        self, login: str, password: str
    ) -> Optional[Account]:
        hashed_password = self._hash_password(password)
        row = await self.storage.select_by_login_password(login, hashed_password)
        if row is None:
            return None

        return Account(
            id=row["id"],
            login=row["login"],
            password=row["password"],
            is_blocked=row["is_blocked"],
        )
