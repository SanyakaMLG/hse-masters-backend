import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import asyncpg

from models.moderation import ModerationResult
from utils.metrics import DB_QUERY_DURATION


@dataclass(frozen=True)
class ModerationResultPostgresStorage:
    pool: asyncpg.Pool

    async def create(self, item_id: int) -> int:
        start_time = time.time()
        async with self.pool.acquire() as conn:
            task_id = await conn.fetchval(
                """
                INSERT INTO moderation_results (item_id, status)
                VALUES ($1, 'pending')
                RETURNING id
                """,
                item_id,
            )
        DB_QUERY_DURATION.labels(query_type="insert").observe(time.time() - start_time)
        return task_id

    async def select_by_id(self, task_id: int) -> Optional[Mapping[str, Any]]:
        start_time = time.time()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, item_id, status, is_violation, probability,
                       error_message, created_at, processed_at
                FROM moderation_results
                WHERE id = $1
                """,
                task_id,
            )

        DB_QUERY_DURATION.labels(query_type="select").observe(time.time() - start_time)
        return dict(row) if row is not None else None

    async def update(
        self,
        task_id: int,
        status: str,
        is_violation: Optional[bool] = None,
        probability: Optional[float] = None,
        error_message: Optional[str] = None,
    ) -> None:
        start_time = time.time()
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE moderation_results
                SET status = $2,
                    is_violation = $3,
                    probability = $4,
                    error_message = $5,
                    processed_at = NOW()
                WHERE id = $1
                """,
                task_id,
                status,
                is_violation,
                probability,
                error_message,
            )

        DB_QUERY_DURATION.labels(query_type="update").observe(time.time() - start_time)

    async def delete_by_item_id(self, item_id: int) -> bool:
        start_time = time.time()
        async with self.pool.acquire() as conn:
            status = await conn.execute(
                "DELETE FROM moderation_results WHERE item_id = $1", item_id
            )

        DB_QUERY_DURATION.labels(query_type="delete").observe(time.time() - start_time)
        return status.startswith("DELETE")


@dataclass(frozen=True)
class ModerationResultRepository:
    storage: ModerationResultPostgresStorage

    def __init__(self, pool: asyncpg.Pool) -> None:
        object.__setattr__(self, "storage", ModerationResultPostgresStorage(pool))

    async def create_task(self, item_id: int) -> int:
        return await self.storage.create(item_id)

    async def get_task(self, task_id: int) -> Optional[ModerationResult]:
        row = await self.storage.select_by_id(task_id)
        if row is None:
            return None

        return ModerationResult(
            task_id=row["id"],
            item_id=row["item_id"],
            status=row["status"],
            is_violation=row["is_violation"],
            probability=row["probability"],
            error_message=row["error_message"],
            created_at=row["created_at"],
            processed_at=row["processed_at"],
        )

    async def update_task(
        self,
        task_id: int,
        status: str,
        is_violation: Optional[bool] = None,
        probability: Optional[float] = None,
        error_message: Optional[str] = None,
    ):
        await self.storage.update(
            task_id=task_id,
            status=status,
            is_violation=is_violation,
            probability=probability,
            error_message=error_message,
        )

    async def delete_by_item_id(self, item_id: int) -> bool:
        return await self.storage.delete_by_item_id(item_id)
