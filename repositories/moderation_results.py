import time
from typing import Optional

import asyncpg

from models.moderation import ModerationResult
from utils.metrics import DB_QUERY_DURATION


class ModerationResultRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def create_task(self, item_id: int) -> int:
        start_time = time.time()
        async with self._pool.acquire() as conn:
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

    async def get_task(self, task_id: int) -> Optional[ModerationResult]:
        start_time = time.time()
        async with self._pool.acquire() as conn:
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
        start_time = time.time()
        async with self._pool.acquire() as conn:
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
