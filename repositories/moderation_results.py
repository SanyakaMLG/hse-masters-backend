import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import asyncpg
from redis.asyncio import Redis

from errors import ModerationTaskNotFoundError
from models.moderation import ModerationResult
from utils.metrics import (
    observe_db_delete,
    observe_db_insert,
    observe_db_select,
    observe_db_update,
)


@dataclass(frozen=True)
class ModerationResultPostgresStorage:
    pool: asyncpg.Pool

    async def create(self, item_id: int) -> int:
        async with observe_db_insert():
            async with self.pool.acquire() as conn:
                task_id = await conn.fetchval(
                    """
                    INSERT INTO moderation_results (item_id, status)
                    VALUES ($1, 'pending')
                    RETURNING id
                    """,
                    item_id,
                )
        return task_id

    async def select_by_id(self, task_id: int) -> Optional[Mapping[str, Any]]:
        async with observe_db_select():
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
        return dict(row) if row is not None else None

    async def update(
        self,
        task_id: int,
        status: str,
        is_violation: Optional[bool] = None,
        probability: Optional[float] = None,
        error_message: Optional[str] = None,
    ) -> None:
        async with observe_db_update():
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

    async def delete_by_item_id(self, item_id: int) -> bool:
        async with observe_db_delete():
            async with self.pool.acquire() as conn:
                status = await conn.execute(
                    "DELETE FROM moderation_results WHERE item_id = $1", item_id
                )
        return status.startswith("DELETE")

    async def select_ids_by_item_id(self, item_id: int) -> list[int]:
        async with observe_db_select():
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT id FROM moderation_results WHERE item_id = $1", item_id
                )
        return [row["id"] for row in rows]

    async def select_latest_pending_task_id(self, item_id: int) -> Optional[int]:
        async with observe_db_select():
            async with self.pool.acquire() as conn:
                return await conn.fetchval(
                    """
                    SELECT id FROM moderation_results
                    WHERE item_id = $1 AND status = 'pending'
                    ORDER BY id DESC LIMIT 1
                    """,
                    item_id,
                )


@dataclass(frozen=True)
class ModerationResultRedisStorage:
    redis: Redis

    TTL_SECONDS = 60 * 60 * 24

    @staticmethod
    def _task_key(task_id: int) -> str:
        return f"moderation:task:{task_id}"

    async def get_task(self, task_id: int) -> Optional[Mapping[str, Any]]:
        data = await self.redis.get(self._task_key(task_id))
        return json.loads(data) if data else None

    async def set_task(self, task: ModerationResult) -> None:
        await self.redis.set(
            self._task_key(task.task_id),
            json.dumps(
                {
                    "task_id": task.task_id,
                    "item_id": task.item_id,
                    "status": task.status,
                    "is_violation": task.is_violation,
                    "probability": task.probability,
                    "error_message": task.error_message,
                    "created_at": task.created_at.isoformat(),
                    "processed_at": (
                        task.processed_at.isoformat()
                        if task.processed_at is not None
                        else None
                    ),
                }
            ),
            ex=self.TTL_SECONDS,
        )

    async def delete_task(self, task_id: int) -> None:
        await self.redis.delete(self._task_key(task_id))


@dataclass(frozen=True)
class ModerationResultRepository:
    storage: ModerationResultPostgresStorage
    cache_storage: Optional[ModerationResultRedisStorage]

    def __init__(self, pool: asyncpg.Pool, redis: Optional[Redis] = None) -> None:
        object.__setattr__(self, "storage", ModerationResultPostgresStorage(pool))
        object.__setattr__(
            self,
            "cache_storage",
            ModerationResultRedisStorage(redis) if redis is not None else None,
        )

    async def create_task(self, item_id: int) -> int:
        return await self.storage.create(item_id)

    async def get_task(self, task_id: int) -> ModerationResult:
        if self.cache_storage is not None:
            cached_task = await self.cache_storage.get_task(task_id)
            if cached_task is not None:
                from datetime import datetime

                return ModerationResult(
                    task_id=cached_task["task_id"],
                    item_id=cached_task["item_id"],
                    status=cached_task["status"],
                    is_violation=cached_task["is_violation"],
                    probability=cached_task["probability"],
                    error_message=cached_task["error_message"],
                    created_at=datetime.fromisoformat(cached_task["created_at"]),
                    processed_at=(
                        datetime.fromisoformat(cached_task["processed_at"])
                        if cached_task["processed_at"] is not None
                        else None
                    ),
                )

        row = await self.storage.select_by_id(task_id)
        if row is None:
            raise ModerationTaskNotFoundError("Задача не найдена")

        task = ModerationResult(
            task_id=row["id"],
            item_id=row["item_id"],
            status=row["status"],
            is_violation=row["is_violation"],
            probability=row["probability"],
            error_message=row["error_message"],
            created_at=row["created_at"],
            processed_at=row["processed_at"],
        )
        if self.cache_storage is not None:
            await self.cache_storage.set_task(task)
        return task

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
        if self.cache_storage is not None:
            await self.cache_storage.delete_task(task_id)

    async def delete_by_item_id(self, item_id: int) -> bool:
        task_ids = await self.storage.select_ids_by_item_id(item_id)
        deleted = await self.storage.delete_by_item_id(item_id)
        if self.cache_storage is not None:
            for task_id in task_ids:
                await self.cache_storage.delete_task(task_id)
        return deleted

    async def get_task_ids_by_item_id(self, item_id: int) -> list[int]:
        return await self.storage.select_ids_by_item_id(item_id)

    async def get_latest_pending_task_id(self, item_id: int) -> Optional[int]:
        return await self.storage.select_latest_pending_task_id(item_id)
