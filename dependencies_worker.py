from dataclasses import dataclass

import asyncpg
from redis.asyncio import Redis

from clients.db import create_standalone_db_pool
from clients.redis import create_standalone_redis
from repositories.items import ItemRepository
from repositories.moderation_results import ModerationResultRepository
from services.moderation_service import ModerationService
from services.worker_service import WorkerService


@dataclass(frozen=True)
class WorkerRuntime:
    worker_service: WorkerService
    db_pool: asyncpg.Pool
    redis_client: Redis

    async def close(self) -> None:
        await self.redis_client.aclose()
        await self.db_pool.close()


async def create_worker_runtime() -> WorkerRuntime:
    db_pool = await create_standalone_db_pool()
    redis_client = await create_standalone_redis()
    item_repo = ItemRepository(db_pool, redis_client)
    mod_repo = ModerationResultRepository(db_pool, redis_client)
    moderation_service = ModerationService(item_repository=item_repo)
    worker_service = WorkerService(
        item_repository=item_repo,
        moderation_result_repository=mod_repo,
        moderation_service=moderation_service,
    )
    return WorkerRuntime(
        worker_service=worker_service,
        db_pool=db_pool,
        redis_client=redis_client,
    )
