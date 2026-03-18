from datetime import UTC, datetime

import pytest

from models.moderation import ModerationResult
from repositories.items import ItemRepository
from repositories.moderation_results import ModerationResultRepository


@pytest.mark.integration
@pytest.mark.anyio
class TestCacheRepository:
    async def test_set_and_get_prediction(self, db_pool, redis_client):
        repo = ItemRepository(db_pool, redis_client)
        item_id = 1
        data = {"is_violation": False, "probability": 0.15}

        await repo.set_prediction(item_id, data)
        cached = await repo.get_prediction(item_id)

        assert cached == data

    async def test_get_prediction_not_found(self, db_pool, redis_client):
        repo = ItemRepository(db_pool, redis_client)
        assert await repo.get_prediction(999) is None

    async def test_delete_prediction(self, db_pool, redis_client):
        repo = ItemRepository(db_pool, redis_client)
        item_id = 2
        data = {"is_violation": True, "probability": 0.95}

        await repo.set_prediction(item_id, data)
        await repo.delete_prediction(item_id)
        cached = await repo.get_prediction(item_id)

        assert cached is None

    async def test_set_and_get_task_cache(self, db_pool, redis_client):
        repo = ModerationResultRepository(db_pool, redis_client)
        task_id = 1

        task = ModerationResult(
            task_id=task_id,
            item_id=10,
            status="completed",
            is_violation=False,
            probability=0.1,
            error_message=None,
            created_at=datetime.now(UTC),
            processed_at=datetime.now(UTC),
        )

        await repo.cache_storage.set_task(task)
        cached = await repo.get_task(task_id)

        assert cached.task_id == task_id
