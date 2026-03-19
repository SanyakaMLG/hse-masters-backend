from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from prometheus_client import REGISTRY

from models.moderation import User
from repositories.accounts import AccountRedisStorage, AccountRepository
from repositories.items import ItemRedisStorage, ItemRepository
from repositories.moderation_results import (
    ModerationResultRedisStorage,
    ModerationResultRepository,
)
from repositories.users import UserRedisStorage, UserRepository


def _cache_metric_value(cache_name: str, result: str) -> float:
    return (
        REGISTRY.get_sample_value(
            "cache_requests_total",
            labels={"cache_name": cache_name, "result": result},
        )
        or 0.0
    )


@pytest.mark.anyio
class TestRepositoryCaching:
    async def test_account_redis_storage_get_miss_and_delete(self):
        redis = AsyncMock()
        redis.get.return_value = None
        storage = AccountRedisStorage(redis)

        assert await storage.get(1) is None

        await storage.delete(1)
        redis.delete.assert_awaited_once_with("account:1")

    async def test_account_repository_reads_from_cache(self):
        repo = AccountRepository(MagicMock())
        object.__setattr__(repo, "storage", AsyncMock())
        object.__setattr__(repo, "cache_storage", AsyncMock())
        repo.cache_storage.get.return_value = {
            "id": 1,
            "login": "vasya",
            "password": "hashed",
            "is_blocked": False,
        }

        account = await repo.get_by_id(1)

        assert account.login == "vasya"
        repo.storage.select_by_id.assert_not_awaited()
        assert _cache_metric_value("account_by_id", "hit") >= 1

    async def test_account_repository_writes_and_invalidates_cache(self):
        repo = AccountRepository(MagicMock())
        object.__setattr__(repo, "storage", AsyncMock())
        object.__setattr__(repo, "cache_storage", AsyncMock())
        repo.storage.create.return_value = {
            "id": 1,
            "login": "vasya",
            "password": "hashed",
            "is_blocked": False,
        }
        repo.storage.delete.return_value = True
        repo.storage.block.return_value = True

        await repo.create_account("vasya", "qwerty")
        await repo.delete(1)
        await repo.block(1)

        assert repo.cache_storage.set.await_count == 1
        assert repo.cache_storage.delete.await_count == 2

    async def test_account_repository_create_account_sets_cache(self):
        repo = AccountRepository(MagicMock())
        object.__setattr__(repo, "storage", AsyncMock())
        object.__setattr__(repo, "cache_storage", AsyncMock())
        repo.storage.create.return_value = {
            "id": 7,
            "login": "petya",
            "password": "hashed",
            "is_blocked": False,
        }

        account = await repo.create_account("petya", "secret")

        repo.cache_storage.set.assert_awaited_once()
        cached_account = repo.cache_storage.set.await_args.args[0]
        assert cached_account.id == account.id
        assert cached_account.login == "petya"

    async def test_item_redis_storage_prediction_and_item_miss(self):
        redis = AsyncMock()
        redis.get.return_value = None
        storage = ItemRedisStorage(redis)

        assert await storage.get_item_with_user(1) is None
        assert await storage.get_prediction(1) is None

        await storage.delete_item_with_user(1)
        await storage.delete_prediction(1)

        assert redis.delete.await_count == 2

    async def test_item_repository_uses_cache_and_prediction_helpers(self):
        repo = ItemRepository(MagicMock())
        object.__setattr__(repo, "storage", AsyncMock())
        object.__setattr__(repo, "cache_storage", AsyncMock())
        repo.cache_storage.get_item_with_user.return_value = {
            "item_id": 1,
            "seller_id": 2,
            "is_verified_seller": True,
            "name": "n",
            "description": "d",
            "category": 3,
            "images_qty": 4,
        }
        repo.cache_storage.get_prediction.return_value = {"is_violation": False}

        item = await repo.get_item_with_user(1)
        prediction = await repo.get_prediction(1)
        await repo.set_prediction(1, {"is_violation": True})
        await repo.delete_prediction(1)

        assert item.seller_id == 2
        assert prediction == {"is_violation": False}
        repo.storage.select_item_with_user.assert_not_awaited()
        repo.cache_storage.set_prediction.assert_awaited_once()
        repo.cache_storage.delete_prediction.assert_awaited_once_with(1)
        assert _cache_metric_value("item_with_user", "hit") >= 1
        assert _cache_metric_value("prediction", "hit") >= 1

    async def test_item_repository_sets_cache_after_db_fetch_and_close_invalidates(
        self,
    ):
        repo = ItemRepository(MagicMock())
        object.__setattr__(repo, "storage", AsyncMock())
        object.__setattr__(repo, "cache_storage", AsyncMock())
        repo.cache_storage.get_item_with_user.return_value = None
        repo.storage.select_item_with_user.return_value = {
            "item_id": 1,
            "seller_id": 2,
            "is_verified_seller": True,
            "name": "n",
            "description": "d",
            "category": 3,
            "images_qty": 4,
        }
        repo.storage.close.return_value = True

        await repo.get_item_with_user(1)
        await repo.close_item(1)

        repo.cache_storage.set_item_with_user.assert_awaited_once()
        repo.cache_storage.delete_item_with_user.assert_awaited_once_with(1)
        repo.cache_storage.delete_prediction.assert_awaited_once_with(1)
        assert _cache_metric_value("item_with_user", "miss") >= 1

    async def test_item_repository_without_cache_returns_none_prediction(self):
        repo = ItemRepository(MagicMock())
        object.__setattr__(repo, "cache_storage", None)

        assert await repo.get_prediction(1) is None
        await repo.set_prediction(1, {"ok": True})
        await repo.delete_prediction(1)

    async def test_moderation_result_redis_storage_get_and_delete(self):
        redis = AsyncMock()
        redis.get.return_value = None
        storage = ModerationResultRedisStorage(redis)

        assert await storage.get_task(1) is None
        await storage.delete_task(1)
        redis.delete.assert_awaited_once_with("moderation:task:1")

    async def test_moderation_result_repository_uses_cache_and_invalidation(self):
        repo = ModerationResultRepository(MagicMock())
        object.__setattr__(repo, "storage", AsyncMock())
        object.__setattr__(repo, "cache_storage", AsyncMock())
        now = datetime.now(UTC)
        repo.cache_storage.get_task.return_value = {
            "task_id": 1,
            "item_id": 2,
            "status": "completed",
            "is_violation": True,
            "probability": 0.9,
            "error_message": None,
            "created_at": now.isoformat(),
            "processed_at": now.isoformat(),
        }
        repo.storage.update.return_value = None
        repo.storage.select_ids_by_item_id.return_value = [1, 2]
        repo.storage.delete_by_item_id.return_value = True
        repo.storage.select_latest_pending_task_id.return_value = 7

        task = await repo.get_task(1)
        await repo.update_task(1, status="failed")
        deleted = await repo.delete_by_item_id(10)
        task_ids = await repo.get_task_ids_by_item_id(10)
        latest = await repo.get_latest_pending_task_id(10)

        assert task.task_id == 1
        assert deleted is True
        assert task_ids == [1, 2]
        assert latest == 7
        assert repo.cache_storage.delete_task.await_count == 3

    async def test_moderation_result_repository_sets_cache_after_db_fetch(self):
        repo = ModerationResultRepository(MagicMock())
        object.__setattr__(repo, "storage", AsyncMock())
        object.__setattr__(repo, "cache_storage", AsyncMock())
        repo.cache_storage.get_task.return_value = None
        now = datetime.now(UTC)
        repo.storage.select_by_id.return_value = {
            "id": 1,
            "item_id": 2,
            "status": "completed",
            "is_violation": True,
            "probability": 0.9,
            "error_message": None,
            "created_at": now,
            "processed_at": now,
        }

        task = await repo.get_task(1)

        assert task.task_id == 1
        repo.cache_storage.set_task.assert_awaited_once()

    async def test_user_redis_storage_and_repository_db_fill(self):
        redis = AsyncMock()
        redis.get.return_value = None
        storage = UserRedisStorage(redis)
        assert await storage.get(1) is None

        repo = UserRepository(MagicMock())
        object.__setattr__(repo, "storage", AsyncMock())
        object.__setattr__(repo, "cache_storage", AsyncMock())
        repo.cache_storage.get.return_value = None
        repo.storage.select_by_id.return_value = {"id": 1, "is_verified_seller": True}

        user = await repo.get_user_by_id(1)

        assert isinstance(user, User)
        repo.cache_storage.set.assert_awaited_once()
        assert _cache_metric_value("user_by_id", "miss") >= 1
