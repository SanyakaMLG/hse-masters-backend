from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from models.moderation import Account, ItemWithUser, ModerationResult, User
from repositories.accounts import AccountPostgresStorage, AccountRedisStorage
from repositories.items import ItemPostgresStorage, ItemRedisStorage
from repositories.moderation_results import (
    ModerationResultPostgresStorage,
    ModerationResultRedisStorage,
)
from repositories.users import UserPostgresStorage, UserRedisStorage


def make_pool_with_conn(conn: AsyncMock) -> MagicMock:
    acquire_ctx = AsyncMock()
    acquire_ctx.__aenter__.return_value = conn
    acquire_ctx.__aexit__.return_value = False
    pool = MagicMock()
    pool.acquire.return_value = acquire_ctx
    return pool


@pytest.mark.anyio
class TestStorageLayers:
    async def test_account_postgres_storage_crud_and_lookup(self):
        conn = AsyncMock()
        conn.fetchrow.side_effect = [
            {"id": 1, "login": "vasya", "password": "hashed", "is_blocked": False},
            {"id": 1, "login": "vasya", "password": "hashed", "is_blocked": False},
            {"id": 1, "login": "vasya", "password": "hashed", "is_blocked": False},
        ]
        conn.execute.side_effect = ["DELETE 1", "UPDATE 1"]
        storage = AccountPostgresStorage(make_pool_with_conn(conn))

        created = await storage.create("vasya", "hashed")
        selected = await storage.select_by_id(1)
        found = await storage.select_by_login_password("vasya", "hashed")
        deleted = await storage.delete(1)
        blocked = await storage.block(1)

        assert created["login"] == "vasya"
        assert selected["id"] == 1
        assert found["login"] == "vasya"
        assert deleted is True
        assert blocked is True

    async def test_account_redis_storage_set_and_get(self):
        redis = AsyncMock()
        redis.get.return_value = (
            '{"id":1,"login":"vasya","password":"hashed","is_blocked":false}'
        )
        storage = AccountRedisStorage(redis)

        await storage.set(
            Account(id=1, login="vasya", password="hashed", is_blocked=False)
        )
        cached = await storage.get(1)

        assert cached["login"] == "vasya"

    async def test_item_postgres_storage_create_select_and_close(self):
        conn = AsyncMock()
        conn.fetchrow.side_effect = [
            {
                "id": 1,
                "user_id": 2,
                "name": "n",
                "description": "d",
                "category": 3,
                "images_qty": 4,
            },
            {
                "item_id": 1,
                "seller_id": 2,
                "is_verified_seller": True,
                "name": "n",
                "description": "d",
                "category": 3,
                "images_qty": 4,
            },
        ]
        conn.execute.return_value = "UPDATE 1"
        storage = ItemPostgresStorage(make_pool_with_conn(conn))

        created = await storage.create(2, "n", "d", 3, 4)
        selected = await storage.select_item_with_user(1)
        closed = await storage.close(1)

        assert created["user_id"] == 2
        assert selected["seller_id"] == 2
        assert closed is True

    async def test_item_redis_storage_set_and_get(self):
        redis = AsyncMock()
        redis.get.side_effect = [
            (
                '{"item_id":1,"seller_id":2,"is_verified_seller":true,'
                '"name":"n","description":"d","category":3,"images_qty":4}'
            ),
            '{"is_violation":true,"probability":0.9}',
        ]
        storage = ItemRedisStorage(redis)

        await storage.set_item_with_user(
            ItemWithUser(
                item_id=1,
                seller_id=2,
                is_verified_seller=True,
                name="n",
                description="d",
                category=3,
                images_qty=4,
            )
        )
        item = await storage.get_item_with_user(1)
        await storage.set_prediction(1, {"is_violation": True, "probability": 0.9})
        prediction = await storage.get_prediction(1)

        assert item["seller_id"] == 2
        assert prediction["is_violation"] is True

    async def test_moderation_result_postgres_storage_full_flow(self):
        conn = AsyncMock()
        conn.fetchval.side_effect = [1, 7]
        conn.fetchrow.return_value = {
            "id": 1,
            "item_id": 2,
            "status": "completed",
            "is_violation": True,
            "probability": 0.9,
            "error_message": None,
            "created_at": "2024-01-01T00:00:00",
            "processed_at": "2024-01-01T00:01:00",
        }
        conn.fetch.return_value = [{"id": 1}, {"id": 2}]
        conn.execute.side_effect = [None, "DELETE 2"]
        storage = ModerationResultPostgresStorage(make_pool_with_conn(conn))

        created = await storage.create(2)
        selected = await storage.select_by_id(1)
        await storage.update(1, status="completed")
        deleted = await storage.delete_by_item_id(2)
        task_ids = await storage.select_ids_by_item_id(2)
        latest = await storage.select_latest_pending_task_id(2)

        assert created == 1
        assert selected["item_id"] == 2
        assert deleted is True
        assert task_ids == [1, 2]
        assert latest == 7

    async def test_moderation_result_redis_storage_set_and_get(self):
        redis = AsyncMock()
        redis.get.return_value = (
            '{"task_id":1,"item_id":2,"status":"completed","is_violation":true,'
            '"probability":0.9,"error_message":null,'
            '"created_at":"2024-01-01T00:00:00+00:00",'
            '"processed_at":"2024-01-01T00:01:00+00:00"}'
        )
        storage = ModerationResultRedisStorage(redis)
        task = ModerationResult(
            task_id=1,
            item_id=2,
            status="completed",
            is_violation=True,
            probability=0.9,
            error_message=None,
            created_at=datetime.now(UTC),
            processed_at=datetime.now(UTC),
        )

        await storage.set_task(task)
        cached = await storage.get_task(1)

        assert cached["task_id"] == 1

    async def test_user_postgres_and_redis_storage(self):
        conn = AsyncMock()
        conn.fetchrow.side_effect = [
            {"id": 1, "is_verified_seller": True},
            {"id": 1, "is_verified_seller": True},
        ]
        pg_storage = UserPostgresStorage(make_pool_with_conn(conn))
        created = await pg_storage.create(True)
        selected = await pg_storage.select_by_id(1)

        redis = AsyncMock()
        redis.get.return_value = '{"id":1,"is_verified_seller":true}'
        redis_storage = UserRedisStorage(redis)
        await redis_storage.set(User(id=1, is_verified_seller=True))
        cached = await redis_storage.get(1)

        assert created["id"] == 1
        assert selected["is_verified_seller"] is True
        assert cached["is_verified_seller"] is True
