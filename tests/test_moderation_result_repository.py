import pytest

from errors import ModerationTaskNotFoundError
from models.moderation import ModerationResult
from repositories.items import ItemRepository
from repositories.moderation_results import ModerationResultRepository
from repositories.users import UserRepository


@pytest.mark.integration
@pytest.mark.anyio
class TestModerationResultRepository:
    async def test_create_and_get_task_success(self, db_pool, redis_client):
        u_repo = UserRepository(db_pool, redis_client)
        i_repo = ItemRepository(db_pool)
        user = await u_repo.create_user(is_verified_seller=False)
        item = await i_repo.create_item(user.id, "Test", "Desc", 1, 0)

        repo = ModerationResultRepository(db_pool)
        task_id = await repo.create_task(item.id)

        assert isinstance(task_id, int)

        task = await repo.get_task(task_id)
        assert isinstance(task, ModerationResult)
        assert task.item_id == item.id
        assert task.status == "pending"

    async def test_get_task_not_found(self, db_pool):
        repo = ModerationResultRepository(db_pool)
        with pytest.raises(ModerationTaskNotFoundError, match="Задача не найдена"):
            await repo.get_task(99999)

    async def test_update_task_full(self, db_pool, redis_client):
        u_repo = UserRepository(db_pool, redis_client)
        i_repo = ItemRepository(db_pool)
        user = await u_repo.create_user(is_verified_seller=False)
        item = await i_repo.create_item(user.id, "Test", "Desc", 1, 0)

        repo = ModerationResultRepository(db_pool)
        task_id = await repo.create_task(item.id)

        await repo.update_task(
            task_id=task_id,
            status="completed",
            is_violation=True,
            probability=0.95,
            error_message="All good",
        )

        updated = await repo.get_task(task_id)

        assert updated is not None
        assert updated.status == "completed"
        assert updated.is_violation is True
        assert updated.probability == 0.95
        assert updated.error_message == "All good"
        assert updated.processed_at is not None

    async def test_delete_by_item_id(self, db_pool, redis_client):
        u_repo = UserRepository(db_pool, redis_client)
        i_repo = ItemRepository(db_pool)
        user = await u_repo.create_user(is_verified_seller=False)
        item = await i_repo.create_item(user.id, "Test", "Desc", 1, 0)

        repo = ModerationResultRepository(db_pool)
        await repo.create_task(item.id)
        await repo.create_task(item.id)

        deleted = await repo.delete_by_item_id(item.id)

        assert deleted is True
        async with db_pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM moderation_results WHERE item_id = $1", item.id
            )
        assert count == 0

    async def test_delete_by_item_id_when_nothing_to_delete(self, db_pool):
        repo = ModerationResultRepository(db_pool)
        deleted = await repo.delete_by_item_id(999999)
        assert deleted is True
