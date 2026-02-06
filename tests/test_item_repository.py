import os
import pytest

from repositories.items import (
    ItemRepository,
    Item,
    ItemWithUser,
)
from repositories.users import UserRepository


@pytest.mark.db
@pytest.mark.anyio
@pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL is not set, DB-dependent tests are skipped",
)
class TestItemRepository:
    async def test_create_item(self, db_pool):
        user_repo = UserRepository(db_pool)
        item_repo = ItemRepository(db_pool)

        user = await user_repo.create_user(is_verified_seller=False)

        item = await item_repo.create_item(
            user_id=user.id,
            name="Test Item",
            description="Short description",
            category=1,
            images_qty=2,
        )

        assert isinstance(item, Item)
        assert isinstance(item.id, int)
        assert item.user_id == user.id
        assert item.name == "Test Item"
        assert item.description == "Short description"
        assert item.category == 1
        assert item.images_qty == 2

    async def test_get_item_with_user_success(self, db_pool):
        user_repo = UserRepository(db_pool)
        item_repo = ItemRepository(db_pool)

        user = await user_repo.create_user(is_verified_seller=True)
        item = await item_repo.create_item(
            user_id=user.id,
            name="Item with user",
            description="Description",
            category=3,
            images_qty=0,
        )

        result = await item_repo.get_item_with_user(item.id)

        assert result is not None
        assert isinstance(result, ItemWithUser)

        assert result.item_id == item.id
        assert result.seller_id == user.id
        assert result.is_verified_seller is True
        assert result.name == item.name
        assert result.description == item.description
        assert result.category == item.category
        assert result.images_qty == item.images_qty

    async def test_get_item_with_user_not_found(self, db_pool):
        item_repo = ItemRepository(db_pool)

        result = await item_repo.get_item_with_user(999999999)

        assert result is None
