import os
import pytest

from repositories.users import UserRepository, User


@pytest.mark.db
@pytest.mark.anyio
@pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL is not set, DB-dependent tests are skipped",
)
class TestUserRepository:
    async def test_create_user(self, db_pool):
        repo = UserRepository(db_pool)

        user = await repo.create_user(is_verified_seller=True)

        assert isinstance(user, User)
        assert isinstance(user.id, int)
        assert user.is_verified_seller is True

    async def test_get_user_by_id_success(self, db_pool):
        repo = UserRepository(db_pool)

        created = await repo.create_user(is_verified_seller=False)
        fetched = await repo.get_user_by_id(created.id)

        assert fetched is not None
        assert isinstance(fetched, User)
        assert fetched.id == created.id
        assert fetched.is_verified_seller is False

    async def test_get_user_by_id_not_found(self, db_pool):
        repo = UserRepository(db_pool)

        user = await repo.get_user_by_id(999999999)

        assert user is None
