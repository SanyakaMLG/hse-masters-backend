import pytest

from errors import UserNotFoundError
from repositories.users import User, UserRepository


@pytest.mark.integration
@pytest.mark.anyio
class TestUserRepository:
    async def test_create_user(self, db_pool, redis_client):
        repo = UserRepository(db_pool, redis_client)

        user = await repo.create_user(is_verified_seller=True)

        assert isinstance(user, User)
        assert isinstance(user.id, int)
        assert user.is_verified_seller is True

    async def test_get_user_by_id_success(self, db_pool, redis_client):
        repo = UserRepository(db_pool, redis_client)

        created = await repo.create_user(is_verified_seller=False)
        fetched = await repo.get_user_by_id(created.id)

        assert fetched is not None
        assert isinstance(fetched, User)
        assert fetched.id == created.id
        assert fetched.is_verified_seller is False

    async def test_get_user_by_id_not_found(self, db_pool, redis_client):
        repo = UserRepository(db_pool, redis_client)

        with pytest.raises(UserNotFoundError, match="Пользователь не найден"):
            await repo.get_user_by_id(999999999)
