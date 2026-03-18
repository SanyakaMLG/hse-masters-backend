import hashlib

import pytest

from errors import AccountNotFoundError
from repositories.accounts import AccountRepository


@pytest.mark.integration
@pytest.mark.anyio
class TestAccountRepository:
    async def test_create_get_and_delete_account(self, db_pool):
        repo = AccountRepository(db_pool)

        created = await repo.create_account(login="vasya", password="qwerty")
        fetched = await repo.get_by_id(created.id)

        assert fetched is not None
        assert fetched.login == "vasya"
        assert fetched.password == hashlib.md5(b"qwerty").hexdigest()
        assert fetched.is_blocked is False

        deleted = await repo.delete(created.id)
        assert deleted is True
        with pytest.raises(AccountNotFoundError, match="Пользователь не найден"):
            await repo.get_by_id(created.id)

    async def test_block_and_find_by_login_password(self, db_pool):
        repo = AccountRepository(db_pool)
        created = await repo.create_account(login="petya", password="secret")

        found = await repo.get_by_login_password("petya", "secret")
        assert found is not None
        assert found.id == created.id

        blocked = await repo.block(created.id)
        assert blocked is True

        blocked_account = await repo.get_by_id(created.id)
        assert blocked_account is not None
        assert blocked_account.is_blocked is True

    async def test_get_by_login_password_not_found(self, db_pool):
        repo = AccountRepository(db_pool)
        with pytest.raises(AccountNotFoundError, match="Пользователь не найден"):
            await repo.get_by_login_password("missing", "missing")
