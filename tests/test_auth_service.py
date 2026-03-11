from unittest.mock import AsyncMock

import jwt
import pytest

from models.moderation import Account
from services.auth_service import (
    AuthService,
    BlockedAccountError,
    InvalidCredentialsError,
    InvalidTokenError,
)


@pytest.mark.anyio
class TestAuthService:
    async def test_login_success(self, monkeypatch):
        monkeypatch.setenv("JWT_SECRET", "test-secret")
        repo = AsyncMock()
        repo.get_by_login_password.return_value = Account(
            id=10, login="vasya", password="hashed", is_blocked=False
        )

        service = AuthService(repo)
        token = await service.login("vasya", "password")

        payload = jwt.decode(token, "test-secret", algorithms=["HS256"])
        assert payload["sub"] == "10"
        assert payload["login"] == "vasya"

    async def test_login_invalid_credentials(self):
        repo = AsyncMock()
        repo.get_by_login_password.return_value = None

        service = AuthService(repo)
        with pytest.raises(InvalidCredentialsError):
            await service.login("vasya", "wrong")

    async def test_login_blocked_account(self):
        repo = AsyncMock()
        repo.get_by_login_password.return_value = Account(
            id=10, login="vasya", password="hashed", is_blocked=True
        )

        service = AuthService(repo)
        with pytest.raises(BlockedAccountError):
            await service.login("vasya", "password")

    async def test_get_account_from_token(self, monkeypatch):
        monkeypatch.setenv("JWT_SECRET", "test-secret")
        repo = AsyncMock()
        service = AuthService(repo)

        token = jwt.encode({"sub": "1", "login": "u"}, "test-secret", algorithm="HS256")
        repo.get_by_id.return_value = Account(id=1, login="u", password="h", is_blocked=False)

        account = await service.get_account_from_token(token)
        assert account.id == 1

    async def test_get_account_from_token_invalid(self):
        repo = AsyncMock()
        service = AuthService(repo)

        with pytest.raises(InvalidTokenError):
            await service.get_account_from_token("broken")

    async def test_get_account_from_token_missing_sub(self, monkeypatch):
        monkeypatch.setenv("JWT_SECRET", "test-secret")
        repo = AsyncMock()
        service = AuthService(repo)

        token = jwt.encode({"login": "u"}, "test-secret", algorithm="HS256")
        with pytest.raises(InvalidTokenError, match="Некорректный токен"):
            await service.get_account_from_token(token)

    async def test_get_account_from_token_account_not_found(self, monkeypatch):
        monkeypatch.setenv("JWT_SECRET", "test-secret")
        repo = AsyncMock()
        repo.get_by_id.return_value = None
        service = AuthService(repo)

        token = jwt.encode({"sub": "111", "login": "u"}, "test-secret", algorithm="HS256")
        with pytest.raises(InvalidTokenError, match="Пользователь не найден"):
            await service.get_account_from_token(token)

    async def test_get_account_from_token_blocked(self, monkeypatch):
        monkeypatch.setenv("JWT_SECRET", "test-secret")
        repo = AsyncMock()
        repo.get_by_id.return_value = Account(id=5, login="u", password="h", is_blocked=True)
        service = AuthService(repo)

        token = jwt.encode({"sub": "5", "login": "u"}, "test-secret", algorithm="HS256")
        with pytest.raises(BlockedAccountError):
            await service.get_account_from_token(token)
