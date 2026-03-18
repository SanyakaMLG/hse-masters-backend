import os
from datetime import UTC, datetime, timedelta

import jwt

from errors import (
    AccountNotFoundError,
    BlockedAccountError,
    InvalidCredentialsError,
    InvalidTokenError,
)
from models.moderation import Account
from repositories.accounts import AccountRepository


class AuthService:
    def __init__(self, account_repository: AccountRepository) -> None:
        self._account_repository = account_repository
        self._jwt_secret = os.getenv("JWT_SECRET", "very-secret-key")
        self._jwt_algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        self._token_ttl_minutes = int(os.getenv("JWT_TTL_MINUTES", "60"))

    async def login(self, login: str, password: str) -> str:
        try:
            account = await self._account_repository.get_by_login_password(
                login, password
            )
        except AccountNotFoundError as exc:
            raise InvalidCredentialsError("Неверный логин или пароль") from exc
        if account.is_blocked:
            raise BlockedAccountError("Аккаунт заблокирован")

        expires_at = datetime.now(UTC) + timedelta(minutes=self._token_ttl_minutes)
        payload = {
            "sub": str(account.id),
            "login": account.login,
            "exp": expires_at,
        }
        return jwt.encode(payload, self._jwt_secret, algorithm=self._jwt_algorithm)

    async def get_account_from_token(self, token: str) -> Account:
        try:
            payload = jwt.decode(
                token, self._jwt_secret, algorithms=[self._jwt_algorithm]
            )
        except jwt.InvalidTokenError as exc:
            raise InvalidTokenError("Некорректный токен") from exc

        account_id = payload.get("sub")
        if account_id is None:
            raise InvalidTokenError("Некорректный токен")

        try:
            account = await self._account_repository.get_by_id(int(account_id))
        except AccountNotFoundError as exc:
            raise InvalidTokenError("Пользователь не найден") from exc
        if account.is_blocked:
            raise BlockedAccountError("Аккаунт заблокирован")

        return account
