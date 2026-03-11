from typing import Annotated

import asyncpg
from fastapi import Cookie, Depends, HTTPException

from clients.db import get_db_pool_dependency
from models.moderation import Account
from repositories.accounts import AccountRepository
from services.auth_service import (
    AuthService,
    BlockedAccountError,
    InvalidTokenError,
)


async def get_current_account(
    pool: Annotated[asyncpg.Pool, Depends(get_db_pool_dependency)],
    access_token: Annotated[str | None, Cookie()] = None,
) -> Account:
    if not access_token:
        raise HTTPException(status_code=401, detail="Пользователь не авторизован")

    auth_service = AuthService(AccountRepository(pool))
    try:
        return await auth_service.get_account_from_token(access_token)
    except (InvalidTokenError, BlockedAccountError) as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
