from typing import Annotated

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Response

from clients.db import get_db_pool_dependency
from repositories.accounts import AccountRepository
from schemas.auth import LoginRequest, LoginResponse
from services.auth_service import (
    AuthService,
    BlockedAccountError,
    InvalidCredentialsError,
)

router = APIRouter()


@router.post("/login", response_model=LoginResponse)
async def login(
    credentials: LoginRequest,
    response: Response,
    pool: Annotated[asyncpg.Pool, Depends(get_db_pool_dependency)],
):
    auth_service = AuthService(AccountRepository(pool))

    try:
        token = await auth_service.login(credentials.login, credentials.password)
    except (InvalidCredentialsError, BlockedAccountError) as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
    )

    return LoginResponse(access_token=token)
