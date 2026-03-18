from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response

from dependencies import get_auth_service
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
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
):
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
