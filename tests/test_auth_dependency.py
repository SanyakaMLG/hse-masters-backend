from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from dependencies import get_current_account
from models.moderation import Account
from services.auth_service import InvalidTokenError


@pytest.mark.anyio
async def test_get_current_account_no_cookie():
    with pytest.raises(HTTPException) as exc_info:
        await get_current_account(auth_service=AsyncMock(), access_token=None)

    assert exc_info.value.status_code == 401


@pytest.mark.anyio
async def test_get_current_account_success():
    expected = Account(id=1, login="u", password="h", is_blocked=False)
    auth_service = AsyncMock()
    auth_service.get_account_from_token.return_value = expected

    account = await get_current_account(auth_service=auth_service, access_token="token")

    assert account.id == expected.id


@pytest.mark.anyio
async def test_get_current_account_invalid_token():
    auth_service = AsyncMock()
    auth_service.get_account_from_token.side_effect = InvalidTokenError(
        "Некорректный токен"
    )
    with pytest.raises(HTTPException) as exc_info:
        await get_current_account(auth_service=auth_service, access_token="bad")

    assert exc_info.value.status_code == 401
