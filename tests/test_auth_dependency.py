from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from dependencies import get_current_account
from models.moderation import Account
from services.auth_service import InvalidTokenError


@pytest.mark.anyio
async def test_get_current_account_no_cookie(db_pool):
    with pytest.raises(HTTPException) as exc_info:
        await get_current_account(pool=db_pool, access_token=None)

    assert exc_info.value.status_code == 401


@pytest.mark.anyio
async def test_get_current_account_success(db_pool):
    expected = Account(id=1, login="u", password="h", is_blocked=False)

    with patch(
        "dependencies.AuthService.get_account_from_token", new_callable=AsyncMock
    ) as mocked:
        mocked.return_value = expected
        account = await get_current_account(pool=db_pool, access_token="token")

    assert account.id == expected.id


@pytest.mark.anyio
async def test_get_current_account_invalid_token(db_pool):
    with patch(
        "dependencies.AuthService.get_account_from_token", new_callable=AsyncMock
    ) as mocked:
        mocked.side_effect = InvalidTokenError("Некорректный токен")
        with pytest.raises(HTTPException) as exc_info:
            await get_current_account(pool=db_pool, access_token="bad")

    assert exc_info.value.status_code == 401
