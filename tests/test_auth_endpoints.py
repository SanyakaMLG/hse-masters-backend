import pytest

from repositories.accounts import AccountRepository


@pytest.mark.integration
@pytest.mark.anyio
class TestAuthEndpoints:
    async def test_login_success_sets_cookie(
        self, integration_app_async_client, db_pool
    ):
        repo = AccountRepository(db_pool)
        await repo.create_account("vasya", "qwerty")

        response = await integration_app_async_client.post(
            "/login", json={"login": "vasya", "password": "qwerty"}
        )

        assert response.status_code == 200
        assert "access_token" in response.cookies
        assert response.json()["token_type"] == "bearer"

    async def test_login_invalid_credentials(self, integration_app_async_client):
        response = await integration_app_async_client.post(
            "/login", json={"login": "missing", "password": "missing"}
        )

        assert response.status_code == 401
