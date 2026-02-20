import pytest

from repositories.cache import PredictionCacheRepository


@pytest.mark.integration
@pytest.mark.anyio
class TestCacheRepository:
    async def test_set_and_get_prediction(self, redis_client):
        repo = PredictionCacheRepository(redis_client)
        item_id = 1
        data = {"is_violation": False, "probability": 0.15}

        await repo.set_prediction(item_id, data)
        cached = await repo.get_prediction(item_id)

        assert cached == data

    async def test_get_prediction_not_found(self, redis_client):
        repo = PredictionCacheRepository(redis_client)
        assert await repo.get_prediction(999) is None

    async def test_delete_prediction(self, redis_client):
        repo = PredictionCacheRepository(redis_client)
        item_id = 2
        data = {"is_violation": True, "probability": 0.95}

        await repo.set_prediction(item_id, data)
        await repo.delete_prediction(item_id)
        cached = await repo.get_prediction(item_id)

        assert cached is None
