from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from errors import ItemNotFoundError, ModelNotLoadedError
from repositories.items import ItemWithUser
from repositories.moderation_results import ModerationResult
from schemas.moderation import PredictionResponse


@pytest.mark.anyio
class TestModerationEndpoints:
    # ENDPOINT: /predict/
    @pytest.mark.parametrize("is_violation, probability", [(True, 0.85), (False, 0.12)])
    @patch("routers.moderation.ModerationService.predict")
    async def test_predict_success(
        self, mock_predict, app_async_client, is_violation, probability
    ):
        mock_predict.return_value = PredictionResponse(
            is_violation=is_violation, probability=probability
        )

        payload = {
            "seller_id": 1,
            "is_verified_seller": False,
            "item_id": 100,
            "name": "Test Item",
            "description": "Desc",
            "category": 1,
            "images_qty": 0,
        }
        resp = await app_async_client.post("/predict/", json=payload)

        assert resp.status_code == 200
        assert resp.json()["is_violation"] is is_violation
        mock_predict.assert_called_once()

    @pytest.mark.parametrize(
        "invalid_payload, error_description",
        [
            (
                {
                    "seller_id": "not_an_int",
                    "is_verified_seller": True,
                    "item_id": 100,
                    "name": "Test",
                    "description": "Desc",
                    "category": 1,
                    "images_qty": 0,
                },
                "Неверный тип seller_id",
            ),
            (
                {
                    "seller_id": 1,
                    "is_verified_seller": True,
                    "item_id": "not_an_int",
                    "name": "Test",
                    "description": "Desc",
                    "category": 1,
                    "images_qty": 0,
                },
                "Неверный тип item_id",
            ),
            (
                {
                    "seller_id": 1,
                    "is_verified_seller": "not_a_bool",
                    "item_id": 100,
                    "name": "Test",
                    "description": "Desc",
                    "category": 1,
                    "images_qty": 0,
                },
                "Неверный тип is_verified_seller",
            ),
            (
                {
                    "seller_id": 1,
                    "is_verified_seller": True,
                    "item_id": 100,
                    "name": "Test",
                    "description": "Desc",
                    "category": "not_an_int",
                    "images_qty": 0,
                },
                "Неверный тип category",
            ),
            (
                {
                    "seller_id": 1,
                    "is_verified_seller": True,
                    "item_id": 100,
                    "name": "Test",
                    "description": "Desc",
                    "category": 1,
                    "images_qty": "not_an_int",
                },
                "Неверный тип images_qty",
            ),
            (
                {
                    "seller_id": 1,
                    "is_verified_seller": True,
                    "name": "Test",
                    "description": "Desc",
                    "category": 1,
                    "images_qty": 0,
                },
                "Пропущено обязательное поле item_id",
            ),
            (
                {
                    "seller_id": 1,
                    "is_verified_seller": True,
                    "item_id": 100,
                    "name": "Test",
                    "description": "Desc",
                    "category": 1,
                    "images_qty": -1,
                },
                "Отрицательное количество изображений (меньше 0)",
            ),
            (
                {
                    "seller_id": 1,
                    "is_verified_seller": True,
                    "item_id": 100,
                    "name": "",
                    "description": "Desc",
                    "category": 1,
                    "images_qty": 0,
                },
                "Пустое название (min_length=1)",
            ),
        ],
    )
    async def test_predict_validation_errors(
        self, app_async_client, invalid_payload, error_description
    ):
        resp = await app_async_client.post("/predict/", json=invalid_payload)
        assert resp.status_code == 422, (
            f"Тест '{error_description}' провалился: ожидался 422 статус"
        )

    @patch("routers.moderation.ModerationService.predict")
    async def test_predict_503_model_not_loaded(self, mock_predict, app_async_client):
        mock_predict.side_effect = ModelNotLoadedError("Model logic fail")
        payload = {
            "seller_id": 1,
            "is_verified_seller": False,
            "item_id": 100,
            "name": "Test",
            "description": "Desc",
            "category": 1,
            "images_qty": 0,
        }
        resp = await app_async_client.post("/predict/", json=payload)
        assert resp.status_code == 503
        assert "Model logic fail" in resp.json()["detail"]

    # ENDPOINT: /predict/simple_predict
    @patch(
        "routers.moderation.ModerationService.simple_predict", new_callable=AsyncMock
    )
    async def test_simple_predict_success(self, mock_simple_predict, app_async_client):
        mock_simple_predict.return_value = PredictionResponse(
            is_violation=False, probability=0.1
        )

        resp = await app_async_client.get(
            "/predict/simple_predict", params={"item_id": 100}
        )
        assert resp.status_code == 200
        assert resp.json()["is_violation"] is False

    @patch("routers.moderation.ModerationService.simple_predict")
    async def test_simple_predict_404(self, mock_simple, app_async_client):
        mock_simple.side_effect = ItemNotFoundError("Not found")
        resp = await app_async_client.get(
            "/predict/simple_predict", params={"item_id": 999}
        )
        assert resp.status_code == 404

    # ENDPOINT: /predict/async_predict
    @patch(
        "routers.moderation.KafkaClient.send_moderation_request", new_callable=AsyncMock
    )
    @patch(
        "routers.moderation.ModerationResultRepository.create_task",
        new_callable=AsyncMock,
    )
    @patch(
        "routers.moderation.ItemRepository.get_item_with_user", new_callable=AsyncMock
    )
    async def test_async_predict_success(
        self, mock_get_item, mock_create_task, mock_kafka_send, app_async_client
    ):
        mock_get_item.return_value = ItemWithUser(
            item_id=1,
            seller_id=2,
            is_verified_seller=False,
            name="Test Async",
            description="Desc",
            category=1,
            images_qty=0,
        )

        mock_create_task.return_value = 42

        resp = await app_async_client.post(
            "/predict/async_predict", params={"item_id": 1}
        )

        assert resp.status_code == 200
        assert resp.json()["status"] == "pending"
        assert resp.json()["task_id"] == 42
        mock_kafka_send.assert_called_once_with(1)

    @patch(
        "routers.moderation.PredictionCacheRepository.set_prediction",
        new_callable=AsyncMock,
    )
    @patch(
        "routers.moderation.ModerationResultRepository.get_task", new_callable=AsyncMock
    )
    async def test_get_moderation_result(
        self, mock_get_task, mock_set_cache, app_async_client
    ):
        mock_get_task.return_value = ModerationResult(
            task_id=42,
            item_id=1,
            status="completed",
            is_violation=True,
            probability=0.99,
            error_message=None,
            created_at=datetime.utcnow(),
            processed_at=datetime.utcnow(),
        )

        resp = await app_async_client.get("/predict/moderation_result/42")

        assert resp.status_code == 200
        assert resp.json()["is_violation"] is True

        mock_set_cache.assert_called_once()

    @patch(
        "routers.moderation.ModerationResultRepository.get_task", new_callable=AsyncMock
    )
    async def test_get_moderation_result_not_found(
        self, mock_get_task, app_async_client
    ):
        mock_get_task.return_value = None
        resp = await app_async_client.get("/predict/moderation_result/999")
        assert resp.status_code == 404

    @patch(
        "routers.moderation.KafkaClient.send_moderation_request", new_callable=AsyncMock
    )
    @patch(
        "routers.moderation.ModerationResultRepository.create_task",
        new_callable=AsyncMock,
    )
    @patch(
        "routers.moderation.ItemRepository.get_item_with_user", new_callable=AsyncMock
    )
    async def test_async_predict_kafka_error(
        self, mock_get, mock_create, mock_kafka, app_async_client
    ):
        mock_get.return_value = AsyncMock()
        mock_create.return_value = 1
        mock_kafka.side_effect = Exception("Kafka connection error")

        resp = await app_async_client.post(
            "/predict/async_predict", params={"item_id": 1}
        )
        assert resp.status_code == 500
        assert "Ошибка при отправке в очередь" in resp.json()["detail"]

    # ENDPOINT: /predict/close
    @patch(
        "routers.moderation.PredictionCacheRepository.delete_prediction",
        new_callable=AsyncMock,
    )
    @patch("routers.moderation.ItemRepository.close_item", new_callable=AsyncMock)
    async def test_close_item_success(
        self, mock_close_item, mock_delete_cache, app_async_client
    ):
        mock_close_item.return_value = True

        resp = await app_async_client.post("/predict/close", params={"item_id": 100})

        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

        mock_delete_cache.assert_called_once_with(100)

    @patch("routers.moderation.ItemRepository.close_item", new_callable=AsyncMock)
    async def test_close_item_not_found(self, mock_close_item, app_async_client):
        mock_close_item.return_value = False

        resp = await app_async_client.post("/predict/close", params={"item_id": 999999})
        assert resp.status_code == 404
