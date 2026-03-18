from unittest.mock import AsyncMock, patch

import pytest

from errors import ItemNotFoundError, ModelNotLoadedError
from schemas.moderation import (
    AsyncPredictResponse,
    PredictionResponse,
    TaskResultResponse,
)


@pytest.mark.anyio
class TestModerationEndpoints:
    # ENDPOINT: /predict/
    @pytest.mark.parametrize("is_violation, probability", [(True, 0.85), (False, 0.12)])
    @patch(
        "routers.moderation.ModerationService.predict_with_cache",
        new_callable=AsyncMock,
    )
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
        mock_predict.assert_awaited_once()

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

    @patch(
        "routers.moderation.ModerationService.predict_with_cache",
        new_callable=AsyncMock,
    )
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

    @patch(
        "routers.moderation.ModerationService.predict_with_cache",
        new_callable=AsyncMock,
    )
    async def test_predict_500_unexpected_error(self, mock_predict, app_async_client):
        mock_predict.side_effect = Exception("unexpected")
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
        assert resp.status_code == 500
        assert "unexpected" in resp.json()["detail"]

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

    @patch("routers.moderation.ModerationService.simple_predict")
    async def test_simple_predict_503_model_not_loaded(
        self, mock_simple, app_async_client
    ):
        mock_simple.side_effect = ModelNotLoadedError("model unloaded")
        resp = await app_async_client.get(
            "/predict/simple_predict", params={"item_id": 999}
        )
        assert resp.status_code == 503
        assert "model unloaded" in resp.json()["detail"]

    @patch("routers.moderation.ModerationService.simple_predict")
    async def test_simple_predict_500_unexpected(self, mock_simple, app_async_client):
        mock_simple.side_effect = Exception("unexpected")
        resp = await app_async_client.get(
            "/predict/simple_predict", params={"item_id": 999}
        )
        assert resp.status_code == 500
        assert "unexpected" in resp.json()["detail"]

    # ENDPOINT: /predict/async_predict
    @patch(
        "routers.moderation.PredictionWorkflowService.async_predict",
        new_callable=AsyncMock,
    )
    async def test_async_predict_success(self, mock_async_predict, app_async_client):
        mock_async_predict.return_value = AsyncPredictResponse(
            task_id=42, status="pending", message="Moderation request accepted"
        )

        resp = await app_async_client.post(
            "/predict/async_predict", params={"item_id": 1}
        )

        assert resp.status_code == 200
        assert resp.json()["status"] == "pending"
        assert resp.json()["task_id"] == 42

    @patch(
        "routers.moderation.PredictionWorkflowService.get_moderation_result",
        new_callable=AsyncMock,
    )
    async def test_get_moderation_result(
        self, mock_get_moderation_result, app_async_client
    ):
        mock_get_moderation_result.return_value = TaskResultResponse(
            task_id=42, status="completed", is_violation=True, probability=0.99
        )

        resp = await app_async_client.get("/predict/moderation_result/42")

        assert resp.status_code == 200
        assert resp.json()["is_violation"] is True

    @patch(
        "routers.moderation.PredictionWorkflowService.get_moderation_result",
        new_callable=AsyncMock,
    )
    async def test_get_moderation_result_not_found(
        self, mock_get_moderation_result, app_async_client
    ):
        from errors import ModerationTaskNotFoundError

        mock_get_moderation_result.side_effect = ModerationTaskNotFoundError(
            "Задача не найдена"
        )
        resp = await app_async_client.get("/predict/moderation_result/999")
        assert resp.status_code == 404

    @patch(
        "routers.moderation.PredictionWorkflowService.async_predict",
        new_callable=AsyncMock,
    )
    async def test_async_predict_kafka_error(
        self, mock_async_predict, app_async_client
    ):
        mock_async_predict.side_effect = Exception("Kafka connection error")

        resp = await app_async_client.post(
            "/predict/async_predict", params={"item_id": 1}
        )
        assert resp.status_code == 500
        assert "Ошибка при отправке в очередь" in resp.json()["detail"]

    @patch(
        "routers.moderation.PredictionWorkflowService.async_predict",
        new_callable=AsyncMock,
    )
    async def test_async_predict_item_not_found(
        self, mock_async_predict, app_async_client
    ):
        mock_async_predict.side_effect = ItemNotFoundError("Объявление не найдено")

        resp = await app_async_client.post(
            "/predict/async_predict", params={"item_id": 1}
        )
        assert resp.status_code == 404
        assert "Объявление не найдено" in resp.json()["detail"]

    # ENDPOINT: /predict/close
    @patch(
        "routers.moderation.PredictionWorkflowService.close_item",
        new_callable=AsyncMock,
    )
    async def test_close_item_success(self, mock_close_item, app_async_client):
        mock_close_item.return_value = {
            "status": "success",
            "message": "Item 100 is closed and cache cleared",
        }

        resp = await app_async_client.post("/predict/close", params={"item_id": 100})

        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    @patch(
        "routers.moderation.PredictionWorkflowService.close_item",
        new_callable=AsyncMock,
    )
    async def test_close_item_not_found(self, mock_close_item, app_async_client):
        mock_close_item.side_effect = ItemNotFoundError("Объявление не найдено")

        resp = await app_async_client.post("/predict/close", params={"item_id": 999999})
        assert resp.status_code == 404
