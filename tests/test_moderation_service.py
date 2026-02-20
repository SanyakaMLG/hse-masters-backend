from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from errors import ItemNotFoundError, ModelNotLoadedError
from schemas.moderation import PredictionRequest
from services.moderation_service import ModerationService


@pytest.mark.anyio
class TestModerationService:
    @pytest.mark.parametrize(
        "probability, expected_violation",
        [
            (0.9, True),
            (0.4, False),
            (0.5, False),
        ],
    )
    @patch("clients.ml_client.MLClient.predict_proba")
    def test_predict_success(self, mock_predict, probability, expected_violation):
        mock_predict.return_value = probability
        request = PredictionRequest(
            seller_id=1,
            is_verified_seller=False,
            item_id=100,
            name="Test",
            description="Desc",
            category=1,
            images_qty=0,
        )

        result = ModerationService.predict(request)

        assert result.probability == probability
        assert result.is_violation == expected_violation
        mock_predict.assert_called_once()

    @patch("clients.ml_client.MLClient.predict_proba")
    def test_predict_model_not_loaded(self, mock_predict):
        mock_predict.side_effect = ModelNotLoadedError("Модель не загружена")
        request = PredictionRequest(
            seller_id=1,
            is_verified_seller=False,
            item_id=100,
            name="Test",
            description="Desc",
            category=1,
            images_qty=0,
        )

        with pytest.raises(ModelNotLoadedError, match="Модель не загружена"):
            ModerationService.predict(request)

    @patch(
        "repositories.items.ItemRepository.get_item_with_user", new_callable=AsyncMock
    )
    async def test_simple_predict_item_not_found(self, mock_get_item):
        mock_get_item.return_value = None
        with pytest.raises(ItemNotFoundError):
            await ModerationService.simple_predict(1, MagicMock())
