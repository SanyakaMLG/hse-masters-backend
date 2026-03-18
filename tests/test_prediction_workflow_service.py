from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from errors import ItemNotFoundError, ModerationTaskNotFoundError
from models.moderation import ModerationResult
from services.prediction_service import PredictionWorkflowService


@pytest.mark.anyio
class TestPredictionWorkflowService:
    @patch("services.prediction_service.ModerationResultRepository")
    @patch("services.prediction_service.ItemRepository")
    async def test_async_predict_success(self, mock_item_repo, mock_mod_repo):
        item_repo = mock_item_repo.return_value
        mod_repo = mock_mod_repo.return_value

        item_repo.get_item_with_user = AsyncMock(return_value=object())
        mod_repo.create_task = AsyncMock(return_value=42)

        kafka = AsyncMock()
        service = PredictionWorkflowService(
            item_repository=item_repo,
            moderation_result_repository=mod_repo,
        )

        result = await service.async_predict(1, kafka)

        assert result.task_id == 42
        kafka.send_moderation_request.assert_awaited_once_with(1)

    @patch("services.prediction_service.ModerationResultRepository")
    @patch("services.prediction_service.ItemRepository")
    async def test_async_predict_item_not_found(self, mock_item_repo, mock_mod_repo):
        item_repo = mock_item_repo.return_value
        item_repo.get_item_with_user = AsyncMock(
            side_effect=ItemNotFoundError("Объявление не найдено")
        )

        service = PredictionWorkflowService(
            item_repository=item_repo,
            moderation_result_repository=mock_mod_repo.return_value,
        )

        with pytest.raises(ItemNotFoundError):
            await service.async_predict(1, AsyncMock())

    @patch("services.prediction_service.ModerationResultRepository")
    @patch("services.prediction_service.ItemRepository")
    async def test_get_moderation_result_not_found(self, mock_item_repo, mock_mod_repo):
        mod_repo = mock_mod_repo.return_value
        mod_repo.get_task = AsyncMock(
            side_effect=ModerationTaskNotFoundError("Задача не найдена")
        )

        service = PredictionWorkflowService(
            item_repository=mock_item_repo.return_value,
            moderation_result_repository=mod_repo,
        )

        with pytest.raises(ModerationTaskNotFoundError):
            await service.get_moderation_result(999)

    @patch("services.prediction_service.ModerationResultRepository")
    @patch("services.prediction_service.ItemRepository")
    async def test_close_item_orchestrates_repositories(
        self, mock_item_repo, mock_mod_repo
    ):
        item_repo = mock_item_repo.return_value
        mod_repo = mock_mod_repo.return_value

        item_repo.close_item = AsyncMock(return_value=None)
        mod_repo.delete_by_item_id = AsyncMock(return_value=True)

        service = PredictionWorkflowService(
            item_repository=item_repo,
            moderation_result_repository=mod_repo,
        )
        result = await service.close_item(100)

        assert result["status"] == "success"
        mod_repo.delete_by_item_id.assert_awaited_once_with(100)

    @patch("services.prediction_service.ModerationResultRepository")
    @patch("services.prediction_service.ItemRepository")
    async def test_async_predict_kafka_error_updates_task(
        self, mock_item_repo, mock_mod_repo
    ):
        item_repo = mock_item_repo.return_value
        mod_repo = mock_mod_repo.return_value

        item_repo.get_item_with_user = AsyncMock(return_value=object())
        mod_repo.create_task = AsyncMock(return_value=77)
        mod_repo.update_task = AsyncMock(return_value=None)

        kafka = AsyncMock()
        kafka.send_moderation_request.side_effect = Exception("kafka error")

        service = PredictionWorkflowService(
            item_repository=item_repo,
            moderation_result_repository=mod_repo,
        )

        with pytest.raises(Exception, match="kafka error"):
            await service.async_predict(10, kafka)

        mod_repo.update_task.assert_awaited_once_with(
            77,
            status="failed",
            error_message="Ошибка при отправке в очередь обработки",
        )

    @patch("services.prediction_service.ModerationResultRepository")
    @patch("services.prediction_service.ItemRepository")
    async def test_close_item_not_found(self, mock_item_repo, mock_mod_repo):
        item_repo = mock_item_repo.return_value
        mod_repo = mock_mod_repo.return_value

        item_repo.close_item = AsyncMock(
            side_effect=ItemNotFoundError("Объявление не найдено")
        )
        mod_repo.delete_by_item_id = AsyncMock(return_value=True)

        service = PredictionWorkflowService(
            item_repository=item_repo,
            moderation_result_repository=mod_repo,
        )

        with pytest.raises(ItemNotFoundError, match="Объявление не найдено"):
            await service.close_item(100)

        mod_repo.delete_by_item_id.assert_not_awaited()

    @patch("services.prediction_service.ModerationResultRepository")
    @patch("services.prediction_service.ItemRepository")
    async def test_get_moderation_result_completed_caches(
        self, mock_item_repo, mock_mod_repo
    ):
        item_repo = mock_item_repo.return_value
        mod_repo = mock_mod_repo.return_value
        mod_repo.get_task = AsyncMock(
            return_value=ModerationResult(
                task_id=1,
                item_id=2,
                status="completed",
                is_violation=True,
                probability=0.7,
                error_message=None,
                created_at=datetime.utcnow(),
                processed_at=datetime.utcnow(),
            )
        )
        item_repo.set_prediction = AsyncMock(return_value=None)

        service = PredictionWorkflowService(
            item_repository=item_repo,
            moderation_result_repository=mod_repo,
        )
        result = await service.get_moderation_result(1)

        assert result.task_id == 1
        item_repo.set_prediction.assert_awaited_once()
