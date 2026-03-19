from unittest.mock import AsyncMock

import pytest

from models.moderation import PredictionOutput
from services.worker_service import WorkerService


@pytest.mark.anyio
class TestWorkerService:
    async def test_moderate_pending_item_returns_none_without_pending_task(self):
        item_repo = AsyncMock()
        mod_repo = AsyncMock()
        moderation_service = AsyncMock()
        mod_repo.get_latest_pending_task_id.return_value = None
        service = WorkerService(
            item_repository=item_repo,
            moderation_result_repository=mod_repo,
            moderation_service=moderation_service,
        )

        result = await service.moderate_pending_item(10)

        assert result is None
        moderation_service.simple_predict.assert_not_awaited()
        mod_repo.update_task.assert_not_awaited()

    async def test_moderate_pending_item_marks_task_completed(self):
        item_repo = AsyncMock()
        mod_repo = AsyncMock()
        moderation_service = AsyncMock()
        mod_repo.get_latest_pending_task_id.return_value = 15
        moderation_service.simple_predict.return_value = PredictionOutput(
            is_violation=True,
            probability=0.83,
        )
        service = WorkerService(
            item_repository=item_repo,
            moderation_result_repository=mod_repo,
            moderation_service=moderation_service,
        )

        result = await service.moderate_pending_item(10)

        assert result == 15
        moderation_service.simple_predict.assert_awaited_once_with(10)
        mod_repo.update_task.assert_awaited_once_with(
            15,
            status="completed",
            is_violation=True,
            probability=0.83,
        )

    async def test_moderate_pending_item_marks_task_failed_on_prediction_error(self):
        item_repo = AsyncMock()
        mod_repo = AsyncMock()
        moderation_service = AsyncMock()
        mod_repo.get_latest_pending_task_id.return_value = 15
        moderation_service.simple_predict.side_effect = RuntimeError("boom")
        service = WorkerService(
            item_repository=item_repo,
            moderation_result_repository=mod_repo,
            moderation_service=moderation_service,
        )

        with pytest.raises(RuntimeError, match="boom"):
            await service.moderate_pending_item(10)

        mod_repo.update_task.assert_awaited_once_with(
            15,
            status="failed",
            error_message="Item not found",
        )

    async def test_mark_latest_task_failed_updates_pending_task(self):
        item_repo = AsyncMock()
        mod_repo = AsyncMock()
        moderation_service = AsyncMock()
        mod_repo.get_latest_pending_task_id.return_value = 42
        service = WorkerService(
            item_repository=item_repo,
            moderation_result_repository=mod_repo,
            moderation_service=moderation_service,
        )

        await service.mark_latest_task_failed(7, "Max retries exceeded")

        mod_repo.update_task.assert_awaited_once_with(
            42,
            status="failed",
            error_message="Max retries exceeded",
        )

    async def test_mark_latest_task_failed_skips_when_pending_task_absent(self):
        item_repo = AsyncMock()
        mod_repo = AsyncMock()
        moderation_service = AsyncMock()
        mod_repo.get_latest_pending_task_id.return_value = None
        service = WorkerService(
            item_repository=item_repo,
            moderation_result_repository=mod_repo,
            moderation_service=moderation_service,
        )

        await service.mark_latest_task_failed(7, "Max retries exceeded")

        mod_repo.update_task.assert_not_awaited()
