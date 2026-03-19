from repositories.items import ItemRepository
from repositories.moderation_results import ModerationResultRepository
from services.moderation_service import ModerationService


class WorkerService:
    def __init__(
        self,
        item_repository: ItemRepository,
        moderation_result_repository: ModerationResultRepository,
        moderation_service: ModerationService,
    ) -> None:
        self._item_repo = item_repository
        self._mod_repo = moderation_result_repository
        self._moderation_service = moderation_service

    async def moderate_pending_item(self, item_id: int) -> int | None:
        task_id = await self._mod_repo.get_latest_pending_task_id(item_id)
        if not task_id:
            return None

        try:
            prediction = await self._moderation_service.simple_predict(item_id)
        except Exception:
            await self._mod_repo.update_task(
                task_id,
                status="failed",
                error_message="Item not found",
            )
            raise

        await self._mod_repo.update_task(
            task_id,
            status="completed",
            is_violation=prediction.is_violation,
            probability=prediction.probability,
        )
        return task_id

    async def mark_latest_task_failed(self, item_id: int, error_message: str) -> None:
        task_id = await self._mod_repo.get_latest_pending_task_id(item_id)
        if task_id:
            await self._mod_repo.update_task(
                task_id,
                status="failed",
                error_message=error_message,
            )
