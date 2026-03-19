import dataclasses

from clients.kafka import KafkaClient
from models.moderation import AsyncPredictionTask, ModerationTaskResult
from repositories.items import ItemRepository
from repositories.moderation_results import ModerationResultRepository


class PredictionWorkflowService:
    def __init__(
        self,
        item_repository: ItemRepository,
        moderation_result_repository: ModerationResultRepository,
        kafka_client: KafkaClient,
    ) -> None:
        self._item_repo = item_repository
        self._mod_repo = moderation_result_repository
        self._kafka_client = kafka_client

    async def async_predict(self, item_id: int) -> AsyncPredictionTask:
        await self._item_repo.get_item_with_user(item_id)
        task_id = await self._mod_repo.create_task(item_id)

        try:
            await self._kafka_client.send_moderation_request(item_id)
        except Exception:
            await self._mod_repo.update_task(
                task_id,
                status="failed",
                error_message="Ошибка при отправке в очередь обработки",
            )
            raise

        return AsyncPredictionTask(
            task_id=task_id,
            status="pending",
            message="Moderation request accepted",
        )

    async def get_moderation_result(self, task_id: int) -> ModerationTaskResult:
        result = await self._mod_repo.get_task(task_id)
        response_data = ModerationTaskResult(
            task_id=result.task_id,
            status=result.status,
            is_violation=result.is_violation,
            probability=result.probability,
        )
        if result.status == "completed":
            await self._item_repo.set_prediction(
                result.item_id,
                dataclasses.asdict(response_data),
            )

        return response_data

    async def close_item(self, item_id: int) -> dict[str, str]:
        await self._item_repo.close_item(item_id)
        await self._mod_repo.delete_by_item_id(item_id)

        return {
            "status": "success",
            "message": f"Item {item_id} is closed and cache cleared",
        }
