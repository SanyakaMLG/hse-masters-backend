import dataclasses

from clients.kafka import KafkaClient
from repositories.items import ItemRepository
from repositories.moderation_results import ModerationResultRepository
from schemas.moderation import AsyncPredictResponse, TaskResultResponse


class PredictionWorkflowService:
    def __init__(
        self,
        item_repository: ItemRepository,
        moderation_result_repository: ModerationResultRepository,
    ) -> None:
        self._item_repo = item_repository
        self._mod_repo = moderation_result_repository

    async def async_predict(
        self, item_id: int, kafka_client: KafkaClient
    ) -> AsyncPredictResponse:
        await self._item_repo.get_item_with_user(item_id)
        task_id = await self._mod_repo.create_task(item_id)

        try:
            await kafka_client.send_moderation_request(item_id)
        except Exception:
            await self._mod_repo.update_task(
                task_id,
                status="failed",
                error_message="Ошибка при отправке в очередь обработки",
            )
            raise

        return AsyncPredictResponse(
            task_id=task_id,
            status="pending",
            message="Moderation request accepted",
        )

    async def get_moderation_result(self, task_id: int) -> TaskResultResponse:
        result = await self._mod_repo.get_task(task_id)
        response_data = TaskResultResponse(**dataclasses.asdict(result))
        if result.status == "completed":
            await self._item_repo.set_prediction(
                result.item_id, response_data.model_dump()
            )

        return response_data

    async def close_item(self, item_id: int) -> dict[str, str]:
        await self._item_repo.close_item(item_id)
        await self._mod_repo.delete_by_item_id(item_id)

        return {
            "status": "success",
            "message": f"Item {item_id} is closed and cache cleared",
        }
