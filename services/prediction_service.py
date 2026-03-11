import dataclasses

import asyncpg
from redis.asyncio import Redis

from clients.kafka import KafkaClient
from errors import ItemNotFoundError, ModerationTaskNotFoundError
from repositories.cache import PredictionCacheRepository
from repositories.items import ItemRepository
from repositories.moderation_results import ModerationResultRepository
from schemas.moderation import AsyncPredictResponse, TaskResultResponse


class PredictionWorkflowService:
    def __init__(self, pool: asyncpg.Pool, redis: Redis) -> None:
        self._item_repo = ItemRepository(pool)
        self._mod_repo = ModerationResultRepository(pool)
        self._cache_repo = PredictionCacheRepository(redis)

    async def async_predict(
        self, item_id: int, kafka_client: KafkaClient
    ) -> AsyncPredictResponse:
        item = await self._item_repo.get_item_with_user(item_id)
        if not item:
            raise ItemNotFoundError("Объявление не найдено")

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
        if not result:
            raise ModerationTaskNotFoundError("Задача не найдена")

        response_data = TaskResultResponse(**dataclasses.asdict(result))

        if result.status == "completed":
            await self._cache_repo.set_prediction(
                result.item_id, response_data.model_dump()
            )

        return response_data

    async def close_item(self, item_id: int) -> dict[str, str]:
        closed = await self._item_repo.close_item(item_id)
        if not closed:
            raise ItemNotFoundError("Объявление не найдено")

        await self._mod_repo.delete_by_item_id(item_id)
        await self._cache_repo.delete_prediction(item_id)

        return {
            "status": "success",
            "message": f"Item {item_id} is closed and cache cleared",
        }
