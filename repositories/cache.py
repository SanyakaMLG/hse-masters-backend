import json
from typing import Optional

from redis.asyncio import Redis


class PredictionCacheRepository:
    # Результаты модерации для конкретного объявления статичны,
    # пока объявление не отредактировано.
    # Сутки - оптимальное время: оно позволяет снизить нагрузку на БД и модель при
    # повторных просмотрах, но не забивает память Redis старыми объявлениями.
    TTL_SECONDS = 60 * 60 * 24

    def __init__(self, redis: Redis):
        self.redis = redis

    def _key(self, item_id: int) -> str:
        return f"moderation:result:{item_id}"

    async def get_prediction(self, item_id: int) -> Optional[dict]:
        data = await self.redis.get(self._key(item_id))
        return json.loads(data) if data else None

    async def set_prediction(self, item_id: int, result: dict):
        await self.redis.set(
            self._key(item_id), json.dumps(result), ex=self.TTL_SECONDS
        )

    async def delete_prediction(self, item_id: int):
        await self.redis.delete(self._key(item_id))
