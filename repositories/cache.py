import json
from dataclasses import dataclass
from typing import Optional

from redis.asyncio import Redis


@dataclass(frozen=True)
class PredictionRedisStorage:
    redis: Redis

    # Результаты модерации для конкретного объявления статичны,
    # пока объявление не отредактировано.
    # Сутки - оптимальное время: оно позволяет снизить нагрузку на БД и модель при
    # повторных просмотрах, но не забивает память Redis старыми объявлениями.
    TTL_SECONDS = 60 * 60 * 24

    @staticmethod
    def _key(item_id: int) -> str:
        return f"moderation:result:{item_id}"

    async def get(self, item_id: int) -> Optional[dict]:
        data = await self.redis.get(self._key(item_id))
        return json.loads(data) if data else None

    async def set(self, item_id: int, result: dict):
        await self.redis.set(
            self._key(item_id), json.dumps(result), ex=self.TTL_SECONDS
        )

    async def delete(self, item_id: int):
        await self.redis.delete(self._key(item_id))


@dataclass(frozen=True)
class PredictionCacheRepository:
    storage: PredictionRedisStorage

    def __init__(self, redis: Redis):
        object.__setattr__(self, "storage", PredictionRedisStorage(redis))

    async def get_prediction(self, item_id: int) -> Optional[dict]:
        return await self.storage.get(item_id)

    async def set_prediction(self, item_id: int, result: dict):
        await self.storage.set(item_id, result)

    async def delete_prediction(self, item_id: int):
        await self.storage.delete(item_id)
