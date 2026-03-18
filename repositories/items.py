import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import asyncpg
from redis.asyncio import Redis

from errors import ItemNotFoundError
from models.moderation import Item, ItemWithUser
from utils.metrics import (
    observe_db_insert,
    observe_db_select,
    observe_db_update,
)


@dataclass(frozen=True)
class ItemPostgresStorage:
    pool: asyncpg.Pool

    async def create(
        self,
        user_id: int,
        name: str,
        description: str,
        category: int,
        images_qty: int,
    ) -> Mapping[str, Any]:
        async with observe_db_insert():
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO items (user_id, name, description, category, images_qty)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING id, user_id, name, description, category, images_qty
                    """,
                    user_id,
                    name,
                    description,
                    category,
                    images_qty,
                )
        return dict(row)

    async def select_item_with_user(self, item_id: int) -> Optional[Mapping[str, Any]]:
        async with observe_db_select():
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT
                        i.id AS item_id,
                        u.id AS seller_id,
                        u.is_verified_seller AS is_verified_seller,
                        i.name AS name,
                        i.description AS description,
                        i.category AS category,
                        i.images_qty AS images_qty
                    FROM items i
                    JOIN users u ON i.user_id = u.id
                    WHERE i.id = $1 AND i.is_closed = FALSE
                    """,
                    item_id,
                )
        return dict(row) if row is not None else None

    async def close(self, item_id: int) -> bool:
        async with observe_db_update():
            async with self.pool.acquire() as conn:
                status = await conn.execute(
                    "UPDATE items SET is_closed = TRUE WHERE id = $1", item_id
                )
        return status == "UPDATE 1"


@dataclass(frozen=True)
class ItemRedisStorage:
    redis: Redis

    TTL_SECONDS = 60 * 60 * 24

    @staticmethod
    def _item_key(item_id: int) -> str:
        return f"item:{item_id}"

    @staticmethod
    def _prediction_key(item_id: int) -> str:
        return f"prediction:{item_id}"

    async def get_item_with_user(self, item_id: int) -> Optional[Mapping[str, Any]]:
        data = await self.redis.get(self._item_key(item_id))
        return json.loads(data) if data else None

    async def set_item_with_user(self, item: ItemWithUser) -> None:
        await self.redis.set(
            self._item_key(item.item_id),
            json.dumps(
                {
                    "item_id": item.item_id,
                    "seller_id": item.seller_id,
                    "is_verified_seller": item.is_verified_seller,
                    "name": item.name,
                    "description": item.description,
                    "category": item.category,
                    "images_qty": item.images_qty,
                }
            ),
            ex=self.TTL_SECONDS,
        )

    async def delete_item_with_user(self, item_id: int) -> None:
        await self.redis.delete(self._item_key(item_id))

    async def get_prediction(self, item_id: int) -> Optional[dict]:
        data = await self.redis.get(self._prediction_key(item_id))
        return json.loads(data) if data else None

    async def set_prediction(self, item_id: int, result: dict) -> None:
        await self.redis.set(
            self._prediction_key(item_id), json.dumps(result), ex=self.TTL_SECONDS
        )

    async def delete_prediction(self, item_id: int) -> None:
        await self.redis.delete(self._prediction_key(item_id))


@dataclass(frozen=True)
class ItemRepository:
    storage: ItemPostgresStorage
    cache_storage: Optional[ItemRedisStorage]

    def __init__(self, pool: asyncpg.Pool, redis: Optional[Redis] = None) -> None:
        object.__setattr__(self, "storage", ItemPostgresStorage(pool))
        object.__setattr__(
            self,
            "cache_storage",
            ItemRedisStorage(redis) if redis is not None else None,
        )

    async def create_item(
        self,
        user_id: int,
        name: str,
        description: str,
        category: int,
        images_qty: int,
    ) -> Item:
        row = await self.storage.create(
            user_id, name, description, category, images_qty
        )
        return Item(
            id=row["id"],
            user_id=row["user_id"],
            name=row["name"],
            description=row["description"],
            category=row["category"],
            images_qty=row["images_qty"],
        )

    async def get_item_with_user(self, item_id: int) -> ItemWithUser:
        if self.cache_storage is not None:
            cached_item = await self.cache_storage.get_item_with_user(item_id)
            if cached_item is not None:
                return ItemWithUser(
                    item_id=cached_item["item_id"],
                    seller_id=cached_item["seller_id"],
                    is_verified_seller=cached_item["is_verified_seller"],
                    name=cached_item["name"],
                    description=cached_item["description"],
                    category=cached_item["category"],
                    images_qty=cached_item["images_qty"],
                )

        row = await self.storage.select_item_with_user(item_id)
        if row is None:
            raise ItemNotFoundError("Объявление не найдено")

        item = ItemWithUser(
            item_id=row["item_id"],
            seller_id=row["seller_id"],
            is_verified_seller=row["is_verified_seller"],
            name=row["name"],
            description=row["description"],
            category=row["category"],
            images_qty=row["images_qty"],
        )
        if self.cache_storage is not None:
            await self.cache_storage.set_item_with_user(item)
        return item

    async def close_item(self, item_id: int) -> None:
        closed = await self.storage.close(item_id)
        if not closed:
            raise ItemNotFoundError("Объявление не найдено")
        if self.cache_storage is not None:
            await self.cache_storage.delete_item_with_user(item_id)
            await self.cache_storage.delete_prediction(item_id)

    async def get_prediction(self, item_id: int) -> Optional[dict]:
        if self.cache_storage is None:
            return None
        return await self.cache_storage.get_prediction(item_id)

    async def set_prediction(self, item_id: int, result: dict) -> None:
        if self.cache_storage is not None:
            await self.cache_storage.set_prediction(item_id, result)

    async def delete_prediction(self, item_id: int) -> None:
        if self.cache_storage is not None:
            await self.cache_storage.delete_prediction(item_id)
