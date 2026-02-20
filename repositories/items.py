from typing import Optional

import asyncpg

from models.moderation import Item, ItemWithUser


class ItemRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def create_item(
        self,
        user_id: int,
        name: str,
        description: str,
        category: int,
        images_qty: int,
    ) -> Item:
        async with self._pool.acquire() as conn:
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
        return Item(
            id=row["id"],
            user_id=row["user_id"],
            name=row["name"],
            description=row["description"],
            category=row["category"],
            images_qty=row["images_qty"],
        )

    async def get_item_with_user(self, item_id: int) -> Optional[ItemWithUser]:
        async with self._pool.acquire() as conn:
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
                WHERE i.id = $1
                """,
                item_id,
            )
        if row is None:
            return None

        return ItemWithUser(
            item_id=row["item_id"],
            seller_id=row["seller_id"],
            is_verified_seller=row["is_verified_seller"],
            name=row["name"],
            description=row["description"],
            category=row["category"],
            images_qty=row["images_qty"],
        )

    async def close_item(self, item_id: int) -> bool:
        async with self._pool.acquire() as conn:
            status = await conn.execute(
                "UPDATE items SET is_closed = TRUE WHERE id = $1", item_id
            )

            await conn.execute(
                "DELETE FROM moderation_results WHERE item_id = $1", item_id
            )
            return status == "UPDATE 1"
