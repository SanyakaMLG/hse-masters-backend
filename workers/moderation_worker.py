import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime

import asyncpg
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from models.moderation import PredictionRequest
from services.moderation_service import ModerationService
from repositories.items import ItemRepository
from repositories.moderation_results import ModerationResultRepository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KAFKA_TOPIC = "moderation"
DLQ_TOPIC = "moderation_dlq"
GROUP_ID = "moderation_group"
MAX_RETRIES = 3
RETRY_DELAY = 2

async def get_db_pool():
    dsn = os.getenv("DATABASE_URL", "postgres://postgres:postgres@localhost:5432/hw")
    return await asyncpg.create_pool(dsn)

async def send_to_dlq(producer: AIOKafkaProducer, original_msg: dict, error_msg: str):
    dlq_message = {
        "original_message": original_msg,
        "error": error_msg,
        "timestamp": datetime.utcnow().isoformat(),
        "retry_count": MAX_RETRIES
    }
    await producer.send_and_wait(DLQ_TOPIC, json.dumps(dlq_message).encode("utf-8"))
    logger.info(f"Message sent to DLQ: {dlq_message}")

async def process_message(msg_value, db_pool, item_repo, mod_repo):
    try:
        data = json.loads(msg_value)
        item_id = data.get("item_id")
        
        if not item_id:
            raise ValueError("item_id missing in message")
        
        async with db_pool.acquire() as conn:
            task_id = await conn.fetchval(
                """
                SELECT id FROM moderation_results 
                WHERE item_id = $1 AND status = 'pending' 
                ORDER BY id DESC LIMIT 1
                """,
                item_id
            )
        
        if not task_id:
            logger.warning(f"No pending task found for item_id {item_id}")
            return

        item_with_user = await item_repo.get_item_with_user(item_id)
        if not item_with_user:
            await mod_repo.update_task(task_id, status="failed", error_message="Item not found")
            raise ValueError(f"Item {item_id} not found in DB")

        req = PredictionRequest(
            seller_id=item_with_user.seller_id,
            is_verified_seller=item_with_user.is_verified_seller,
            item_id=item_with_user.item_id,
            name=item_with_user.name,
            description=item_with_user.description,
            category=item_with_user.category,
            images_qty=item_with_user.images_qty
        )

        result = ModerationService.predict(req)

        await mod_repo.update_task(
            task_id, 
            status="completed", 
            is_violation=result.is_violation, 
            probability=result.probability
        )
        logger.info(f"Task {task_id} completed for item {item_id}")

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise e

async def consume():
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
    
    try:
        ModerationService.load_model()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    db_pool = await get_db_pool()
    item_repo = ItemRepository(db_pool)
    mod_repo = ModerationResultRepository(db_pool)

    consumer = AIOKafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=bootstrap_servers,
        group_id=GROUP_ID,
        auto_offset_reset='earliest'
    )
    
    producer = AIOKafkaProducer(bootstrap_servers=bootstrap_servers)

    await consumer.start()
    await producer.start()
    
    logger.info("Worker started, waiting for messages...")

    try:
        async for msg in consumer:
            logger.info(f"Received message: {msg.value}")
            
            for attempt in range(1, MAX_RETRIES + 2):
                try:
                    # тут тестовый рейз чтоб показать DLQ
                    # if random.random() < 0.5:
                    #     raise Exception("Random exception")
                    await process_message(msg.value, db_pool, item_repo, mod_repo)
                    break
                except Exception as e:
                    if attempt <= MAX_RETRIES:
                        logger.warning(f"Attempt {attempt} failed, retrying in {RETRY_DELAY}s...")
                        await asyncio.sleep(RETRY_DELAY)
                    else:
                        logger.error("All retries failed. Sending to DLQ.")
                        try:
                            try:
                                original_msg = json.loads(msg.value)
                            except:
                                original_msg = {"raw": msg.value.decode('utf-8', errors='ignore')}
                                
                            await send_to_dlq(producer, original_msg, str(e))
                            
                            try:
                                data = json.loads(msg.value)
                                item_id = data.get("item_id")
                                if item_id:
                                    async with db_pool.acquire() as conn:
                                        await conn.execute(
                                            "UPDATE moderation_results SET status='failed', error_message=$1 WHERE item_id=$2 AND status='pending'",
                                            f"Max retries exceeded: {str(e)}", item_id
                                        )
                            except:
                                pass
                                
                        except Exception as dlq_error:
                            logger.critical(f"Failed to send to DLQ: {dlq_error}")
    finally:
        await consumer.stop()
        await producer.stop()
        await db_pool.close()

if __name__ == "__main__":
    asyncio.run(consume())
