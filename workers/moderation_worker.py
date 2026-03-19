import asyncio
import json
import logging
import os
from datetime import UTC, datetime

import sentry_sdk
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from clients.db import create_standalone_db_pool
from clients.redis import create_standalone_redis
from errors import ItemNotFoundError
from models.moderation import PredictionInput
from repositories.items import ItemRepository
from repositories.moderation_results import ModerationResultRepository
from services.moderation_service import ModerationService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KAFKA_TOPIC = "moderation"
DLQ_TOPIC = "moderation_dlq"
GROUP_ID = "moderation_group"
MAX_RETRIES = 3
RETRY_DELAY = 2


async def send_to_dlq(producer: AIOKafkaProducer, original_msg: dict, error_msg: str):
    dlq_message = {
        "original_message": original_msg,
        "error": error_msg,
        "timestamp": datetime.now(UTC).isoformat(),
        "retry_count": MAX_RETRIES,
    }
    await producer.send_and_wait(DLQ_TOPIC, json.dumps(dlq_message).encode("utf-8"))
    logger.info(f"Message sent to DLQ: {dlq_message}")


async def process_message(msg_value, mod_repo, item_repo):
    try:
        data = json.loads(msg_value)
        item_id = data.get("item_id")

        if not item_id:
            raise ValueError("item_id missing in message")

        task_id = await mod_repo.get_latest_pending_task_id(item_id)

        if not task_id:
            logger.warning(f"No pending task found for item_id {item_id}")
            return

        try:
            item_with_user = await item_repo.get_item_with_user(item_id)
        except ItemNotFoundError:
            await mod_repo.update_task(
                task_id, status="failed", error_message="Item not found"
            )
            raise ValueError(f"Item {item_id} not found in DB") from None

        req = PredictionInput(
            seller_id=item_with_user.seller_id,
            is_verified_seller=item_with_user.is_verified_seller,
            item_id=item_with_user.item_id,
            name=item_with_user.name,
            description=item_with_user.description,
            category=item_with_user.category,
            images_qty=item_with_user.images_qty,
        )

        result = ModerationService.predict(req)

        await mod_repo.update_task(
            task_id,
            status="completed",
            is_violation=result.is_violation,
            probability=result.probability,
        )
        await item_repo.set_prediction(
            item_id,
            {
                "task_id": task_id,
                "status": "completed",
                "is_violation": result.is_violation,
                "probability": result.probability,
            },
        )
        logger.info(f"Task {task_id} completed for item {item_id}")

    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"Error processing message: {e}")
        raise e


async def consume():
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")

    try:
        ModerationService.load_model()
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"Failed to load model: {e}")
        return

    db_pool = await create_standalone_db_pool()
    redis_client = await create_standalone_redis()
    item_repo = ItemRepository(db_pool, redis_client)
    mod_repo = ModerationResultRepository(db_pool, redis_client)

    consumer = AIOKafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=bootstrap_servers,
        group_id=GROUP_ID,
        auto_offset_reset="earliest",
    )

    producer = AIOKafkaProducer(bootstrap_servers=bootstrap_servers)

    await consumer.start()
    await producer.start()

    logger.info("Worker started, waiting for messages...")

    try:
        async for msg in consumer:
            msg_value = msg.value
            if msg_value is None:
                logger.warning(
                    f"Received empty message (tombstone) at offset {msg.offset}"
                )
                continue

            logger.info(f"Received message: {msg.value}")

            for attempt in range(1, MAX_RETRIES + 2):
                try:
                    # тут тестовый рейз чтоб показать DLQ
                    # if random.random() < 0.5:
                    #     raise Exception("Random exception")
                    await process_message(msg.value, mod_repo, item_repo)
                    break
                except Exception as e:
                    if attempt <= MAX_RETRIES:
                        logger.warning(
                            f"Attempt {attempt} failed, retrying in {RETRY_DELAY}s..."
                        )
                        await asyncio.sleep(RETRY_DELAY)
                    else:
                        logger.error("All retries failed. Sending to DLQ.")
                        try:
                            try:
                                original_msg = json.loads(msg_value)
                            except Exception:
                                original_msg = {
                                    "raw": msg_value.decode("utf-8", errors="ignore")
                                }

                            await send_to_dlq(producer, original_msg, str(e))

                            try:
                                data = json.loads(msg_value)
                                item_id = data.get("item_id")
                                if item_id:
                                    task_id = await mod_repo.get_latest_pending_task_id(
                                        item_id
                                    )
                                    if task_id:
                                        await mod_repo.update_task(
                                            task_id,
                                            status="failed",
                                            error_message=(
                                                f"Max retries exceeded: {str(e)}"
                                            ),
                                        )
                            except Exception:
                                pass

                        except Exception as dlq_error:
                            sentry_sdk.capture_exception(dlq_error)
                            logger.critical(f"Failed to send to DLQ: {dlq_error}")
    finally:
        await consumer.stop()
        await producer.stop()
        await redis_client.aclose()
        await db_pool.close()


if __name__ == "__main__":
    asyncio.run(consume())
