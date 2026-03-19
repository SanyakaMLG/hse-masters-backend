import asyncio
import json
import logging
import os
from datetime import UTC, datetime

import sentry_sdk
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from dependencies_worker import create_worker_runtime
from services.moderation_service import ModerationService
from services.worker_service import WorkerService

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


async def process_message(msg_value, worker_service: WorkerService):
    try:
        data = json.loads(msg_value)
        item_id = data.get("item_id")

        if not item_id:
            raise ValueError("item_id missing in message")

        task_id = await worker_service.moderate_pending_item(item_id)
        if not task_id:
            logger.warning(f"No pending task found for item_id {item_id}")
            return
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

    runtime = await create_worker_runtime()
    worker_service = runtime.worker_service

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
                    await process_message(msg.value, worker_service)
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
                                    await worker_service.mark_latest_task_failed(
                                        item_id,
                                        f"Max retries exceeded: {str(e)}",
                                    )
                            except Exception:
                                pass

                        except Exception as dlq_error:
                            sentry_sdk.capture_exception(dlq_error)
                            logger.critical(f"Failed to send to DLQ: {dlq_error}")
    finally:
        await consumer.stop()
        await producer.stop()
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(consume())
