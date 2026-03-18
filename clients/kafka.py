import json
import os
import time
from typing import Optional

from aiokafka import AIOKafkaProducer
from fastapi import FastAPI, Request

KAFKA_TOPIC = "moderation"


async def get_kafka_producer() -> AIOKafkaProducer:
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
    producer = AIOKafkaProducer(bootstrap_servers=bootstrap_servers)
    try:
        await producer.start()
    except Exception:
        await producer.stop()
        raise
    return producer


class KafkaClient:
    def __init__(self, app: Optional[FastAPI] = None):
        self.producer: Optional[AIOKafkaProducer] = None
        if app:
            self.producer = getattr(app.state, "kafka_producer", None)

    async def send_moderation_request(self, item_id: int):
        if not self.producer:
            temp_producer = await get_kafka_producer()
            try:
                await self._send(temp_producer, item_id)
            finally:
                await temp_producer.stop()
        else:
            await self._send(self.producer, item_id)

    async def _send(self, producer: AIOKafkaProducer, item_id: int):
        message = {"item_id": item_id, "timestamp": time.time()}
        value_json = json.dumps(message).encode("utf-8")
        await producer.send_and_wait(KAFKA_TOPIC, value_json)


async def init_kafka_producer(app: FastAPI):
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
    producer = AIOKafkaProducer(bootstrap_servers=bootstrap_servers)
    try:
        await producer.start()
    except Exception:
        await producer.stop()
        raise
    app.state.kafka_producer = producer


async def close_kafka_producer(app: FastAPI):
    producer = getattr(app.state, "kafka_producer", None)
    if producer:
        await producer.stop()


def get_kafka_client_dependency(request: Request) -> KafkaClient:
    return KafkaClient(request.app)
