import pytest
from aiokafka.errors import KafkaConnectionError

from clients.kafka import KafkaClient


@pytest.mark.integration
@pytest.mark.anyio
async def test_kafka_client_can_send_message_to_broker():
    client = KafkaClient()
    try:
        await client.send_moderation_request(123)
    except KafkaConnectionError as exc:
        pytest.skip(f"Kafka is unavailable for integration tests: {exc}")
