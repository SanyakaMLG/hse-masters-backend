import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from errors import ItemNotFoundError
from workers.moderation_worker import consume, process_message, send_to_dlq


@pytest.mark.anyio
class TestModerationWorker:
    async def test_process_message_success(self):
        worker_service = AsyncMock()
        worker_service.moderate_pending_item.return_value = 10
        msg_value = json.dumps({"item_id": 1, "timestamp": 123456}).encode("utf-8")

        await process_message(msg_value, worker_service)

        worker_service.moderate_pending_item.assert_awaited_once_with(1)

    async def test_process_message_without_item_id(self):
        with pytest.raises(ValueError, match="item_id missing"):
            await process_message(b"{}", AsyncMock())

    async def test_process_message_without_pending_task(self):
        worker_service = AsyncMock()
        worker_service.moderate_pending_item.return_value = None

        result = await process_message(
            json.dumps({"item_id": 1}).encode("utf-8"),
            worker_service,
        )

        assert result is None

    async def test_process_message_item_not_found(self):
        worker_service = AsyncMock()
        worker_service.moderate_pending_item.side_effect = ItemNotFoundError(
            "Объявление не найдено"
        )

        msg = json.dumps({"item_id": 999}).encode("utf-8")

        with pytest.raises(ItemNotFoundError, match="Объявление не найдено"):
            await process_message(msg, worker_service)

    @patch("workers.moderation_worker.AIOKafkaConsumer")
    @patch("workers.moderation_worker.AIOKafkaProducer")
    @patch("workers.moderation_worker.create_worker_runtime")
    @patch("workers.moderation_worker.ModerationService.load_model")
    async def test_consume_tombstone_skip(
        self, mock_load, mock_runtime_factory, mock_producer, mock_consumer
    ):
        mock_msg = MagicMock()
        mock_msg.value = None
        mock_msg.offset = 123

        mock_consumer.return_value.__aiter__.return_value = [mock_msg]
        mock_consumer.return_value.start = AsyncMock()
        mock_consumer.return_value.stop = AsyncMock()
        mock_producer.return_value.start = AsyncMock()
        mock_producer.return_value.stop = AsyncMock()

        runtime = AsyncMock()
        runtime.worker_service = AsyncMock()
        mock_runtime_factory.return_value = runtime
        await consume()

        mock_consumer.return_value.start.assert_called()
        mock_consumer.return_value.stop.assert_called()

    @patch("workers.moderation_worker.logger")
    @patch("workers.moderation_worker.ModerationService.load_model")
    async def test_consume_load_model_failure(self, mock_load, mock_logger):
        mock_load.side_effect = RuntimeError("model fail")

        await consume()

        mock_logger.error.assert_called_once()

    @patch("workers.moderation_worker.send_to_dlq", new_callable=AsyncMock)
    @patch("workers.moderation_worker.process_message", new_callable=AsyncMock)
    @patch("workers.moderation_worker.AIOKafkaConsumer")
    @patch("workers.moderation_worker.AIOKafkaProducer")
    @patch("workers.moderation_worker.create_worker_runtime")
    @patch("workers.moderation_worker.asyncio.sleep", new_callable=AsyncMock)
    @patch("workers.moderation_worker.ModerationService.load_model")
    async def test_worker_retries_and_dlq(
        self,
        mock_load,
        mock_sleep,
        mock_runtime_factory,
        mock_prod,
        mock_cons,
        mock_process,
        mock_dlq,
    ):
        mock_msg = MagicMock()
        mock_msg.value = b'{"item_id": 1}'
        mock_cons.return_value.__aiter__.return_value = [mock_msg]
        mock_cons.return_value.start = AsyncMock()
        mock_cons.return_value.stop = AsyncMock()
        mock_prod.return_value.start = AsyncMock()
        mock_prod.return_value.stop = AsyncMock()
        mock_process.side_effect = Exception("Fatal Kafka/DB Error")

        worker_service = AsyncMock()
        runtime = AsyncMock()
        runtime.worker_service = worker_service
        mock_runtime_factory.return_value = runtime
        await consume()

        assert mock_process.call_count == 4
        mock_dlq.assert_called_once()
        worker_service.mark_latest_task_failed.assert_awaited_once()
        args, _ = worker_service.mark_latest_task_failed.await_args
        assert args[0] == 1
        assert "Max retries exceeded" in args[1]

    async def test_send_to_dlq_execution(self):
        mock_producer = AsyncMock()
        original_msg = {"item_id": 1, "test": "data"}
        error_msg = "Test error"

        await send_to_dlq(mock_producer, original_msg, error_msg)

        assert mock_producer.send_and_wait.called

        args, _ = mock_producer.send_and_wait.call_args
        topic = args[0]
        payload = json.loads(args[1].decode("utf-8"))

        assert topic == "moderation_dlq"
        assert payload["original_message"] == original_msg
        assert payload["error"] == error_msg
        assert "timestamp" in payload

    @patch("workers.moderation_worker.send_to_dlq", new_callable=AsyncMock)
    @patch("workers.moderation_worker.process_message", new_callable=AsyncMock)
    @patch("workers.moderation_worker.AIOKafkaConsumer")
    @patch("workers.moderation_worker.AIOKafkaProducer")
    @patch("workers.moderation_worker.create_worker_runtime")
    @patch("workers.moderation_worker.asyncio.sleep", new_callable=AsyncMock)
    @patch("workers.moderation_worker.ModerationService.load_model")
    async def test_worker_dlq_uses_raw_message_on_bad_json(
        self,
        mock_load,
        mock_sleep,
        mock_runtime_factory,
        mock_prod,
        mock_cons,
        mock_process,
        mock_dlq,
    ):
        mock_msg = MagicMock()
        mock_msg.value = b"not-json"
        mock_cons.return_value.__aiter__.return_value = [mock_msg]
        mock_cons.return_value.start = AsyncMock()
        mock_cons.return_value.stop = AsyncMock()
        mock_prod.return_value.start = AsyncMock()
        mock_prod.return_value.stop = AsyncMock()
        mock_process.side_effect = Exception("boom")

        runtime = AsyncMock()
        runtime.worker_service = AsyncMock()
        mock_runtime_factory.return_value = runtime
        await consume()

        args, _ = mock_dlq.await_args
        assert args[1] == {"raw": "not-json"}

    @patch("workers.moderation_worker.send_to_dlq", new_callable=AsyncMock)
    @patch("workers.moderation_worker.process_message", new_callable=AsyncMock)
    @patch("workers.moderation_worker.AIOKafkaConsumer")
    @patch("workers.moderation_worker.AIOKafkaProducer")
    @patch("workers.moderation_worker.create_worker_runtime")
    @patch("workers.moderation_worker.asyncio.sleep", new_callable=AsyncMock)
    @patch("workers.moderation_worker.ModerationService.load_model")
    @patch("workers.moderation_worker.logger")
    async def test_worker_logs_dlq_failure(
        self,
        mock_logger,
        mock_load,
        mock_sleep,
        mock_runtime_factory,
        mock_prod,
        mock_cons,
        mock_process,
        mock_dlq,
    ):
        mock_msg = MagicMock()
        mock_msg.value = b'{"item_id": 1}'
        mock_cons.return_value.__aiter__.return_value = [mock_msg]
        mock_cons.return_value.start = AsyncMock()
        mock_cons.return_value.stop = AsyncMock()
        mock_prod.return_value.start = AsyncMock()
        mock_prod.return_value.stop = AsyncMock()
        mock_process.side_effect = Exception("boom")
        mock_dlq.side_effect = RuntimeError("dlq fail")

        worker_service = AsyncMock()
        runtime = AsyncMock()
        runtime.worker_service = worker_service
        mock_runtime_factory.return_value = runtime
        await consume()

        mock_logger.critical.assert_called_once()
