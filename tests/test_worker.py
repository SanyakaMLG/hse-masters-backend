import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from workers.moderation_worker import consume, process_message, send_to_dlq


@pytest.mark.anyio
class TestModerationWorker:
    @patch("workers.moderation_worker.ModerationService.predict")
    async def test_process_message_success(self, mock_predict):
        from repositories.items import ItemWithUser
        from schemas.moderation import PredictionResponse

        mock_predict.return_value = PredictionResponse(
            is_violation=False, probability=0.2
        )

        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 10

        mock_acquire_context = AsyncMock()
        mock_acquire_context.__aenter__.return_value = mock_conn
        mock_acquire_context.__aexit__.return_value = False

        mock_db_pool = MagicMock()
        mock_db_pool.acquire.return_value = mock_acquire_context

        mock_item_repo = AsyncMock()
        mock_item_repo.get_item_with_user.return_value = ItemWithUser(
            item_id=1,
            seller_id=1,
            is_verified_seller=False,
            name="W",
            description="D",
            category=1,
            images_qty=0,
        )

        mock_mod_repo = AsyncMock()

        msg_value = json.dumps({"item_id": 1, "timestamp": 123456}).encode("utf-8")

        await process_message(msg_value, mock_db_pool, mock_item_repo, mock_mod_repo)

        mock_mod_repo.update_task.assert_called_once_with(
            10, status="completed", is_violation=False, probability=0.2
        )
        mock_predict.assert_called_once()

    @patch("workers.moderation_worker.AIOKafkaConsumer")
    @patch("workers.moderation_worker.AIOKafkaProducer")
    @patch("workers.moderation_worker.create_standalone_db_pool")
    @patch("workers.moderation_worker.ModerationService.load_model")
    async def test_consume_tombstone_skip(
        self, mock_load, mock_db, mock_producer, mock_consumer
    ):

        mock_msg = MagicMock()
        mock_msg.value = None
        mock_msg.offset = 123

        mock_consumer.return_value.__aiter__.return_value = [mock_msg]

        mock_consumer.return_value.start = AsyncMock()
        mock_consumer.return_value.stop = AsyncMock()
        mock_producer.return_value.start = AsyncMock()
        mock_producer.return_value.stop = AsyncMock()

        mock_db.return_value = AsyncMock()

        await consume()

        mock_consumer.return_value.start.assert_called()
        mock_consumer.return_value.stop.assert_called()

    @patch("workers.moderation_worker.send_to_dlq", new_callable=AsyncMock)
    @patch("workers.moderation_worker.process_message", new_callable=AsyncMock)
    @patch("workers.moderation_worker.AIOKafkaConsumer")
    @patch("workers.moderation_worker.AIOKafkaProducer")
    @patch("workers.moderation_worker.create_standalone_db_pool")
    @patch("workers.moderation_worker.asyncio.sleep", new_callable=AsyncMock)
    @patch("workers.moderation_worker.ModerationService.load_model")
    async def test_worker_retries_and_dlq(
        self,
        mock_load,
        mock_sleep,
        mock_db,
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

        mock_conn = AsyncMock()

        mock_acquire_context = AsyncMock()
        mock_acquire_context.__aenter__.return_value = mock_conn

        mock_pool = MagicMock()
        mock_pool.acquire.return_value = mock_acquire_context
        mock_pool.close = AsyncMock()

        mock_db.return_value = mock_pool

        from workers.moderation_worker import consume

        await consume()

        assert mock_process.call_count == 4
        mock_dlq.assert_called_once()

        assert mock_conn.execute.called
        args, _ = mock_conn.execute.call_args
        assert "UPDATE moderation_results" in args[0]
        assert "Max retries exceeded" in args[1]

    async def test_process_message_item_not_found(self):
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value.fetchval = AsyncMock(
            return_value=10
        )

        mock_item_repo = AsyncMock()
        mock_item_repo.get_item_with_user.return_value = None

        mock_mod_repo = AsyncMock()

        msg = json.dumps({"item_id": 999}).encode("utf-8")

        with pytest.raises(ValueError, match="not found in DB"):
            await process_message(msg, mock_pool, mock_item_repo, mock_mod_repo)

        mock_mod_repo.update_task.assert_called_with(
            10, status="failed", error_message="Item not found"
        )

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
