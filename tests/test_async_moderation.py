import pytest
import os
import asyncio
from unittest.mock import AsyncMock, patch
from repositories.moderation_results import ModerationResultRepository
from repositories.items import ItemRepository
from repositories.users import UserRepository

@pytest.mark.anyio
async def test_async_predict_endpoint(app_async_client, db_pool):
    user_repo = UserRepository(db_pool)
    item_repo = ItemRepository(db_pool)
    
    user = await user_repo.create_user(is_verified_seller=False)
    item = await item_repo.create_item(
        user_id=user.id,
        name="Test Async Item",
        description="Async desc",
        category=1,
        images_qty=1
    )

    with patch("routers.moderation.KafkaClient") as MockKafkaClient:
        mock_instance = MockKafkaClient.return_value
        mock_instance.send_moderation_request = AsyncMock()

        resp = await app_async_client.post(
            "/predict/async_predict",
            params={"item_id": item.id}
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pending"
        assert "task_id" in data
        
        task_id = data["task_id"]
        
        mod_repo = ModerationResultRepository(db_pool)
        task = await mod_repo.get_task(task_id)
        assert task is not None
        assert task.item_id == item.id
        assert task.status == "pending"
        
        mock_instance.send_moderation_request.assert_called_once_with(item.id)

@pytest.mark.anyio
async def test_get_moderation_result(app_async_client, db_pool):
    item_repo = ItemRepository(db_pool)
    user_repo = UserRepository(db_pool)
    mod_repo = ModerationResultRepository(db_pool)

    user = await user_repo.create_user(is_verified_seller=True)
    item = await item_repo.create_item(user.id, "Name", "Desc", 1, 0)
    
    task_id = await mod_repo.create_task(item.id)
    await mod_repo.update_task(task_id, "completed", is_violation=False, probability=0.05)

    resp = await app_async_client.get(f"/predict/moderation_result/{task_id}")
    
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_id"] == task_id
    assert data["status"] == "completed"
    assert data["is_violation"] is False
    assert data["probability"] == 0.05

@pytest.mark.anyio
async def test_worker_logic(db_pool):
    from workers.moderation_worker import process_message
    
    user_repo = UserRepository(db_pool)
    item_repo = ItemRepository(db_pool)
    mod_repo = ModerationResultRepository(db_pool)

    user = await user_repo.create_user(is_verified_seller=False)
    item = await item_repo.create_item(user.id, "Worker Item", "Desc", 1, 0)
    task_id = await mod_repo.create_task(item.id)
    
    msg_value = f'{{"item_id": {item.id}, "timestamp": 123456}}'.encode('utf-8')
    
    from services.moderation_service import ModerationService
    if ModerationService.model is None:
        from model import train_model
        ModerationService.model = train_model()

    await process_message(msg_value, db_pool, item_repo, mod_repo)

    result = await mod_repo.get_task(task_id)
    assert result.status == "completed"
    assert result.probability is not None