import dataclasses
from typing import Annotated

import asyncpg
from fastapi import APIRouter, Depends, HTTPException
from redis.asyncio import Redis

from clients.db import get_db_pool_dependency
from clients.kafka import KafkaClient, get_kafka_client_dependency
from clients.redis import get_redis_dependency
from errors import ItemNotFoundError, ModelNotLoadedError
from repositories.cache import PredictionCacheRepository
from repositories.items import ItemRepository
from repositories.moderation_results import ModerationResultRepository
from schemas.moderation import (
    AsyncPredictResponse,
    PredictionRequest,
    PredictionResponse,
    TaskResultResponse,
)
from services.moderation_service import ModerationService

root_router = APIRouter()


@root_router.post("/", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        return ModerationService.predict(request)
    except ModelNotLoadedError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Ошибка при обработке запроса: {str(e)}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обработке запроса: {str(e)}",
        ) from e


@root_router.get("/simple_predict", response_model=PredictionResponse)
async def simple_predict(
    item_id: int, pool: Annotated[asyncpg.Pool, Depends(get_db_pool_dependency)]
):
    try:
        return await ModerationService.simple_predict(item_id, pool)
    except ItemNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ModelNotLoadedError as e:
        raise HTTPException(
            status_code=503, detail=f"Ошибка при обработке запроса: {str(e)}"
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Ошибка при обработке запроса: {str(e)}"
        ) from e


@root_router.post("/async_predict", response_model=AsyncPredictResponse)
async def async_predict(
    item_id: int,
    pool: Annotated[asyncpg.Pool, Depends(get_db_pool_dependency)],
    kafka_client: Annotated[KafkaClient, Depends(get_kafka_client_dependency)],
):
    item_repo = ItemRepository(pool)
    item = await item_repo.get_item_with_user(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Объявление не найдено")

    mod_repo = ModerationResultRepository(pool)
    task_id = await mod_repo.create_task(item_id)

    try:
        await kafka_client.send_moderation_request(item_id)
    except Exception as e:
        await mod_repo.update_task(task_id, status="failed", error_message=str(e))
        raise HTTPException(
            status_code=500, detail="Ошибка при отправке в очередь обработки"
        ) from e

    return AsyncPredictResponse(
        task_id=task_id, status="pending", message="Moderation request accepted"
    )


@root_router.get("/moderation_result/{task_id}", response_model=TaskResultResponse)
async def get_moderation_result(
    task_id: int,
    pool: Annotated[asyncpg.Pool, Depends(get_db_pool_dependency)],
    redis: Annotated[Redis, Depends(get_redis_dependency)],
):
    mod_repo = ModerationResultRepository(pool)
    result = await mod_repo.get_task(task_id)

    if not result:
        raise HTTPException(status_code=404, detail="Задача не найдена")

    response_data = TaskResultResponse(**dataclasses.asdict(result))

    if result.status == "completed":
        cache_repo = PredictionCacheRepository(redis)
        await cache_repo.set_prediction(result.item_id, response_data.model_dump())

    return response_data


@root_router.post("/close")
async def close_item(
    item_id: int,
    pool: Annotated[asyncpg.Pool, Depends(get_db_pool_dependency)],
    redis: Annotated[Redis, Depends(get_redis_dependency)],
):
    item_repo = ItemRepository(pool)
    cache_repo = PredictionCacheRepository(redis)

    closed = await item_repo.close_item(item_id)
    if not closed:
        raise HTTPException(status_code=404, detail="Объявление не найдено")

    await cache_repo.delete_prediction(item_id)

    return {
        "status": "success",
        "message": f"Item {item_id} is closed and cache cleared",
    }
