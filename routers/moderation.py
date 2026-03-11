from typing import Annotated

import asyncpg
import sentry_sdk
from fastapi import APIRouter, Depends, HTTPException
from redis.asyncio import Redis

from clients.db import get_db_pool_dependency
from clients.kafka import KafkaClient, get_kafka_client_dependency
from clients.redis import get_redis_dependency
from dependencies import get_current_account
from errors import ItemNotFoundError, ModelNotLoadedError, ModerationTaskNotFoundError
from models.moderation import Account
from schemas.moderation import (
    AsyncPredictResponse,
    PredictionRequest,
    PredictionResponse,
    TaskResultResponse,
)
from services.moderation_service import ModerationService
from services.prediction_service import PredictionWorkflowService

root_router = APIRouter()


@root_router.post("/", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    current_account: Annotated[Account, Depends(get_current_account)],
):
    try:
        return ModerationService.predict(request)
    except ModelNotLoadedError as e:
        sentry_sdk.capture_exception(e)
        raise HTTPException(
            status_code=503,
            detail=f"Ошибка при обработке запроса: {str(e)}",
        ) from e
    except Exception as e:
        sentry_sdk.capture_exception(e)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обработке запроса: {str(e)}",
        ) from e


@root_router.get("/simple_predict", response_model=PredictionResponse)
async def simple_predict(
    item_id: int,
    pool: Annotated[asyncpg.Pool, Depends(get_db_pool_dependency)],
    current_account: Annotated[Account, Depends(get_current_account)],
):
    try:
        return await ModerationService.simple_predict(item_id, pool)
    except ItemNotFoundError as e:
        sentry_sdk.capture_exception(e)
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ModelNotLoadedError as e:
        sentry_sdk.capture_exception(e)
        raise HTTPException(
            status_code=503, detail=f"Ошибка при обработке запроса: {str(e)}"
        ) from e
    except Exception as e:
        sentry_sdk.capture_exception(e)
        raise HTTPException(
            status_code=500, detail=f"Ошибка при обработке запроса: {str(e)}"
        ) from e


@root_router.post("/async_predict", response_model=AsyncPredictResponse)
async def async_predict(
    item_id: int,
    pool: Annotated[asyncpg.Pool, Depends(get_db_pool_dependency)],
    redis: Annotated[Redis, Depends(get_redis_dependency)],
    kafka_client: Annotated[KafkaClient, Depends(get_kafka_client_dependency)],
    current_account: Annotated[Account, Depends(get_current_account)],
):
    service = PredictionWorkflowService(pool, redis)
    try:
        return await service.async_predict(item_id, kafka_client)
    except ItemNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        sentry_sdk.capture_exception(e)
        raise HTTPException(
            status_code=500, detail="Ошибка при отправке в очередь обработки"
        ) from e


@root_router.get("/moderation_result/{task_id}", response_model=TaskResultResponse)
async def get_moderation_result(
    task_id: int,
    pool: Annotated[asyncpg.Pool, Depends(get_db_pool_dependency)],
    redis: Annotated[Redis, Depends(get_redis_dependency)],
    current_account: Annotated[Account, Depends(get_current_account)],
):
    service = PredictionWorkflowService(pool, redis)
    try:
        return await service.get_moderation_result(task_id)
    except ModerationTaskNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@root_router.post("/close")
async def close_item(
    item_id: int,
    pool: Annotated[asyncpg.Pool, Depends(get_db_pool_dependency)],
    redis: Annotated[Redis, Depends(get_redis_dependency)],
    current_account: Annotated[Account, Depends(get_current_account)],
):
    service = PredictionWorkflowService(pool, redis)
    try:
        return await service.close_item(item_id)
    except ItemNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
