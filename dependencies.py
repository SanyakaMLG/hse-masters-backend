from typing import Annotated

import asyncpg
from fastapi import Cookie, Depends, HTTPException
from redis.asyncio import Redis

from clients.db import get_db_pool_dependency
from clients.kafka import KafkaClient, get_kafka_client_dependency
from clients.redis import get_redis_dependency
from models.moderation import Account
from repositories.accounts import AccountRepository
from repositories.items import ItemRepository
from repositories.moderation_results import ModerationResultRepository
from repositories.users import UserRepository
from services.auth_service import (
    AuthService,
    BlockedAccountError,
    InvalidTokenError,
)
from services.moderation_service import ModerationService
from services.prediction_service import PredictionWorkflowService


def get_account_repository(
    pool: Annotated[asyncpg.Pool, Depends(get_db_pool_dependency)],
    redis: Annotated[Redis, Depends(get_redis_dependency)],
) -> AccountRepository:
    return AccountRepository(pool, redis)


def get_item_repository(
    pool: Annotated[asyncpg.Pool, Depends(get_db_pool_dependency)],
    redis: Annotated[Redis, Depends(get_redis_dependency)],
) -> ItemRepository:
    return ItemRepository(pool, redis)


def get_moderation_result_repository(
    pool: Annotated[asyncpg.Pool, Depends(get_db_pool_dependency)],
    redis: Annotated[Redis, Depends(get_redis_dependency)],
) -> ModerationResultRepository:
    return ModerationResultRepository(pool, redis)


def get_user_repository(
    pool: Annotated[asyncpg.Pool, Depends(get_db_pool_dependency)],
    redis: Annotated[Redis, Depends(get_redis_dependency)],
) -> UserRepository:
    return UserRepository(pool, redis)


def get_auth_service(
    account_repository: Annotated[AccountRepository, Depends(get_account_repository)],
) -> AuthService:
    return AuthService(account_repository)


def get_prediction_workflow_service(
    item_repository: Annotated[ItemRepository, Depends(get_item_repository)],
    moderation_result_repository: Annotated[
        ModerationResultRepository, Depends(get_moderation_result_repository)
    ],
    kafka_client: Annotated[KafkaClient, Depends(get_kafka_client_dependency)],
) -> PredictionWorkflowService:
    return PredictionWorkflowService(
        item_repository=item_repository,
        moderation_result_repository=moderation_result_repository,
        kafka_client=kafka_client,
    )


def get_moderation_service(
    item_repository: Annotated[ItemRepository, Depends(get_item_repository)],
) -> ModerationService:
    return ModerationService(item_repository=item_repository)


async def get_current_account(
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
    access_token: Annotated[str | None, Cookie()] = None,
) -> Account:
    if not access_token:
        raise HTTPException(status_code=401, detail="Пользователь не авторизован")

    try:
        return await auth_service.get_account_from_token(access_token)
    except (InvalidTokenError, BlockedAccountError) as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
