import os

import redis.asyncio as redis
from fastapi import FastAPI, Request


async def init_redis_pool(app: FastAPI):
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    app.state.redis = redis.from_url(redis_url, decode_responses=True)


async def close_redis_pool(app: FastAPI):
    if hasattr(app.state, "redis"):
        await app.state.redis.aclose()
        app.state.redis = None


async def create_standalone_redis() -> redis.Redis:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    return redis.from_url(redis_url, decode_responses=True)


def get_redis_dependency(request: Request) -> redis.Redis:
    return request.app.state.redis
