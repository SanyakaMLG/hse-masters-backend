from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from clients.db import close_db_pool, create_db_pool
from clients.kafka import close_kafka_producer, init_kafka_producer
from clients.redis import init_redis_pool
from routers.moderation import root_router
from services.moderation_service import ModerationService


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ModerationService.load_model()
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
    await create_db_pool(app)
    await init_redis_pool(app)
    try:
        await init_kafka_producer(app)
    except Exception as e:
        print(f"Ошибка при подключении к Kafka: {e}")

    try:
        yield
    finally:
        await close_kafka_producer(app)
        await init_redis_pool(app)
        await close_db_pool(app)


app = FastAPI(lifespan=lifespan)
app.include_router(root_router, prefix="/predict")


@app.get("/")
async def root():
    return {"message": "Moderation Service API"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
