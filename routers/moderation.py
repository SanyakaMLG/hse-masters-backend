from fastapi import APIRouter, Depends, HTTPException, Request
from services.moderation_service import ModerationService
from models.moderation import PredictionRequest, PredictionResponse, AsyncPredictResponse, TaskResultResponse
from errors import ModelNotLoadedError
from repositories.items import ItemRepository
from repositories.moderation_results import ModerationResultRepository
from clients.kafka import KafkaClient


def get_db_pool(request: Request):
    pool = getattr(request.app.state, "db_pool", None)
    if pool is None:
        raise HTTPException(
            status_code=503,
            detail="База данных не настроена",
        )
    return pool


def get_kafka_client(request: Request) -> KafkaClient:
    return KafkaClient(request.app)


root_router = APIRouter()

@root_router.post("/", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        return ModerationService.predict(request)
    except ModelNotLoadedError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Ошибка при обработке запроса: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обработке запроса: {str(e)}",
        )


@root_router.get("/simple_predict", response_model=PredictionResponse)
async def simple_predict(
    item_id: int,
    pool=Depends(get_db_pool),
):
    item_repo = ItemRepository(pool)
    item_with_user = await item_repo.get_item_with_user(item_id)

    if item_with_user is None:
        raise HTTPException(status_code=404, detail="Объявление не найдено")

    request_for_model = PredictionRequest(
        seller_id=item_with_user.seller_id,
        is_verified_seller=bool(item_with_user.is_verified_seller),
        item_id=item_with_user.item_id,
        name=item_with_user.name,
        description=item_with_user.description,
        category=item_with_user.category,
        images_qty=item_with_user.images_qty,
    )

    try:
        return ModerationService.predict(request_for_model)
    except ModelNotLoadedError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Ошибка при обработке запроса: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обработке запроса: {str(e)}",
        )


@root_router.post("/async_predict", response_model=AsyncPredictResponse)
async def async_predict(
    item_id: int,
    pool=Depends(get_db_pool),
    kafka_client: KafkaClient = Depends(get_kafka_client)
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
        raise HTTPException(status_code=500, detail="Ошибка при отправке в очередь обработки")

    return AsyncPredictResponse(
        task_id=task_id,
        status="pending",
        message="Moderation request accepted"
    )


@root_router.get("/moderation_result/{task_id}", response_model=TaskResultResponse)
async def get_moderation_result(
    task_id: int,
    pool=Depends(get_db_pool)
):
    mod_repo = ModerationResultRepository(pool)
    result = await mod_repo.get_task(task_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Задача не найдена")
        
    return TaskResultResponse(
        task_id=result.task_id,
        status=result.status,
        is_violation=result.is_violation,
        probability=result.probability
    )
