from fastapi import APIRouter, Depends, HTTPException, Request
from services.moderation_service import ModerationService
from models.moderation import PredictionRequest, PredictionResponse
from errors import ModelNotLoadedError
from repositories.items import ItemRepository

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


def get_db_pool(request: Request):
    pool = getattr(request.app.state, "db_pool", None)
    if pool is None:
        raise HTTPException(
            status_code=503,
            detail="База данных не настроена",
        )
    return pool


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
