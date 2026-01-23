from fastapi import APIRouter, HTTPException
from services.moderation_service import ModerationService
from models.moderation import PredictionRequest, PredictionResponse
from errors import ModelNotLoadedError

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

