import dataclasses
from typing import Annotated

import sentry_sdk
from fastapi import APIRouter, Depends, HTTPException

from dependencies import (
    get_current_account,
    get_moderation_service,
    get_prediction_workflow_service,
)
from errors import ItemNotFoundError, ModelNotLoadedError, ModerationTaskNotFoundError
from models.moderation import Account, PredictionInput
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
    service: Annotated[ModerationService, Depends(get_moderation_service)],
    current_account: Annotated[Account, Depends(get_current_account)],
):
    try:
        result = await service.predict_with_cache(
            PredictionInput(**request.model_dump())
        )
        return PredictionResponse(**dataclasses.asdict(result))
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
    service: Annotated[ModerationService, Depends(get_moderation_service)],
    current_account: Annotated[Account, Depends(get_current_account)],
):
    try:
        result = await service.simple_predict(item_id)
        return PredictionResponse(**dataclasses.asdict(result))
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
    service: Annotated[
        PredictionWorkflowService, Depends(get_prediction_workflow_service)
    ],
    current_account: Annotated[Account, Depends(get_current_account)],
):
    try:
        result = await service.async_predict(item_id)
        return AsyncPredictResponse(**dataclasses.asdict(result))
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
    service: Annotated[
        PredictionWorkflowService, Depends(get_prediction_workflow_service)
    ],
    current_account: Annotated[Account, Depends(get_current_account)],
):
    try:
        result = await service.get_moderation_result(task_id)
        return TaskResultResponse(**dataclasses.asdict(result))
    except ModerationTaskNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@root_router.post("/close")
async def close_item(
    item_id: int,
    service: Annotated[
        PredictionWorkflowService, Depends(get_prediction_workflow_service)
    ],
    current_account: Annotated[Account, Depends(get_current_account)],
):
    try:
        return await service.close_item(item_id)
    except ItemNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
