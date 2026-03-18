import logging

from clients.ml_client import MLClient
from repositories.items import ItemRepository
from schemas.moderation import PredictionRequest, PredictionResponse
from utils.metrics import MODEL_PREDICTION_PROBABILITY, PREDICTIONS_TOTAL

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


THRESHOLD = 0.5


class ModerationService:
    model = None

    def __init__(
        self,
        item_repository: ItemRepository,
    ) -> None:
        self._item_repository = item_repository

    @classmethod
    def load_model(cls):
        MLClient.load()

    @classmethod
    def predict(cls, request: PredictionRequest) -> PredictionResponse:
        logger.info(
            f"Запрос на предсказание - seller_id: {request.seller_id}, "
            f"item_id: {request.item_id}, "
            f"is_verified_seller: {request.is_verified_seller}, "
            f"images_qty: {request.images_qty}, "
            f"description_length: {len(request.description)}, "
            f"category: {request.category}"
        )

        features = [
            1.0 if request.is_verified_seller else 0.0,
            request.images_qty / 10.0,
            len(request.description) / 1000.0,
            request.category / 100.0,
        ]

        probability = MLClient.predict_proba(features)
        is_violation = probability > THRESHOLD

        logger.info(
            f"Результат предсказания - seller_id: {request.seller_id}, "
            f"item_id: {request.item_id}, "
            f"is_violation: {is_violation}, "
            f"probability: {probability:.4f}"
        )

        result_label = "violation" if is_violation else "no_violation"
        PREDICTIONS_TOTAL.labels(result=result_label).inc()
        MODEL_PREDICTION_PROBABILITY.observe(probability)

        return PredictionResponse(is_violation=is_violation, probability=probability)

    async def predict_with_cache(
        self, request: PredictionRequest
    ) -> PredictionResponse:
        cached_prediction = await self._item_repository.get_prediction(request.item_id)
        if cached_prediction is not None:
            return PredictionResponse(**cached_prediction)

        prediction = self.predict(request)
        await self._item_repository.set_prediction(
            request.item_id, prediction.model_dump()
        )

        return prediction

    async def simple_predict(self, item_id: int) -> PredictionResponse:
        cached_prediction = await self._item_repository.get_prediction(item_id)
        if cached_prediction is not None:
            return PredictionResponse(**cached_prediction)

        item_with_user = await self._item_repository.get_item_with_user(item_id)
        request_for_model = PredictionRequest(
            seller_id=item_with_user.seller_id,
            is_verified_seller=item_with_user.is_verified_seller,
            item_id=item_with_user.item_id,
            name=item_with_user.name,
            description=item_with_user.description,
            category=item_with_user.category,
            images_qty=item_with_user.images_qty,
        )
        prediction = self.predict(request_for_model)
        await self._item_repository.set_prediction(item_id, prediction.model_dump())

        return prediction
