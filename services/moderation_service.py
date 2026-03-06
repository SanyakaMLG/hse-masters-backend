import dataclasses
import logging

from clients.ml_client import MLClient
from errors import ItemNotFoundError
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

    @classmethod
    async def simple_predict(cls, item_id: int, pool) -> PredictionResponse:
        item_repo = ItemRepository(pool)
        item_with_user = await item_repo.get_item_with_user(item_id)

        if not item_with_user:
            logger.warning(f"Объявление не найдено: item_id={item_id}")
            raise ItemNotFoundError("Объявление не найдено")

        request_for_model = PredictionRequest(**dataclasses.asdict(item_with_user))
        return cls.predict(request_for_model)
