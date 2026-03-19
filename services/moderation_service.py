import dataclasses
import logging

from clients.ml_client import MLClient
from models.moderation import PredictionInput, PredictionOutput
from repositories.items import ItemRepository
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
    def predict(cls, request: PredictionInput) -> PredictionOutput:
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

        return PredictionOutput(is_violation=is_violation, probability=probability)

    @staticmethod
    def _prediction_from_cache(cached_prediction: dict) -> PredictionOutput:
        return PredictionOutput(
            is_violation=cached_prediction["is_violation"],
            probability=cached_prediction["probability"],
        )

    async def predict_with_cache(self, request: PredictionInput) -> PredictionOutput:
        cached_prediction = await self._item_repository.get_prediction(request.item_id)
        if cached_prediction is not None:
            return self._prediction_from_cache(cached_prediction)

        prediction = self.predict(request)
        await self._item_repository.set_prediction(
            request.item_id, dataclasses.asdict(prediction)
        )

        return prediction

    async def simple_predict(self, item_id: int) -> PredictionOutput:
        cached_prediction = await self._item_repository.get_prediction(item_id)
        if cached_prediction is not None:
            return self._prediction_from_cache(cached_prediction)

        item_with_user = await self._item_repository.get_item_with_user(item_id)
        request_for_model = PredictionInput(**dataclasses.asdict(item_with_user))
        prediction = self.predict(request_for_model)
        await self._item_repository.set_prediction(
            item_id, dataclasses.asdict(prediction)
        )

        return prediction
