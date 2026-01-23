import logging
import os
from models.moderation import PredictionRequest, PredictionResponse
from model import load_model_mlflow, save_model_mlflow, train_model, load_model, save_model
from errors import ModelNotLoadedError
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


THRESHOLD = 0.5


class ModerationService:
    model: LogisticRegression = None

    @classmethod
    def load_model(cls, model_name: str = "moderation_model", model_path: str = "model.pkl"):
        use_mlflow = os.getenv("USE_MLFLOW", "false") == "true"
        
        if use_mlflow:
            try:
                cls.model = load_model_mlflow(model_name)
                logger.info(f"Модель загружена из MLflow: {model_name}")
            except FileNotFoundError:
                logger.info(f"Модель не найдена в MLflow: {model_name}. Запускаем обучение")
                try:
                    model = train_model()
                    logger.info(f"Модель обучена, сохраняем в MLflow: {model_name}")
                    save_model_mlflow(model, model_name)
                    cls.model = model
                    logger.info(f"Модель сохранена и загружена: {model_name}")
                except Exception as e:
                    logger.error(f"Ошибка при сохранении модели в MLflow: {e}")
                    raise
        else:
            try:
                cls.model = load_model(model_path)
                logger.info(f"Модель загружена из локального файла: {model_path}")
            except FileNotFoundError:
                logger.info(f"Модель не найдена локально: {model_path}. Запускаем обучение")
                model = train_model()
                logger.info(f"Модель обучена, сохраняем локально: {model_path}")
                save_model(model, model_path)
                cls.model = model
                logger.info(f"Модель сохранена и загружена: {model_path}")
    
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
        
        if cls.model is None:
            raise ModelNotLoadedError("Модель не загружена.")

        import numpy as np
        
        prepared_data = np.array([
            1.0 if request.is_verified_seller else 0.0,
            request.images_qty / 10.0,
            len(request.description) / 1000.0,
            request.category / 100.0,
        ])
        
        prepared_data = np.clip(prepared_data, 0.0, 1.0)

        logger.info(f"Обработанные признаки для модели: {prepared_data.tolist()}")
        
        prediction = cls.model.predict_proba([prepared_data.tolist()])
        probability = float(prediction[0][1])
        is_violation = probability > THRESHOLD
        
        logger.info(
            f"Результат предсказания - seller_id: {request.seller_id}, "
            f"item_id: {request.item_id}, "
            f"is_violation: {is_violation}, "
            f"probability: {probability:.4f}"
        )
        
        return PredictionResponse(is_violation=is_violation, probability=probability)
