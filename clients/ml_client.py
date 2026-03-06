import logging
import os
import time
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from errors import ModelNotLoadedError
from model import (
    load_model_local,
    load_model_mlflow,
    save_model,
    save_model_mlflow,
    train_model,
)
from utils.metrics import PREDICTION_DURATION, PREDICTION_ERRORS_TOTAL

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class MLClient:
    model: Optional[LogisticRegression] = None

    @classmethod
    def load(cls, model_name: str = "moderation_model", model_path: str = "model.pkl"):
        use_mlflow = os.getenv("USE_MLFLOW", "false") == "true"

        if use_mlflow:
            try:
                cls.model = load_model_mlflow(model_name)
                logger.info(f"Модель загружена из MLflow: {model_name}")
            except FileNotFoundError:
                logger.info(
                    f"Модель не найдена в MLflow: {model_name}. Запускаем обучение"
                )
                cls.model = train_model()
                save_model_mlflow(cls.model, model_name)
                logger.info(f"Модель сохранена и загружена из MLflow: {model_name}")
        else:
            try:
                cls.model = load_model_local(model_path)
                logger.info(f"Модель загружена из локального файла: {model_path}")
            except FileNotFoundError:
                logger.info(
                    f"Модель не найдена локально: {model_path}. Запускаем обучение"
                )
                cls.model = train_model()
                save_model(cls.model, model_path)
                logger.info(f"Модель обучена, сохраняем локально: {model_path}")

    @classmethod
    def predict_proba(cls, features: list[float]) -> float:
        if cls.model is None:
            PREDICTION_ERRORS_TOTAL.labels(error_type="model_unavailable").inc()
            raise ModelNotLoadedError("Модель не загружена.")

        prepared_data = np.clip(np.array(features), 0.0, 1.0)
        logger.info(f"Обработанные признаки для модели: {prepared_data.tolist()}")

        try:
            start_time = time.time()
            prediction = cls.model.predict_proba(prepared_data.reshape(1, -1))
            duration = time.time() - start_time
            PREDICTION_DURATION.observe(duration)

            return float(prediction[0][1])
        except Exception as e:
            PREDICTION_ERRORS_TOTAL.labels(error_type="prediction_error").inc()
            raise e
