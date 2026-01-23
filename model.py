from mlflow.exceptions import MlflowTraceDataNotFound
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
import mlflow
import os
from typing import Optional


def train_model():
    np.random.seed(42)
    X = np.random.rand(1000, 4)
    y = (X[:, 0] < 0.3) & (X[:, 1] < 0.2)
    y = y.astype(int)
    
    model = LogisticRegression()
    model.fit(X, y)
    return model


def save_model(model, path: str = "model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str = "model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model_mlflow(
    model,
    model_name: str = "moderation_model",
    registered_model_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    experiment_name: str = "moderation"
):
    if tracking_uri is None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
    
    mlflow.set_tracking_uri(tracking_uri)
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
    except Exception:
        try:
            mlflow.create_experiment(experiment_name)
        except Exception:
            experiment = mlflow.get_experiment_by_name(experiment_name)
    
    mlflow.set_experiment(experiment_name)
    
    if registered_model_name is None:
        registered_model_name = model_name
    
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=model_name,
            registered_model_name=registered_model_name
        )
        model_uri = mlflow.get_artifact_uri(model_name)
    
    return model_uri


def load_model_mlflow(
    model_name: Optional[str] = None,
    tracking_uri: Optional[str] = None
):
    if tracking_uri is None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
    
    mlflow.set_tracking_uri(tracking_uri)
    
    try:
        return mlflow.sklearn.load_model(f"models:/{model_name}/latest")
    except Exception as e:
        raise FileNotFoundError(f"Модель '{model_name}' не найдена или произошла ошибка при загрузке: {e}")
