import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from prometheus_client import Counter, Histogram

DB_QUERY_TYPE_SELECT = "select"
DB_QUERY_TYPE_INSERT = "insert"
DB_QUERY_TYPE_UPDATE = "update"
DB_QUERY_TYPE_DELETE = "delete"

PREDICTIONS_TOTAL = Counter(
    "predictions_total", "Total number of predictions", ["result"]
)

PREDICTION_DURATION = Histogram(
    "prediction_duration_seconds", "Time spent on ML model inference"
)

PREDICTION_ERRORS_TOTAL = Counter(
    "prediction_errors_total", "Number of prediction errors", ["error_type"]
)

DB_QUERY_DURATION = Histogram(
    "db_query_duration_seconds", "Time spent on DB queries", ["query_type"]
)

MODEL_PREDICTION_PROBABILITY = Histogram(
    "model_prediction_probability",
    "Distribution of model prediction probabilities",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


@asynccontextmanager
async def observe_duration(histogram: Histogram, **labels: str) -> AsyncIterator[None]:
    start_time = time.perf_counter()
    try:
        yield
    finally:
        histogram.labels(**labels).observe(time.perf_counter() - start_time)


@asynccontextmanager
async def observe_db_select() -> AsyncIterator[None]:
    async with observe_duration(DB_QUERY_DURATION, query_type=DB_QUERY_TYPE_SELECT):
        yield


@asynccontextmanager
async def observe_db_insert() -> AsyncIterator[None]:
    async with observe_duration(DB_QUERY_DURATION, query_type=DB_QUERY_TYPE_INSERT):
        yield


@asynccontextmanager
async def observe_db_update() -> AsyncIterator[None]:
    async with observe_duration(DB_QUERY_DURATION, query_type=DB_QUERY_TYPE_UPDATE):
        yield


@asynccontextmanager
async def observe_db_delete() -> AsyncIterator[None]:
    async with observe_duration(DB_QUERY_DURATION, query_type=DB_QUERY_TYPE_DELETE):
        yield
