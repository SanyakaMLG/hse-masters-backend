from prometheus_client import Counter, Histogram

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
