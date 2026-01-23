#!/bin/bash
PATH_TO_MLRUNS="/Users/alexandr.alekseev/hse/hse-masters-backend/mlruns"
mlflow server --backend-store-uri "file://${PATH_TO_MLRUNS}" --default-artifact-root "file://${PATH_TO_MLRUNS}" --host 0.0.0.0 --port 5001
