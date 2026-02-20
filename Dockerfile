FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV PDM_CHECK_UPDATE=false
ENV PDM_USE_VENV=false

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pdm

COPY pyproject.toml pdm.lock* /app/

RUN pdm install --prod --no-self

COPY . /app