from fastapi import FastAPI
from routers.moderation import root_router
from services.moderation_service import ModerationService
import uvicorn
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        import os
        use_mlflow = os.getenv("USE_MLFLOW", "false") == "true"
        if use_mlflow:
            ModerationService.load_model("moderation_model")
        else:
            ModerationService.load_model(model_path="model.pkl")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(root_router, prefix="/predict")


@app.get("/")
async def root():
    return {"message": "Moderation Service API"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
