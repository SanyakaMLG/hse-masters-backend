from fastapi import FastAPI, HTTPException
from models.moderation import PredictionRequest, PredictionResponse
from services.moderation_service import ModerationService
import uvicorn

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Moderation Service API"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        is_approved = ModerationService.predict(request)
        
        return PredictionResponse(is_approved=is_approved)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обработке запроса: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
