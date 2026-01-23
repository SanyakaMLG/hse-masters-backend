from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    seller_id: int = Field(..., description="ID продавца")
    is_verified_seller: bool = Field(..., description="Подтвержден ли продавец")
    item_id: int = Field(..., description="ID товара")
    name: str = Field(..., min_length=1, description="Название объявления")
    description: str = Field(..., description="Описание объявления")
    category: int = Field(..., description="Категория товара")
    images_qty: int = Field(..., ge=0, description="Количество изображений")


class PredictionResponse(BaseModel):
    is_violation: bool = Field(..., description="Предсказание модели")
    probability: float = Field(..., description="Вероятность нарушения")
