from models.moderation import PredictionRequest


class ModerationService:
    @staticmethod
    def predict(request: PredictionRequest) -> bool:
        if request.is_verified_seller:
            return True
        
        return request.images_qty > 0
