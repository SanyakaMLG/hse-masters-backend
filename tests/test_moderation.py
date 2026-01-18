import pytest
from unittest.mock import patch
from models.moderation import PredictionRequest
from services.moderation_service import ModerationService


class TestModerationService:
    def test_verified_seller_always_approved(self):
        request = PredictionRequest(
            seller_id=1,
            is_verified_seller=True,
            item_id=100,
            name="Test Item",
            description="Test Description",
            category=1,
            images_qty=0
        )
        result = ModerationService.predict(request)
        assert result is True, "Подтвержденные продавцы должны всегда проходить модерацию"
    
    def test_unverified_seller_with_images_approved(self):
        request = PredictionRequest(
            seller_id=2,
            is_verified_seller=False,
            item_id=101,
            name="Test Item 2",
            description="Test Description 2",
            category=2,
            images_qty=3
        )
        result = ModerationService.predict(request)
        assert result is True, "Неподтвержденные продавцы с изображениями должны проходить модерацию"
    
    def test_unverified_seller_without_images_rejected(self):
        request = PredictionRequest(
            seller_id=3,
            is_verified_seller=False,
            item_id=102,
            name="Test Item 3",
            description="Test Description 3",
            category=3,
            images_qty=0
        )
        result = ModerationService.predict(request)
        assert result is False, "Неподтвержденные продавцы без изображений не должны проходить модерацию"


class TestPredictEndpoint:
    def test_predict_verified_seller_success(self, app_client):
        response = app_client.post(
            "/predict",
            json={
                "seller_id": 1,
                "is_verified_seller": True,
                "item_id": 100,
                "name": "Test Item",
                "description": "Test Description",
                "category": 1,
                "images_qty": 0
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "is_approved" in data
        assert data["is_approved"] is True
    
    def test_predict_unverified_seller_with_images_success(self, app_client):
        response = app_client.post(
            "/predict",
            json={
                "seller_id": 2,
                "is_verified_seller": False,
                "item_id": 101,
                "name": "Test Item 2",
                "description": "Test Description 2",
                "category": 2,
                "images_qty": 5
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "is_approved" in data
        assert data["is_approved"] is True
    
    def test_predict_unverified_seller_without_images_rejected(self, app_client):
        response = app_client.post(
            "/predict",
            json={
                "seller_id": 3,
                "is_verified_seller": False,
                "item_id": 102,
                "name": "Test Item 3",
                "description": "Test Description 3",
                "category": 3,
                "images_qty": 0
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "is_approved" in data
        assert data["is_approved"] is False
    
    def test_predict_validation_missing_field(self, app_client):
        response = app_client.post(
            "/predict",
            json={
                "seller_id": 1,
                "is_verified_seller": True,
                "name": "Test Item",
                "description": "Test Description",
                "category": 1,
                "images_qty": 0,
                # item_id
            }
        )
        assert response.status_code == 422
    
    def test_predict_validation_wrong_type(self, app_client):
        response = app_client.post(
            "/predict",
            json={
                "seller_id": "not_an_int",
                "is_verified_seller": True,
                "item_id": 100,
                "name": "Test Item",
                "description": "Test Description",
                "category": 1,
                "images_qty": 0,
            }
        )
        assert response.status_code == 422
    
    def test_predict_validation_negative_images(self, app_client):
        response = app_client.post(
            "/predict",
            json={
                "seller_id": 1,
                "is_verified_seller": True,
                "item_id": 100,
                "name": "Test Item",
                "description": "Test Description",
                "category": 1,
                "images_qty": -1,
            }
        )
        assert response.status_code == 422
    
    def test_predict_validation_empty_name(self, app_client):
        response = app_client.post(
            "/predict",
            json={
                "seller_id": 1,
                "is_verified_seller": True,
                "item_id": 100,
                "name": "",
                "description": "Test Description",
                "category": 1,
                "images_qty": 0
            }
        )
        assert response.status_code == 422
    
    def test_predict_business_logic_error_handling(self, app_client):
        with patch('services.moderation_service.ModerationService.predict') as mock_predict:
            mock_predict.side_effect = ValueError("Ошибка бизнес-логики: некорректные данные")
            
            response = app_client.post(
                "/predict",
                json={
                    "seller_id": 1,
                    "is_verified_seller": True,
                    "item_id": 100,
                    "name": "Test Item",
                    "description": "Test Description",
                    "category": 1,
                    "images_qty": 0
                }
            )
            
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "Ошибка при обработке запроса" in data["detail"]
            assert "Ошибка бизнес-логики" in data["detail"]