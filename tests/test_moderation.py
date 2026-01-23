import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from models.moderation import PredictionRequest, PredictionResponse
from services.moderation_service import ModerationService
from errors import ModelNotLoadedError


class TestModerationService:
    @pytest.fixture(autouse=True)
    def setup_model(self):
        from model import train_model
        ModerationService.model = train_model()
        yield
        ModerationService.model = None
    
    def test_predict_success_is_violation_true(self):
        request = PredictionRequest(
            seller_id=1,
            is_verified_seller=False,
            item_id=100,
            name="Test Item",
            description="Short description",
            category=1,
            images_qty=0
        )
        
        result = ModerationService.predict(request)
        
        assert isinstance(result, PredictionResponse)
        assert hasattr(result, 'is_violation')
        assert hasattr(result, 'probability')
        assert isinstance(result.is_violation, bool)
        assert isinstance(result.probability, float)
        assert 0.0 <= result.probability <= 1.0
    
    def test_predict_success_is_violation_false(self):
        request = PredictionRequest(
            seller_id=2,
            is_verified_seller=True,
            item_id=101,
            name="Test Item 2",
            description="Long description with many words to make it longer",
            category=2,
            images_qty=5
        )
        
        result = ModerationService.predict(request)
        
        assert isinstance(result, PredictionResponse)
        assert hasattr(result, 'is_violation')
        assert hasattr(result, 'probability')
        assert isinstance(result.is_violation, bool)
        assert isinstance(result.probability, float)
        assert 0.0 <= result.probability <= 1.0
    
    def test_predict_model_not_loaded_error(self):
        ModerationService.model = None
        
        request = PredictionRequest(
            seller_id=1,
            is_verified_seller=False,
            item_id=100,
            name="Test Item",
            description="Test Description",
            category=1,
            images_qty=0
        )
        
        with pytest.raises(ModelNotLoadedError, match="Модель не загружена"):
            ModerationService.predict(request)
    
    def test_predict_with_different_inputs(self):
        test_cases = [
            {
                "seller_id": 1,
                "is_verified_seller": False,
                "item_id": 100,
                "name": "Item 1",
                "description": "A" * 100,
                "category": 10,
                "images_qty": 3
            },
            {
                "seller_id": 2,
                "is_verified_seller": True,
                "item_id": 200,
                "name": "Item 2",
                "description": "B" * 500,
                "category": 50,
                "images_qty": 10
            },
            {
                "seller_id": 3,
                "is_verified_seller": False,
                "item_id": 300,
                "name": "Item 3",
                "description": "C" * 1000,
                "category": 99,
                "images_qty": 0
            }
        ]
        
        for case in test_cases:
            request = PredictionRequest(**case)
            result = ModerationService.predict(request)
            
            assert isinstance(result, PredictionResponse)
            assert isinstance(result.is_violation, bool)
            assert isinstance(result.probability, float)
            assert 0.0 <= result.probability <= 1.0


class TestPredictEndpoint:
    @pytest.fixture(autouse=True)
    def setup_model(self):
        from model import train_model
        ModerationService.model = train_model()
        yield
        ModerationService.model = None
    
    def test_predict_success_is_violation_true(self, app_client):
        response = app_client.post(
            "/predict/",
            json={
                "seller_id": 1,
                "is_verified_seller": False,
                "item_id": 100,
                "name": "Test Item",
                "description": "Short description",
                "category": 1,
                "images_qty": 0
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "is_violation" in data
        assert "probability" in data
        assert isinstance(data["is_violation"], bool)
        assert isinstance(data["probability"], float)
        assert 0.0 <= data["probability"] <= 1.0
    
    def test_predict_success_is_violation_false(self, app_client):
        response = app_client.post(
            "/predict/",
            json={
                "seller_id": 2,
                "is_verified_seller": True,
                "item_id": 101,
                "name": "Test Item 2",
                "description": "Long description with many words to make it longer and more detailed",
                "category": 2,
                "images_qty": 5
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "is_violation" in data
        assert "probability" in data
        assert isinstance(data["is_violation"], bool)
        assert isinstance(data["probability"], float)
        assert 0.0 <= data["probability"] <= 1.0
    
    def test_predict_validation_wrong_type_seller_id(self, app_client):
        response = app_client.post(
            "/predict/",
            json={
                "seller_id": "not_an_int",
                "is_verified_seller": True,
                "item_id": 100,
                "name": "Test Item",
                "description": "Test Description",
                "category": 1,
                "images_qty": 0
            }
        )
        
        assert response.status_code == 422
    
    def test_predict_validation_wrong_type_item_id(self, app_client):
        response = app_client.post(
            "/predict/",
            json={
                "seller_id": 1,
                "is_verified_seller": True,
                "item_id": "not_an_int",
                "name": "Test Item",
                "description": "Test Description",
                "category": 1,
                "images_qty": 0
            }
        )
        
        assert response.status_code == 422
    
    def test_predict_validation_wrong_type_is_verified_seller(self, app_client):
        response = app_client.post(
            "/predict/",
            json={
                "seller_id": 1,
                "is_verified_seller": "not_a_bool",
                "item_id": 100,
                "name": "Test Item",
                "description": "Test Description",
                "category": 1,
                "images_qty": 0
            }
        )
        
        assert response.status_code == 422
    
    def test_predict_validation_wrong_type_category(self, app_client):
        response = app_client.post(
            "/predict/",
            json={
                "seller_id": 1,
                "is_verified_seller": True,
                "item_id": 100,
                "name": "Test Item",
                "description": "Test Description",
                "category": "not_an_int",
                "images_qty": 0
            }
        )
        
        assert response.status_code == 422
    
    def test_predict_validation_wrong_type_images_qty(self, app_client):
        response = app_client.post(
            "/predict/",
            json={
                "seller_id": 1,
                "is_verified_seller": True,
                "item_id": 100,
                "name": "Test Item",
                "description": "Test Description",
                "category": 1,
                "images_qty": "not_an_int"
            }
        )
        
        assert response.status_code == 422
    
    def test_predict_validation_missing_field(self, app_client):
        response = app_client.post(
            "/predict/",
            json={
                "seller_id": 1,
                "is_verified_seller": True,
                "name": "Test Item",
                "description": "Test Description",
                "category": 1,
                "images_qty": 0
            }
        )
        
        assert response.status_code == 422
    
    def test_predict_validation_negative_images(self, app_client):
        response = app_client.post(
            "/predict/",
            json={
                "seller_id": 1,
                "is_verified_seller": True,
                "item_id": 100,
                "name": "Test Item",
                "description": "Test Description",
                "category": 1,
                "images_qty": -1
            }
        )
        
        assert response.status_code == 422
    
    def test_predict_validation_empty_name(self, app_client):
        response = app_client.post(
            "/predict/",
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
    
    def test_predict_model_not_loaded_error(self, app_client):
        ModerationService.model = None
        
        response = app_client.post(
            "/predict/",
            json={
                "seller_id": 1,
                "is_verified_seller": False,
                "item_id": 100,
                "name": "Test Item",
                "description": "Test Description",
                "category": 1,
                "images_qty": 0
            }
        )
        
        assert response.status_code == 503
        data = response.json()
        assert "detail" in data
        assert "Модель не загружена" in data["detail"]
    
    def test_predict_general_error_handling(self, app_client):
        with patch('services.moderation_service.ModerationService.predict') as mock_predict:
            mock_predict.side_effect = ValueError("Ошибка при обработке данных")
            
            response = app_client.post(
                "/predict/",
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
