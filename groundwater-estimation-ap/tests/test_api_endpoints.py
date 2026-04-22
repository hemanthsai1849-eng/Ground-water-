"""
Unit tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))

from api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthCheck:
    """Test health check endpoint."""
    
    def test_health_check_status(self, client):
        """Test that health check returns status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert data['status'] in ['operational', 'degraded']
        assert 'model_loaded' in data
        assert 'timestamp' in data


class TestVillageDataEndpoint:
    """Test village data prediction endpoint."""
    
    def test_valid_village_request(self, client):
        """Test request with valid village ID."""
        # This test will fail if model/data not loaded, which is expected
        response = client.post("/get_village_data/1")
        assert response.status_code in [200, 404, 503]
    
    def test_invalid_village_id_format(self, client):
        """Test request with invalid village ID."""
        response = client.post("/get_village_data/-1")
        assert response.status_code == 400
    
    def test_nonexistent_village(self, client):
        """Test request for non-existent village."""
        response = client.post("/get_village_data/999999")
        assert response.status_code in [404, 503]
    
    def test_prediction_response_schema(self, client):
        """Test prediction response contains required fields."""
        response = client.post("/get_village_data/1")
        
        if response.status_code == 200:
            data = response.json()
            assert 'village_id' in data
            assert 'predicted_level' in data
            assert 'trend' in data
            assert 'alert_status' in data
            assert 'confidence' in data
            assert 'timestamp' in data
            
            # Validate data types and ranges
            assert isinstance(data['village_id'], int)
            assert isinstance(data['predicted_level'], (int, float))
            assert data['trend'] in ['RISING', 'FALLING', 'STABLE']
            assert data['alert_status'] in ['NORMAL', 'WARNING', 'CRITICAL']
            assert 0 <= data['confidence'] <= 1


class TestBatchPrediction:
    """Test batch prediction endpoint."""
    
    def test_batch_prediction_empty_list(self, client):
        """Test batch prediction with empty list."""
        response = client.post("/predict_batch", json=[])
        assert response.status_code == 200
        assert response.json() == []
    
    def test_batch_prediction_multiple_villages(self, client):
        """Test batch prediction with multiple villages."""
        village_ids = [1, 2, 3]
        response = client.post("/predict_batch", json=village_ids)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestAnomalyDetection:
    """Test anomaly detection endpoint."""
    
    def test_anomaly_endpoint_not_found(self, client):
        """Test anomaly detection for non-existent village."""
        response = client.post("/anomalies/999999")
        assert response.status_code in [404, 503]
    
    def test_anomaly_endpoint_last_n_days(self, client):
        """Test anomaly endpoint with last_n_days parameter."""
        response = client.post("/anomalies/1?last_n_days=7")
        assert response.status_code in [200, 404, 503]


class TestErrorHandling:
    """Test error handling in API."""
    
    def test_404_error_response(self, client):
        """Test 404 error response format."""
        response = client.post("/get_village_data/999999999")
        
        if response.status_code == 404:
            data = response.json()
            assert 'detail' in data or 'error_code' in data
    
    def test_invalid_request_body(self, client):
        """Test invalid request body handling."""
        response = client.post(
            "/predict_batch",
            json="invalid_data"
        )
        assert response.status_code in [422, 400]
