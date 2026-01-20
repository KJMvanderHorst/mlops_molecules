"""
Integration tests for the molecule prediction API.
This file uses unittests.mock patch to to not rely on the actual model.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from project_name.api import app

client = TestClient(app)


class TestRoot:
    """Test health check endpoint."""

    def test_health_check_returns_healthy(self):
        """Test that health check returns status 200 with healthy status."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_health_check_response_structure(self):
        """Test health check response has correct structure."""
        response = client.get("/")
        data = response.json()
        assert "status" in data
        assert isinstance(data["status"], str)


class TestPredictEndpoint:
    """Test predict endpoint."""

    @staticmethod
    def get_valid_request():
        """Get valid prediction request."""
        return {
            "node_features": [[1.0] * 11 for _ in range(5)],
            "edge_index": [[0, 1, 2], [1, 2, 3]],
        }

    def test_predict_valid_request(self):
        """Test predict endpoint with valid input."""
        with patch("project_name.api.service") as mock_service:
            mock_service.predict.return_value = 2.5
            response = client.post("/predict", json=self.get_valid_request())
            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
            assert isinstance(data["prediction"], float)
            assert data["prediction"] == 2.5

    def test_predict_service_not_ready(self):
        """Test predict endpoint when service is None (not ready)."""
        with patch("project_name.api.service", None):
            response = client.post("/predict", json=self.get_valid_request())
            assert response.status_code == 503
            assert "not ready" in response.json()["detail"].lower()

    def test_predict_empty_node_features(self):
        """Test predict endpoint rejects empty node features."""
        request = {
            "node_features": [],
            "edge_index": [[0], [1]],
        }
        response = client.post("/predict", json=request)
        assert response.status_code == 422

    def test_predict_invalid_feature_count(self):
        """Test predict endpoint rejects wrong feature count."""
        request = {
            "node_features": [[1.0] * 10],
            "edge_index": [[0], [1]],
        }
        response = client.post("/predict", json=request)
        assert response.status_code == 422

    def test_predict_inconsistent_feature_count(self):
        """Test predict endpoint rejects inconsistent feature counts across nodes."""
        request = {
            "node_features": [[1.0] * 11, [1.0] * 10],
            "edge_index": [[0], [1]],
        }
        response = client.post("/predict", json=request)
        assert response.status_code == 422

    def test_predict_calls_service(self):
        """Test that predict endpoint calls the inference service."""
        with patch("project_name.api.service") as mock_service:
            mock_service.predict.return_value = 1.23
            client.post("/predict", json=self.get_valid_request())
            mock_service.predict.assert_called_once()

    def test_predict_passes_correct_data_to_service(self):
        """Test that predict passes correct data to service."""
        with patch("project_name.api.service") as mock_service:
            mock_service.predict.return_value = 1.0
            request = self.get_valid_request()
            client.post("/predict", json=request)
            mock_service.predict.assert_called_once_with(
                request["node_features"],
                request["edge_index"],
            )

    def test_predict_service_exception_handling(self):
        """Test that service exceptions are caught and return 500."""
        with patch("project_name.api.service") as mock_service:
            mock_service.predict.side_effect = RuntimeError("Model inference failed")
            response = client.post("/predict", json=self.get_valid_request())
            assert response.status_code == 500
            assert "Model inference failed" in response.json()["detail"]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @staticmethod
    def get_valid_request():
        """Get valid prediction request."""
        return {
            "node_features": [[1.0] * 11 for _ in range(5)],
            "edge_index": [[0, 1, 2], [1, 2, 3]],
        }

    def test_predict_with_large_graph(self):
        """Test predict with larger graph."""
        large_request = {
            "node_features": [[float(i % 11)] * 11 for i in range(100)],
            "edge_index": [[i, i + 1] for i in range(99)] + [[i + 1, i] for i in range(99)],
        }
        with patch("project_name.api.service") as mock_service:
            mock_service.predict.return_value = 3.14
            response = client.post("/predict", json=large_request)
            assert response.status_code == 200

    def test_predict_with_negative_values(self):
        """Test predict accepts negative feature values."""
        request = {
            "node_features": [[-1.0] * 11 for _ in range(3)],
            "edge_index": [[0, 1], [1, 2]],
        }
        with patch("project_name.api.service") as mock_service:
            mock_service.predict.return_value = -0.5
            response = client.post("/predict", json=request)
            assert response.status_code == 200

    def test_predict_with_zero_values(self):
        """Test predict accepts zero feature values."""
        request = {
            "node_features": [[0.0] * 11 for _ in range(3)],
            "edge_index": [[0, 1], [1, 2]],
        }
        with patch("project_name.api.service") as mock_service:
            mock_service.predict.return_value = 0.0
            response = client.post("/predict", json=request)
            assert response.status_code == 200

    def test_predict_multiple_requests_consistency(self):
        """Test that multiple requests with same input return consistent predictions."""
        request = self.get_valid_request()
        with patch("project_name.api.service") as mock_service:
            mock_service.predict.return_value = 1.5
            response1 = client.post("/predict", json=request)
            response2 = client.post("/predict", json=request)
            assert response1.status_code == 200
            assert response2.status_code == 200
            assert response1.json()["prediction"] == response2.json()["prediction"]
