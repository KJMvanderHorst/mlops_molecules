"""
Load testing for molecule prediction API using Locust.

Run with:
    locust -f tests/performance_tests/load_test_api.py --host=https://gcp-app-1064661101575.europe-west1.run.app

Or use command line:
    locust -f tests/performance_tests/load_test_api.py --host=https://your-api-url.run.app --users=100 --spawn-rate=10 --run-time=5m
"""

import random

from locust import HttpUser, between, task


class MoleculeAPIPredictionUser(HttpUser):
    """Simulates a user making requests to the molecule prediction API."""

    # Wait between 2-5 seconds between requests
    wait_time = between(2, 5)

    def get_random_valid_request(self):
        """Generate a random but valid prediction request."""
        num_nodes = random.randint(3, 20)
        # Create valid edge index: PyTorch Geometric format [sources, targets]
        num_edges = num_nodes - 1
        sources = list(range(num_edges))
        targets = list(range(1, num_edges + 1))
        edge_index = [sources, targets]
        return {
            "node_features": [[float(i % 11)] * 11 for i in range(num_nodes)],
            "edge_index": edge_index,
        }

    @task(1)
    def health_check(self):
        """Task: Check API health (lightweight, happens 1x for every 5 predictions)."""
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @task(5)
    def make_prediction(self):
        """Task: Make a prediction request (heavyweight, happens 5x more than health checks)."""
        request_data = self.get_random_valid_request()

        with self.client.post(
            "/predict",
            json=request_data,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "prediction" in data and isinstance(data["prediction"], (int, float)):
                        response.success()
                    else:
                        response.failure(f"Invalid response format: {data}")
                except Exception as e:
                    response.failure(f"Failed to parse response: {e}")
            else:
                response.failure(f"Prediction failed: {response.status_code}")

    @task(2)
    def make_prediction_edge_case(self):
        """Task: Test edge cases like small graphs and zero values."""
        # Small graph (2-5 nodes) with correct edge index format
        num_nodes = random.randint(2, 5)
        num_edges = num_nodes - 1
        small_graph = {
            "node_features": [[0.0] * 11 for _ in range(num_nodes)],
            "edge_index": [list(range(num_edges)), list(range(1, num_edges + 1))],
        }

        with self.client.post(
            "/predict",
            json=small_graph,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Edge case failed: {response.status_code}")

    @task(1)
    def test_invalid_feature_count(self):
        """Task: Test that API rejects wrong feature count (422)."""
        invalid_request = {
            "node_features": [[1.0] * 10],  # Should be 11, not 10
            "edge_index": [[0], [0]],
        }

        with self.client.post(
            "/predict",
            json=invalid_request,
            catch_response=True,
        ) as response:
            if response.status_code == 422:
                response.success()  # ✅ Correct: validation error
            else:
                response.failure(f"Expected 422, got {response.status_code}")

    @task(1)
    def test_empty_node_features(self):
        """Task: Test that API rejects empty node features (422)."""
        invalid_request = {
            "node_features": [],  # Empty
            "edge_index": [[0], [0]],
        }

        with self.client.post(
            "/predict",
            json=invalid_request,
            catch_response=True,
        ) as response:
            if response.status_code == 422:
                response.success()  # ✅ Correct: validation error
            else:
                response.failure(f"Expected 422, got {response.status_code}")

    @task(1)
    def test_inconsistent_features(self):
        """Task: Test that API rejects inconsistent feature counts (422)."""
        invalid_request = {
            "node_features": [[1.0] * 11, [1.0] * 10],  # Inconsistent
            "edge_index": [[0], [1]],
        }

        with self.client.post(
            "/predict",
            json=invalid_request,
            catch_response=True,
        ) as response:
            if response.status_code == 422:
                response.success()  # ✅ Correct: validation error
            else:
                response.failure(f"Expected 422, got {response.status_code}")


class LightweightUser(HttpUser):
    """A lightweight user that primarily checks health."""

    wait_time = between(5, 10)

    @task
    def health_check(self):
        """Frequent health checks."""
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")


class HeavyweightUser(HttpUser):
    """A heavyweight user that makes frequent predictions."""

    wait_time = between(1, 3)

    def get_random_valid_request(self):
        """Generate a random but valid prediction request."""
        num_nodes = random.randint(10, 50)
        # Create valid edge index: PyTorch Geometric format [sources, targets]
        num_edges = num_nodes - 1
        sources = list(range(num_edges))
        targets = list(range(1, num_edges + 1))
        edge_index = [sources, targets]
        return {
            "node_features": [[float(i % 11)] * 11 for i in range(num_nodes)],
            "edge_index": edge_index,
        }

    @task
    def make_prediction(self):
        """Constantly make predictions."""
        request_data = self.get_random_valid_request()

        with self.client.post(
            "/predict",
            json=request_data,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Prediction failed: {response.status_code}")
