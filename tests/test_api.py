import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import tempfile
import os

from aureon.api.main import app
from aureon.api.models import PredictionRequest, TrainingRequest, RetrainRequest
from aureon.models.model_registry import ModelRegistry
from aureon.models.classification_models import RandomForestModel

client = TestClient(app)


class TestAPIEndpoints:
    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "AUREON API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"

    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "models_loaded" in data
        assert "active_models" in data
        assert "system_info" in data

    def test_list_models_endpoint(self):
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "total_count" in data
        assert "timestamp" in data
        assert isinstance(data["models"], list)

    def test_get_model_info_endpoint(self):
        with patch("aureon.api.main.registry") as mock_registry:
            mock_model = Mock()
            mock_model.get_model_info.return_value = {
                "model_name": "TestModel",
                "task_type": "classification",
                "is_trained": True,
                "model_metrics": {"accuracy": 0.95},
            }
            mock_registry.get_model.return_value = mock_model

            response = client.get("/api/v1/models/1")
            assert response.status_code == 200
            data = response.json()
            assert "model_id" in data
            assert "model_info" in data
            assert data["model_id"] == 1

    def test_get_nonexistent_model(self):
        with patch("aureon.api.main.registry") as mock_registry:
            mock_registry.get_model.side_effect = ValueError("Model not found")

            response = client.get("/api/v1/models/999")
            assert response.status_code == 404

    def test_delete_model_endpoint(self):
        with patch("aureon.api.main.registry") as mock_registry:
            mock_registry.delete_model.return_value = None

            response = client.delete("/api/v1/models/1")
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            assert "timestamp" in data

    def test_system_status_endpoint(self):
        response = client.get("/api/v1/system/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "alerts" in data
        assert "metrics" in data

    def test_system_metrics_endpoint(self):
        response = client.get("/api/v1/system/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data


class TestPredictionEndpoints:
    def test_predict_endpoint(self):
        with patch("aureon.api.predict.registry") as mock_registry:
            mock_model = Mock()
            mock_model.is_trained = True
            mock_model.model_name = "TestModel"
            mock_model.predict.return_value = np.array([0, 1, 0])
            mock_registry.get_model.return_value = mock_model

            request_data = {
                "data": [
                    {"feature1": 1, "feature2": 2},
                    {"feature1": 3, "feature2": 4},
                    {"feature1": 5, "feature2": 6},
                ],
                "model_id": 1,
                "return_probabilities": False,
            }

            response = client.post("/api/v1/predict", json=request_data)
            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data
            assert "model_used" in data
            assert "model_id" in data
            assert "prediction_time" in data
            assert len(data["predictions"]) == 3

    def test_predict_with_probabilities(self):
        with patch("aureon.api.predict.registry") as mock_registry:
            mock_model = Mock()
            mock_model.is_trained = True
            mock_model.model_name = "TestModel"
            mock_model.predict.return_value = np.array([0, 1])
            mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
            mock_registry.get_model.return_value = mock_model

            request_data = {
                "data": [
                    {"feature1": 1, "feature2": 2},
                    {"feature1": 3, "feature2": 4},
                ],
                "model_id": 1,
                "return_probabilities": True,
            }

            response = client.post("/api/v1/predict", json=request_data)
            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data
            assert "probabilities" in data
            assert len(data["probabilities"]) == 2

    def test_predict_with_model_name(self):
        with patch("aureon.api.predict.registry") as mock_registry:
            mock_model = Mock()
            mock_model.is_trained = True
            mock_model.model_name = "TestModel"
            mock_model.predict.return_value = np.array([0, 1])
            mock_registry.get_latest_model.return_value = mock_model

            request_data = {
                "data": [
                    {"feature1": 1, "feature2": 2},
                    {"feature1": 3, "feature2": 4},
                ],
                "model_name": "TestModel",
                "return_probabilities": False,
            }

            response = client.post("/api/v1/predict", json=request_data)
            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data
            assert "model_used" == "TestModel"

    def test_predict_no_data(self):
        request_data = {"data": [], "model_id": 1}

        response = client.post("/api/v1/predict", json=request_data)
        assert response.status_code == 400

    def test_predict_no_model_specified(self):
        request_data = {"data": [{"feature1": 1, "feature2": 2}]}

        response = client.post("/api/v1/predict", json=request_data)
        assert response.status_code == 400

    def test_predict_untrained_model(self):
        with patch("aureon.api.predict.registry") as mock_registry:
            mock_model = Mock()
            mock_model.is_trained = False
            mock_registry.get_model.return_value = mock_model

            request_data = {"data": [{"feature1": 1, "feature2": 2}], "model_id": 1}

            response = client.post("/api/v1/predict", json=request_data)
            assert response.status_code == 400

    def test_predict_batch_endpoint(self):
        with patch("aureon.api.predict.registry") as mock_registry:
            mock_model = Mock()
            mock_model.is_trained = True
            mock_model.model_name = "TestModel"
            mock_model.predict.return_value = np.array([0])
            mock_registry.get_model.return_value = mock_model

            request_data = [
                {"data": [{"feature1": 1, "feature2": 2}], "model_id": 1},
                {"data": [{"feature1": 3, "feature2": 4}], "model_id": 1},
            ]

            response = client.post("/api/v1/predict/batch", json=request_data)
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 2

    def test_predict_with_preprocessing(self):
        with patch("aureon.api.predict.registry") as mock_registry, patch(
            "aureon.api.predict.DataPipeline"
        ) as mock_pipeline:

            mock_model = Mock()
            mock_model.is_trained = True
            mock_model.model_name = "TestModel"
            mock_model.predict.return_value = np.array([0, 1])
            mock_registry.get_model.return_value = mock_model

            mock_pipeline_instance = Mock()
            mock_pipeline_instance.transform_new_data.return_value = pd.DataFrame(
                {"feature1": [1, 3], "feature2": [2, 4]}
            )
            mock_pipeline.return_value = mock_pipeline_instance

            request_data = {
                "data": [
                    {"feature1": 1, "feature2": 2},
                    {"feature1": 3, "feature2": 4},
                ],
                "model_id": 1,
            }

            response = client.post(
                "/api/v1/predict/with_preprocessing?pipeline_id=test", json=request_data
            )
            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data
            assert "preprocessing_applied" in data
            assert data["preprocessing_applied"] is True

    def test_prediction_health_endpoint(self):
        with patch("aureon.api.predict.registry") as mock_registry:
            mock_registry.get_model_statistics.return_value = {"active_models": 5}

            response = client.get("/api/v1/predict/health")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "active_models" in data
            assert "timestamp" in data


class TestTrainingEndpoints:
    def test_train_endpoint(self):
        with patch("aureon.api.training.DataPipeline") as mock_data_pipeline, patch(
            "aureon.api.training.ModelPipeline"
        ) as mock_model_pipeline, patch("aureon.api.training.Path") as mock_path:

            mock_path.return_value.exists.return_value = True

            mock_data_pipeline_instance = Mock()
            mock_data_pipeline_instance.run_pipeline.return_value = {
                "splits": (pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series())
            }
            mock_data_pipeline.return_value = mock_data_pipeline_instance

            mock_model_pipeline_instance = Mock()
            mock_model_pipeline_instance.train_models.return_value = {
                "trained_models": ["random_forest"],
                "best_model": "random_forest",
                "best_score": 0.95,
            }
            mock_model_pipeline_instance.register_all_models.return_value = [1, 2]
            mock_model_pipeline.return_value = mock_model_pipeline_instance

            request_data = {
                "data_source": "test_data.csv",
                "target_column": "target",
                "task_type": "classification",
                "model_types": ["random_forest"],
                "experiment_name": "test_experiment",
            }

            response = client.post("/api/v1/train", json=request_data)
            assert response.status_code == 200
            data = response.json()
            assert "experiment_id" in data
            assert "models_trained" in data
            assert "status" in data

    def test_train_nonexistent_file(self):
        with patch("aureon.api.training.Path") as mock_path:
            mock_path.return_value.exists.return_value = False

            request_data = {
                "data_source": "nonexistent.csv",
                "target_column": "target",
                "task_type": "classification",
            }

            response = client.post("/api/v1/train", json=request_data)
            assert response.status_code == 400

    def test_get_training_status(self):
        with patch("aureon.api.training.training_tasks") as mock_tasks:
            mock_tasks.__contains__.return_value = True
            mock_tasks.__getitem__.return_value = {
                "status": "completed",
                "started_at": "2023-01-01T00:00:00",
                "completed_at": "2023-01-01T01:00:00",
                "results": {"best_model": "random_forest"},
            }

            response = client.get("/api/v1/train/status/test_experiment")
            assert response.status_code == 200
            data = response.json()
            assert "experiment_id" in data
            assert "status" in data
            assert data["status"] == "completed"

    def test_get_nonexistent_training_status(self):
        with patch("aureon.api.training.training_tasks") as mock_tasks:
            mock_tasks.__contains__.return_value = False

            response = client.get("/api/v1/train/status/nonexistent")
            assert response.status_code == 404

    def test_retrain_endpoint(self):
        with patch("aureon.api.training.registry") as mock_registry, patch(
            "aureon.api.training.DataPipeline"
        ) as mock_data_pipeline, patch(
            "aureon.api.training.ModelPipeline"
        ) as mock_model_pipeline, patch(
            "aureon.api.training.Path"
        ) as mock_path:

            mock_path.return_value.exists.return_value = True

            mock_model = Mock()
            mock_registry.get_model.return_value = mock_model

            mock_data_pipeline_instance = Mock()
            mock_data_pipeline_instance.run_pipeline.return_value = {
                "splits": (pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series())
            }
            mock_data_pipeline.return_value = mock_data_pipeline_instance

            mock_model_pipeline_instance = Mock()
            mock_model_pipeline_instance.retrain_model.return_value = {
                "new_model_id": 2,
                "evaluation_results": {"accuracy": 0.95},
            }
            mock_model_pipeline.return_value = mock_model_pipeline_instance

            request_data = {
                "model_id": 1,
                "data_source": "new_data.csv",
                "target_column": "target",
            }

            response = client.post("/api/v1/retrain", json=request_data)
            assert response.status_code == 200
            data = response.json()
            assert "new_model_id" in data
            assert "original_model_id" in data
            assert "status" in data

    def test_train_sync_endpoint(self):
        with patch("aureon.api.training.DataPipeline") as mock_data_pipeline, patch(
            "aureon.api.training.ModelPipeline"
        ) as mock_model_pipeline, patch("aureon.api.training.Path") as mock_path:

            mock_path.return_value.exists.return_value = True

            mock_data_pipeline_instance = Mock()
            mock_data_pipeline_instance.run_pipeline.return_value = {
                "splits": (pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series())
            }
            mock_data_pipeline.return_value = mock_data_pipeline_instance

            mock_model_pipeline_instance = Mock()
            mock_model_pipeline_instance.train_models.return_value = {
                "trained_models": ["random_forest"],
                "best_model": "random_forest",
                "best_score": 0.95,
            }
            mock_model_pipeline_instance.register_all_models.return_value = [1, 2]
            mock_model_pipeline_instance.experiment_name = "sync_experiment"
            mock_model_pipeline.return_value = mock_model_pipeline_instance

            request_data = {
                "data_source": "test_data.csv",
                "target_column": "target",
                "task_type": "classification",
                "model_types": ["random_forest"],
            }

            response = client.post("/api/v1/train/sync", json=request_data)
            assert response.status_code == 200
            data = response.json()
            assert "experiment_id" in data
            assert "status" == "completed"

    def test_get_training_history(self):
        with patch("aureon.api.training.registry") as mock_registry:
            mock_registry.list_models.return_value = [
                {
                    "id": 1,
                    "model_name": "TestModel",
                    "task_type": "classification",
                    "created_at": "2023-01-01T00:00:00",
                    "metrics": {"accuracy": 0.95},
                    "description": "Test model",
                }
            ]

            response = client.get("/api/v1/train/history")
            assert response.status_code == 200
            data = response.json()
            assert "training_history" in data
            assert "total_models" in data
            assert len(data["training_history"]) == 1

    def test_delete_experiment(self):
        with patch("aureon.api.training.training_tasks") as mock_tasks:
            mock_tasks.__contains__.return_value = True
            mock_tasks.__delitem__ = Mock()

            response = client.delete("/api/v1/train/experiment/test_experiment")
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            assert "timestamp" in data

    def test_delete_nonexistent_experiment(self):
        with patch("aureon.api.training.training_tasks") as mock_tasks:
            mock_tasks.__contains__.return_value = False

            response = client.delete("/api/v1/train/experiment/nonexistent")
            assert response.status_code == 404
