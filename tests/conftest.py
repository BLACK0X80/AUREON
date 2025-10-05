import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            "feature3": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "category": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
            "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        }
    )


@pytest.fixture
def sample_regression_data():
    return pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            "feature3": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "target": [3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
        }
    )


@pytest.fixture
def sample_data_with_missing():
    return pd.DataFrame(
        {
            "feature1": [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10],
            "feature2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            "feature3": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "category": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
            "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        }
    )


@pytest.fixture
def sample_data_with_duplicates():
    return pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "feature2": [2, 4, 6, 8, 10, 2, 4, 6, 8, 10],
            "feature3": [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5],
            "category": ["A", "B", "A", "C", "B", "A", "B", "A", "C", "B"],
            "target": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }
    )


@pytest.fixture
def temp_csv_file(sample_data):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_data.to_csv(f.name, index=False)
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_parquet_file(sample_data):
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        sample_data.to_parquet(f.name, index=False)
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_directory():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir

    import shutil

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_model():
    from unittest.mock import Mock

    model = Mock()
    model.model_name = "TestModel"
    model.task_type = "classification"
    model.is_trained = True
    model.model_params = {"param1": "value1"}
    model.model_metrics = {"accuracy": 0.95, "f1_score": 0.92}
    model.feature_importance = pd.Series([0.6, 0.4], index=["feature1", "feature2"])

    def mock_predict(X):
        return np.array([0, 1, 0, 1, 0])

    def mock_predict_proba(X):
        return np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])

    def mock_get_model_info():
        return {
            "model_name": "TestModel",
            "task_type": "classification",
            "is_trained": True,
            "model_params": {"param1": "value1"},
            "model_metrics": {"accuracy": 0.95, "f1_score": 0.92},
            "feature_importance": {"feature1": 0.6, "feature2": 0.4},
        }

    model.predict = mock_predict
    model.predict_proba = mock_predict_proba
    model.get_model_info = mock_get_model_info

    return model


@pytest.fixture
def mock_registry():
    from unittest.mock import Mock

    registry = Mock()

    def mock_list_models(task_type=None, model_name=None):
        return [
            {
                "id": 1,
                "model_name": "TestModel",
                "model_type": "RandomForest",
                "version": "1.0",
                "task_type": "classification",
                "created_at": "2023-01-01T00:00:00",
                "description": "Test model",
                "metrics": {"accuracy": 0.95, "f1_score": 0.92},
            }
        ]

    def mock_get_model(model_id):
        if model_id == 1:
            return mock_model()
        else:
            raise ValueError(f"Model with ID {model_id} not found")

    def mock_get_model_statistics():
        return {
            "total_models": 1,
            "active_models": 1,
            "inactive_models": 0,
            "models_by_task_type": {"classification": 1},
        }

    registry.list_models = mock_list_models
    registry.get_model = mock_get_model
    registry.get_model_statistics = mock_get_model_statistics

    return registry


@pytest.fixture
def mock_system_metrics():
    return {
        "timestamp": "2023-01-01T00:00:00",
        "cpu_percent": 25.5,
        "memory_percent": 60.2,
        "memory_available_gb": 8.5,
        "disk_percent": 45.0,
        "disk_free_gb": 100.0,
        "platform": "Windows",
        "platform_version": "10.0.22000",
        "python_version": "3.11.0",
        "cpu_count": 8,
    }


@pytest.fixture
def sample_prediction_request():
    return {
        "data": [
            {"feature1": 1, "feature2": 2, "feature3": 0.1},
            {"feature1": 3, "feature2": 4, "feature3": 0.3},
            {"feature1": 5, "feature2": 6, "feature3": 0.5},
        ],
        "model_id": 1,
        "return_probabilities": False,
    }


@pytest.fixture
def sample_training_request():
    return {
        "data_source": "test_data.csv",
        "target_column": "target",
        "task_type": "classification",
        "model_types": ["random_forest", "logistic_regression"],
        "experiment_name": "test_experiment",
        "hyperparameter_search": {"enabled": False},
    }


@pytest.fixture
def sample_retrain_request():
    return {"model_id": 1, "data_source": "new_data.csv", "target_column": "target"}


@pytest.fixture(scope="session")
def test_config():
    return {
        "database": {"url": "sqlite:///./test_aureon.db", "echo": False},
        "models": {
            "registry_path": "test_models_registry",
            "save_path": "test_trained_models",
            "default_model_type": "random_forest",
        },
        "api": {"host": "0.0.0.0", "port": 8001, "reload": False, "workers": 1},
        "logging": {"level": "DEBUG", "file": "test_logs/aureon.log", "format": "json"},
        "monitoring": {"drift_threshold": 0.1, "performance_threshold": 0.7},
        "reporting": {"output_path": "test_reports", "default_format": "pdf"},
        "data_pipeline": {
            "missing_values_strategy": "mean",
            "categorical_encoding_strategy": "one_hot",
            "feature_scaling_strategy": "standard",
            "test_size": 0.2,
            "random_state": 42,
        },
        "scheduler": {"enabled": False, "interval_minutes": 60},
    }


@pytest.fixture(autouse=True)
def setup_test_environment(test_config, temp_directory):
    import os
    from unittest.mock import patch

    with patch("aureon.config.settings.config") as mock_config:
        for key, value in test_config.items():
            setattr(mock_config, key, value)

        os.environ["AUREON_DATABASE_URL"] = f"sqlite:///{temp_directory}/test.db"
        os.environ["AUREON_MODELS_REGISTRY_PATH"] = f"{temp_directory}/models"
        os.environ["AUREON_LOGGING_LEVEL"] = "DEBUG"

        yield mock_config


@pytest.fixture
def sample_explanation_data():
    return {
        "explanation_type": "shap",
        "sample_index": 0,
        "feature_importance": {"feature1": 0.3, "feature2": 0.7},
        "base_value": 0.5,
        "prediction": 1,
        "timestamp": "2023-01-01T00:00:00",
    }


@pytest.fixture
def sample_drift_results():
    return {
        "drift_detected": True,
        "drift_score": 0.15,
        "feature_drift": {"feature1": 0.12, "feature2": 0.15, "feature3": 0.08},
        "timestamp": "2023-01-01T00:00:00",
        "threshold": 0.1,
    }


@pytest.fixture
def sample_performance_metrics():
    return {
        "model_name": "TestModel",
        "performance_score": 0.95,
        "timestamp": "2023-01-01T00:00:00",
        "below_threshold": False,
    }
