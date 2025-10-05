import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from aureon.models.base_model import BaseModel
from aureon.models.classification_models import (
    RandomForestModel,
    GradientBoostingModel,
    LogisticRegressionModel,
    ClassificationModelFactory,
)
from aureon.models.regression_models import (
    RandomForestRegressorModel,
    LinearRegressionModel,
    RidgeModel,
    RegressionModelFactory,
)
from aureon.models.model_trainer import ModelTrainer
from aureon.models.model_registry import ModelRegistry, ModelMetadata


class TestBaseModel:
    def test_model_initialization(self):
        class TestModel(BaseModel):
            def _create_model(self):
                from sklearn.linear_model import LinearRegression

                return LinearRegression()

        model = TestModel("TestModel", {"param1": "value1"})

        assert model.model_name == "TestModel"
        assert model.model_params == {"param1": "value1"}
        assert not model.is_trained
        assert model.model_metrics == {}
        assert model.feature_importance is None
        assert model.task_type is None

    def test_fit_and_predict(self):
        class TestModel(BaseModel):
            def _create_model(self):
                from sklearn.linear_model import LinearRegression

                return LinearRegression()

        model = TestModel("TestModel")

        X = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [2, 4, 6, 8, 10]})
        y = pd.Series([3, 6, 9, 12, 15])

        model.fit(X, y)

        assert model.is_trained
        predictions = model.predict(X)
        assert len(predictions) == 5
        assert isinstance(predictions, np.ndarray)

    def test_evaluate_classification(self):
        class TestClassifier(BaseModel):
            def _create_model(self):
                from sklearn.linear_model import LogisticRegression

                return LogisticRegression()

        model = TestClassifier("TestClassifier")

        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            }
        )
        y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        model.fit(X, y)
        metrics = model.evaluate(X, y, "classification")

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert all(isinstance(v, float) for v in metrics.values())

    def test_evaluate_regression(self):
        class TestRegressor(BaseModel):
            def _create_model(self):
                from sklearn.linear_model import LinearRegression

                return LinearRegression()

        model = TestRegressor("TestRegressor")

        X = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [2, 4, 6, 8, 10]})
        y = pd.Series([3, 6, 9, 12, 15])

        model.fit(X, y)
        metrics = model.evaluate(X, y, "regression")

        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2_score" in metrics
        assert all(isinstance(v, float) for v in metrics.values())

    def test_save_and_load_model(self):
        class TestModel(BaseModel):
            def _create_model(self):
                from sklearn.linear_model import LinearRegression

                return LinearRegression()

        model = TestModel("TestModel")

        X = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [2, 4, 6, 8, 10]})
        y = pd.Series([3, 6, 9, 12, 15])

        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            temp_path = f.name

        try:
            model.save_model(temp_path)

            new_model = TestModel("NewModel")
            new_model.load_model(temp_path)

            assert new_model.is_trained
            predictions1 = model.predict(X)
            predictions2 = new_model.predict(X)
            np.testing.assert_array_almost_equal(predictions1, predictions2)

        finally:
            os.unlink(temp_path)


class TestClassificationModels:
    def test_random_forest_model(self):
        model = RandomForestModel({"n_estimators": 10, "random_state": 42})

        assert model.model_name == "RandomForest"
        assert model.task_type == "classification"

        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            }
        )
        y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)

    def test_gradient_boosting_model(self):
        model = GradientBoostingModel({"n_estimators": 10, "random_state": 42})

        assert model.model_name == "GradientBoosting"
        assert model.task_type == "classification"

        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            }
        )
        y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)

    def test_logistic_regression_model(self):
        model = LogisticRegressionModel({"random_state": 42})

        assert model.model_name == "LogisticRegression"
        assert model.task_type == "classification"

        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            }
        )
        y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)

    def test_classification_model_factory(self):
        available_models = ClassificationModelFactory.get_available_models()

        assert "random_forest" in available_models
        assert "gradient_boosting" in available_models
        assert "logistic_regression" in available_models

        model = ClassificationModelFactory.create_model(
            "random_forest", {"n_estimators": 5}
        )
        assert isinstance(model, RandomForestModel)

        with pytest.raises(ValueError):
            ClassificationModelFactory.create_model("unknown_model")


class TestRegressionModels:
    def test_random_forest_regressor_model(self):
        model = RandomForestRegressorModel({"n_estimators": 10, "random_state": 42})

        assert model.model_name == "RandomForestRegressor"
        assert model.task_type == "regression"

        X = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [2, 4, 6, 8, 10]})
        y = pd.Series([3, 6, 9, 12, 15])

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == 5
        assert all(isinstance(pred, (int, float)) for pred in predictions)

    def test_linear_regression_model(self):
        model = LinearRegressionModel()

        assert model.model_name == "LinearRegression"
        assert model.task_type == "regression"

        X = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [2, 4, 6, 8, 10]})
        y = pd.Series([3, 6, 9, 12, 15])

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == 5
        assert all(isinstance(pred, (int, float)) for pred in predictions)

    def test_ridge_model(self):
        model = RidgeModel({"alpha": 1.0})

        assert model.model_name == "Ridge"
        assert model.task_type == "regression"

        X = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [2, 4, 6, 8, 10]})
        y = pd.Series([3, 6, 9, 12, 15])

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == 5
        assert all(isinstance(pred, (int, float)) for pred in predictions)

    def test_regression_model_factory(self):
        available_models = RegressionModelFactory.get_available_models()

        assert "random_forest" in available_models
        assert "linear_regression" in available_models
        assert "ridge" in available_models

        model = RegressionModelFactory.create_model("linear_regression")
        assert isinstance(model, LinearRegressionModel)

        with pytest.raises(ValueError):
            RegressionModelFactory.create_model("unknown_model")


class TestModelTrainer:
    def test_train_single_model_classification(self):
        trainer = ModelTrainer("classification")

        X_train = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
                "feature2": [2, 4, 6, 8, 10, 12, 14, 16],
            }
        )
        y_train = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])

        model = trainer.train_single_model("random_forest", X_train, y_train)

        assert model.is_trained
        assert "random_forest" in trainer.models
        assert trainer.models["random_forest"] == model

    def test_train_single_model_regression(self):
        trainer = ModelTrainer("regression")

        X_train = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "feature2": [2, 4, 6, 8, 10]}
        )
        y_train = pd.Series([3, 6, 9, 12, 15])

        model = trainer.train_single_model("linear_regression", X_train, y_train)

        assert model.is_trained
        assert "linear_regression" in trainer.models
        assert trainer.models["linear_regression"] == model

    def test_train_multiple_models(self):
        trainer = ModelTrainer("classification")

        X_train = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
                "feature2": [2, 4, 6, 8, 10, 12, 14, 16],
            }
        )
        y_train = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])

        model_types = ["random_forest", "logistic_regression"]
        trained_models = trainer.train_multiple_models(model_types, X_train, y_train)

        assert len(trained_models) == 2
        assert "random_forest" in trained_models
        assert "logistic_regression" in trained_models
        assert all(model.is_trained for model in trained_models.values())

    def test_evaluate_models(self):
        trainer = ModelTrainer("classification")

        X_train = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
                "feature2": [2, 4, 6, 8, 10, 12, 14, 16],
            }
        )
        y_train = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])

        X_test = pd.DataFrame({"feature1": [9, 10], "feature2": [18, 20]})
        y_test = pd.Series([1, 1])

        trainer.train_single_model("random_forest", X_train, y_train)
        evaluation_results = trainer.evaluate_models(X_test, y_test)

        assert "random_forest" in evaluation_results
        assert "accuracy" in evaluation_results["random_forest"]
        assert "f1_score" in evaluation_results["random_forest"]

    def test_compare_models(self):
        trainer = ModelTrainer("classification")

        X_train = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
                "feature2": [2, 4, 6, 8, 10, 12, 14, 16],
            }
        )
        y_train = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])

        X_test = pd.DataFrame({"feature1": [9, 10], "feature2": [18, 20]})
        y_test = pd.Series([1, 1])

        trainer.train_multiple_models(
            ["random_forest", "logistic_regression"], X_train, y_train
        )
        comparison_df = trainer.compare_models(X_test, y_test)

        assert isinstance(comparison_df, pd.DataFrame)
        assert "model" in comparison_df.columns
        assert "f1_score" in comparison_df.columns
        assert len(comparison_df) == 2


class TestModelRegistry:
    def test_register_model(self):
        registry = ModelRegistry()

        model = RandomForestModel({"n_estimators": 5, "random_state": 42})

        X_train = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "feature2": [2, 4, 6, 8, 10]}
        )
        y_train = pd.Series([0, 0, 1, 1, 0])

        model.fit(X_train, y_train)

        model_id = registry.register_model(
            model, version="1.0", task_type="classification"
        )

        assert isinstance(model_id, int)
        assert model_id > 0

    def test_get_model(self):
        registry = ModelRegistry()

        model = RandomForestModel({"n_estimators": 5, "random_state": 42})

        X_train = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "feature2": [2, 4, 6, 8, 10]}
        )
        y_train = pd.Series([0, 0, 1, 1, 0])

        model.fit(X_train, y_train)

        model_id = registry.register_model(
            model, version="1.0", task_type="classification"
        )
        retrieved_model = registry.get_model(model_id)

        assert retrieved_model.is_trained
        assert retrieved_model.model_name == "RandomForest"

    def test_list_models(self):
        registry = ModelRegistry()

        model = RandomForestModel({"n_estimators": 5, "random_state": 42})

        X_train = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "feature2": [2, 4, 6, 8, 10]}
        )
        y_train = pd.Series([0, 0, 1, 1, 0])

        model.fit(X_train, y_train)

        registry.register_model(model, version="1.0", task_type="classification")
        models = registry.list_models()

        assert len(models) >= 1
        assert all("model_name" in model_info for model_info in models)
        assert all("task_type" in model_info for model_info in models)

    def test_get_model_statistics(self):
        registry = ModelRegistry()

        model = RandomForestModel({"n_estimators": 5, "random_state": 42})

        X_train = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "feature2": [2, 4, 6, 8, 10]}
        )
        y_train = pd.Series([0, 0, 1, 1, 0])

        model.fit(X_train, y_train)

        registry.register_model(model, version="1.0", task_type="classification")
        stats = registry.get_model_statistics()

        assert "total_models" in stats
        assert "active_models" in stats
        assert "inactive_models" in stats
        assert "models_by_task_type" in stats
        assert stats["active_models"] >= 1
