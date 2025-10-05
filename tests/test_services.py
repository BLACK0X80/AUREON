import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os
from pathlib import Path

from aureon.services.monitoring import (
    DriftDetector,
    PerformanceMonitor,
    SystemMonitor,
    ModelMonitor,
)
from aureon.services.scheduler import TaskScheduler
from aureon.services.reporting import ReportGenerator
from aureon.services.explainability import ModelExplainer, ModelInterpretability
from aureon.models.classification_models import RandomForestModel


class TestDriftDetector:
    def test_drift_detection(self):
        detector = DriftDetector(threshold=0.1)

        reference_data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.normal(0, 1, 100),
                "target": np.random.randint(0, 2, 100),
            }
        )

        current_data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.normal(0, 1, 100),
                "target": np.random.randint(0, 2, 100),
            }
        )

        result = detector.detect_drift(reference_data, current_data)

        assert "drift_detected" in result
        assert "drift_score" in result
        assert "feature_drift" in result
        assert "timestamp" in result
        assert "threshold" in result
        assert isinstance(result["drift_detected"], bool)
        assert isinstance(result["drift_score"], float)

    def test_drift_detection_with_shift(self):
        detector = DriftDetector(threshold=0.1)

        reference_data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.normal(0, 1, 100),
            }
        )

        current_data = pd.DataFrame(
            {
                "feature1": np.random.normal(2, 1, 100),
                "feature2": np.random.normal(0, 1, 100),
            }
        )

        result = detector.detect_drift(reference_data, current_data)

        assert result["drift_detected"] is True
        assert result["drift_score"] > 0.1
        assert "feature1" in result["feature_drift"]

    def test_model_drift_detection(self):
        detector = DriftDetector()

        model = RandomForestModel({"n_estimators": 10, "random_state": 42})

        reference_data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            }
        )

        current_data = pd.DataFrame(
            {
                "feature1": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                "feature2": [22, 24, 26, 28, 30, 32, 34, 36, 38, 40],
                "target": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            }
        )

        model.fit(reference_data[["feature1", "feature2"]], reference_data["target"])

        result = detector.detect_model_drift(
            model, reference_data, current_data, "target"
        )

        assert "performance_drift" in result
        assert "prediction_drift" in result
        assert "drift_detected" in result
        assert "timestamp" in result
        assert isinstance(result["performance_drift"], float)
        assert isinstance(result["prediction_drift"], float)

    def test_get_drift_history(self):
        detector = DriftDetector()

        reference_data = pd.DataFrame({"feature1": [1, 2, 3]})
        current_data = pd.DataFrame({"feature1": [4, 5, 6]})

        detector.detect_drift(reference_data, current_data)
        history = detector.get_drift_history()

        assert len(history) >= 1
        assert all("timestamp" in entry for entry in history)


class TestPerformanceMonitor:
    def test_monitor_model_performance(self):
        monitor = PerformanceMonitor()

        model = RandomForestModel({"n_estimators": 10, "random_state": 42})

        X_train = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            }
        )
        y_train = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        X_test = pd.DataFrame(
            {"feature1": [11, 12, 13, 14, 15], "feature2": [22, 24, 26, 28, 30]}
        )
        y_test = pd.Series([1, 1, 1, 0, 0])

        model.fit(X_train, y_train)

        result = monitor.monitor_model_performance(model, X_test, y_test)

        assert "model_name" in result
        assert "performance_score" in result
        assert "timestamp" in result
        assert "below_threshold" in result
        assert result["model_name"] == "RandomForest"
        assert isinstance(result["performance_score"], float)

    def test_get_performance_alerts(self):
        monitor = PerformanceMonitor()

        model = RandomForestModel({"n_estimators": 10, "random_state": 42})

        X_train = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "feature2": [2, 4, 6, 8, 10]}
        )
        y_train = pd.Series([0, 0, 0, 1, 1])

        X_test = pd.DataFrame(
            {"feature1": [6, 7, 8, 9, 10], "feature2": [12, 14, 16, 18, 20]}
        )
        y_test = pd.Series([1, 1, 1, 1, 1])

        model.fit(X_train, y_train)
        monitor.monitor_model_performance(model, X_test, y_test)

        alerts = monitor.get_performance_alerts()
        assert isinstance(alerts, list)

    def test_get_performance_trend(self):
        monitor = PerformanceMonitor()

        model = RandomForestModel({"n_estimators": 10, "random_state": 42})

        X_train = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "feature2": [2, 4, 6, 8, 10]}
        )
        y_train = pd.Series([0, 0, 0, 1, 1])

        X_test = pd.DataFrame(
            {"feature1": [6, 7, 8, 9, 10], "feature2": [12, 14, 16, 18, 20]}
        )
        y_test = pd.Series([1, 1, 1, 1, 1])

        model.fit(X_train, y_train)
        monitor.monitor_model_performance(model, X_test, y_test)

        trend = monitor.get_performance_trend("RandomForest", days=30)
        assert isinstance(trend, list)


class TestSystemMonitor:
    def test_get_system_metrics(self):
        monitor = SystemMonitor()

        metrics = monitor.get_system_metrics()

        assert "timestamp" in metrics
        assert "cpu_percent" in metrics
        assert "memory_percent" in metrics
        assert "memory_available_gb" in metrics
        assert "disk_percent" in metrics
        assert "disk_free_gb" in metrics
        assert isinstance(metrics["cpu_percent"], (int, float))
        assert isinstance(metrics["memory_percent"], (int, float))

    def test_check_system_health(self):
        monitor = SystemMonitor()

        health = monitor.check_system_health()

        assert "status" in health
        assert "alerts" in health
        assert "metrics" in health
        assert health["status"] in ["healthy", "warning", "critical"]
        assert isinstance(health["alerts"], list)

    def test_get_system_trend(self):
        monitor = SystemMonitor()

        monitor.get_system_metrics()
        trend = monitor.get_system_trend(hours=24)

        assert isinstance(trend, list)


class TestModelMonitor:
    def test_comprehensive_monitoring(self):
        monitor = ModelMonitor()

        model = RandomForestModel({"n_estimators": 10, "random_state": 42})

        reference_data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            }
        )

        current_data = pd.DataFrame(
            {
                "feature1": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                "feature2": [22, 24, 26, 28, 30, 32, 34, 36, 38, 40],
                "target": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            }
        )

        model.fit(reference_data[["feature1", "feature2"]], reference_data["target"])

        result = monitor.comprehensive_monitoring(
            model, reference_data, current_data, "target"
        )

        assert "timestamp" in result
        assert "data_drift" in result
        assert "model_drift" in result
        assert "system_health" in result
        assert "performance" in result

    def test_generate_monitoring_report(self):
        monitor = ModelMonitor()

        monitoring_results = {
            "timestamp": "2023-01-01T00:00:00",
            "data_drift": {"drift_detected": False, "drift_score": 0.05},
            "model_drift": {"drift_detected": False, "performance_drift": 0.02},
            "system_health": {"status": "healthy", "alerts": []},
            "performance": {"performance_score": 0.95, "below_threshold": False},
        }

        report = monitor.generate_monitoring_report(monitoring_results)

        assert "timestamp" in report
        assert "summary" in report
        assert "details" in report
        assert "recommendations" in report
        assert isinstance(report["recommendations"], list)


class TestTaskScheduler:
    def test_scheduler_initialization(self):
        scheduler = TaskScheduler()

        assert not scheduler.is_running
        assert scheduler.interval_minutes == 60
        assert isinstance(scheduler.tasks, dict)

    def test_add_task(self):
        scheduler = TaskScheduler()

        def test_task():
            return "test_result"

        scheduler.add_task("test_task", test_task, "hourly")

        assert "test_task" in scheduler.tasks
        assert scheduler.tasks["test_task"]["function"] == test_task
        assert scheduler.tasks["test_task"]["schedule"] == "hourly"

    def test_remove_task(self):
        scheduler = TaskScheduler()

        def test_task():
            return "test_result"

        scheduler.add_task("test_task", test_task, "hourly")
        assert "test_task" in scheduler.tasks

        scheduler.remove_task("test_task")
        assert "test_task" not in scheduler.tasks

    def test_get_task_status(self):
        scheduler = TaskScheduler()

        def test_task():
            return "test_result"

        scheduler.add_task("test_task", test_task, "hourly")
        status = scheduler.get_task_status()

        assert "scheduler_running" in status
        assert "interval_minutes" in status
        assert "tasks" in status
        assert "next_run_times" in status
        assert status["scheduler_running"] == False
        assert "test_task" in status["tasks"]


class TestReportGenerator:
    def test_generate_model_report(self):
        generator = ReportGenerator()

        model = RandomForestModel({"n_estimators": 10, "random_state": 42})

        X_train = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "feature2": [2, 4, 6, 8, 10]}
        )
        y_train = pd.Series([0, 0, 0, 1, 1])

        model.fit(X_train, y_train)
        model.evaluate(X_train, y_train, "classification")

        report = generator.generate_model_report(model)

        assert "report_type" in report
        assert "timestamp" in report
        assert "model_info" in report
        assert "performance_metrics" in report
        assert "feature_importance" in report
        assert "model_parameters" in report
        assert report["report_type"] == "model_report"

    def test_generate_experiment_report(self):
        generator = ReportGenerator()

        with patch("aureon.services.reporting.ModelRegistry") as mock_registry:
            mock_registry_instance = Mock()
            mock_registry_instance.list_models.return_value = [
                {
                    "id": 1,
                    "model_name": "TestModel",
                    "description": "test_experiment_1",
                    "metrics": {"f1_score": 0.95},
                }
            ]
            mock_registry.return_value = mock_registry_instance

            generator.registry = mock_registry_instance

            report = generator.generate_experiment_report("test_experiment")

            assert "report_type" in report
            assert "experiment_id" in report
            assert "models" in report
            assert "summary" in report
            assert report["report_type"] == "experiment_report"
            assert report["experiment_id"] == "test_experiment"

    def test_export_pdf_report(self):
        generator = ReportGenerator()

        model = RandomForestModel({"n_estimators": 10, "random_state": 42})

        X_train = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "feature2": [2, 4, 6, 8, 10]}
        )
        y_train = pd.Series([0, 0, 0, 1, 1])

        model.fit(X_train, y_train)
        model.evaluate(X_train, y_train, "classification")

        report_data = generator.generate_model_report(model)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            temp_path = f.name

        try:
            generator.export_pdf_report(report_data, temp_path)
            assert os.path.exists(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_export_html_report(self):
        generator = ReportGenerator()

        model = RandomForestModel({"n_estimators": 10, "random_state": 42})

        X_train = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "feature2": [2, 4, 6, 8, 10]}
        )
        y_train = pd.Series([0, 0, 0, 1, 1])

        model.fit(X_train, y_train)
        model.evaluate(X_train, y_train, "classification")

        report_data = generator.generate_model_report(model)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            temp_path = f.name

        try:
            generator.export_html_report(report_data, temp_path)
            assert os.path.exists(temp_path)

            with open(temp_path, "r") as f:
                content = f.read()
                assert "AUREON" in content
                assert "Model Report" in content
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestModelExplainer:
    def test_explain_prediction_shap(self):
        explainer = ModelExplainer()

        model = RandomForestModel({"n_estimators": 10, "random_state": 42})

        X_train = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            }
        )
        y_train = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        model.fit(X_train, y_train)

        X_test = pd.DataFrame({"feature1": [11, 12, 13], "feature2": [22, 24, 26]})

        with patch("aureon.services.explainability.SHAP_AVAILABLE", True):
            result = explainer.explain_prediction(model, X_test, "shap", 0)

            if "error" not in result:
                assert "explanation_type" in result
                assert "sample_index" in result
                assert "feature_importance" in result
                assert "prediction" in result
                assert "timestamp" in result
                assert result["explanation_type"] == "shap"

    def test_explain_prediction_lime(self):
        explainer = ModelExplainer()

        model = RandomForestModel({"n_estimators": 10, "random_state": 42})

        X_train = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            }
        )
        y_train = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        model.fit(X_train, y_train)

        X_test = pd.DataFrame({"feature1": [11, 12, 13], "feature2": [22, 24, 26]})

        with patch("aureon.services.explainability.LIME_AVAILABLE", True):
            result = explainer.explain_prediction(model, X_test, "lime", 0)

            if "error" not in result:
                assert "explanation_type" in result
                assert "sample_index" in result
                assert "feature_importance" in result
                assert "prediction" in result
                assert "timestamp" in result
                assert result["explanation_type"] == "lime"

    def test_get_feature_importance_global(self):
        explainer = ModelExplainer()

        model = RandomForestModel({"n_estimators": 10, "random_state": 42})

        X_train = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            }
        )
        y_train = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        model.fit(X_train, y_train)

        result = explainer.get_feature_importance_global(model, X_train)

        if "error" not in result:
            assert "feature_importance" in result
            assert "top_features" in result
            assert "timestamp" in result
            assert isinstance(result["feature_importance"], dict)
            assert isinstance(result["top_features"], list)

    def test_get_explanation_history(self):
        explainer = ModelExplainer()

        history = explainer.get_explanation_history()

        assert isinstance(history, list)

    def test_clear_explanation_cache(self):
        explainer = ModelExplainer()

        explainer.explainer_cache["test"] = "value"
        explainer.explanation_history.append({"test": "value"})

        explainer.clear_explanation_cache()

        assert len(explainer.explainer_cache) == 0
        assert len(explainer.explanation_history) == 0


class TestModelInterpretability:
    def test_comprehensive_analysis(self):
        interpretability = ModelInterpretability()

        model = RandomForestModel({"n_estimators": 10, "random_state": 42})

        X_train = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            }
        )
        y_train = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        model.fit(X_train, y_train)

        analysis = interpretability.comprehensive_analysis(model, X_train, y_train)

        assert "model_name" in analysis
        assert "task_type" in analysis
        assert "timestamp" in analysis
        assert "global_importance" in analysis
        assert "sample_explanations" in analysis
        assert "predictions" in analysis
        assert "actual_values" in analysis
        assert analysis["model_name"] == "RandomForest"
        assert analysis["task_type"] == "classification"
