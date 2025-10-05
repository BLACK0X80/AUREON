import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import asyncio
from unittest.mock import patch, MagicMock
import httpx
from fastapi.testclient import TestClient

from aureon.api.main import app
from aureon.services.database import DatabaseService
from aureon.models.model_registry import ModelRegistry
from aureon.pipeline.model_pipeline import ModelPipeline


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_data_file(self, sample_data):
        """Create sample data file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        yield temp_path
        
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def test_health_endpoint(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert 'timestamp' in data
        assert 'system_info' in data
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data['message'] == "AUREON API"
        assert data['version'] == "1.0.0"
        assert data['status'] == "running"
    
    def test_models_list_endpoint(self, client):
        """Test models list endpoint."""
        response = client.get("/api/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        assert 'models' in data
        assert 'total_count' in data
        assert isinstance(data['models'], list)
    
    def test_system_status_endpoint(self, client):
        """Test system status endpoint."""
        response = client.get("/api/v1/system/status")
        
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert 'timestamp' in data
    
    def test_system_metrics_endpoint(self, client):
        """Test system metrics endpoint."""
        response = client.get("/api/v1/system/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert 'cpu_percent' in data
        assert 'memory_percent' in data
        assert 'disk_percent' in data


@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests for database operations."""
    
    @pytest.fixture
    def db_service(self, temp_directory):
        """Create database service."""
        db_url = f"sqlite:///{temp_directory}/test.db"
        return DatabaseService(db_url)
    
    def test_experiment_lifecycle(self, db_service):
        """Test complete experiment lifecycle."""
        experiment_id = db_service.create_experiment(
            'integration_test',
            'Integration test experiment',
            'classification',
            {'param1': 'value1', 'param2': 'value2'}
        )
        
        assert isinstance(experiment_id, int)
        assert experiment_id > 0
        
        experiment = db_service.get_experiment(experiment_id)
        assert experiment is not None
        assert experiment['name'] == 'integration_test'
        assert experiment['task_type'] == 'classification'
        
        db_service.update_experiment(experiment_id, status='running')
        updated_experiment = db_service.get_experiment(experiment_id)
        assert updated_experiment['status'] == 'running'
        
        db_service.update_experiment(
            experiment_id, 
            status='completed',
            results={'accuracy': 0.95, 'f1_score': 0.92}
        )
        completed_experiment = db_service.get_experiment(experiment_id)
        assert completed_experiment['status'] == 'completed'
        assert 'accuracy' in completed_experiment['results']
    
    def test_model_run_lifecycle(self, db_service):
        """Test complete model run lifecycle."""
        experiment_id = db_service.create_experiment(
            'model_run_test',
            'Model run test experiment',
            'classification',
            {}
        )
        
        run_id = db_service.create_model_run(
            experiment_id,
            'RandomForest',
            'random_forest',
            {'n_estimators': 100, 'max_depth': 5}
        )
        
        assert isinstance(run_id, int)
        assert run_id > 0
        
        db_service.update_model_run(
            run_id,
            metrics={'accuracy': 0.95, 'precision': 0.93, 'recall': 0.97},
            status='completed'
        )
        
        runs = db_service.get_model_runs(experiment_id)
        assert len(runs) == 1
        assert runs[0]['status'] == 'completed'
        assert 'accuracy' in runs[0]['metrics']
    
    def test_dataset_management(self, db_service, sample_data):
        """Test dataset management operations."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            dataset_id = db_service.create_dataset(
                'integration_dataset',
                temp_path,
                'csv',
                os.path.getsize(temp_path),
                len(sample_data),
                len(sample_data.columns),
                'target',
                {'description': 'Integration test dataset'}
            )
            
            assert isinstance(dataset_id, int)
            assert dataset_id > 0
            
            dataset = db_service.get_dataset(dataset_id)
            assert dataset is not None
            assert dataset['name'] == 'integration_dataset'
            assert dataset['file_type'] == 'csv'
            assert dataset['rows'] == len(sample_data)
            
            datasets = db_service.list_datasets()
            assert len(datasets) == 1
            assert datasets[0]['name'] == 'integration_dataset'
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_performance_logging(self, db_service):
        """Test performance logging operations."""
        experiment_id = db_service.create_experiment(
            'performance_test',
            'Performance test experiment',
            'classification',
            {}
        )
        
        run_id = db_service.create_model_run(
            experiment_id,
            'TestModel',
            'random_forest',
            {}
        )
        
        log_id = db_service.log_performance(
            run_id,
            'accuracy',
            0.95,
            metadata={'dataset': 'test', 'split': 'validation'}
        )
        
        assert isinstance(log_id, int)
        assert log_id > 0
        
        history = db_service.get_performance_history(run_id, 'accuracy')
        assert len(history) == 1
        assert history[0]['metric_value'] == 0.95
        assert history[0]['metric_name'] == 'accuracy'
    
    def test_database_statistics(self, db_service):
        """Test database statistics."""
        db_service.create_experiment('exp1', 'Exp 1', 'classification', {})
        db_service.create_experiment('exp2', 'Exp 2', 'regression', {})
        db_service.create_experiment('exp3', 'Exp 3', 'classification', {})
        
        stats = db_service.get_database_stats()
        
        assert stats['experiments'] == 3
        assert stats['experiments_by_task_type']['classification'] == 2
        assert stats['experiments_by_task_type']['regression'] == 1


@pytest.mark.integration
class TestModelPipelineIntegration:
    """Integration tests for model pipeline."""
    
    @pytest.fixture
    def sample_data_file(self, sample_data):
        """Create sample data file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        yield temp_path
        
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def test_complete_training_pipeline(self, sample_data_file, temp_directory):
        """Test complete training pipeline."""
        pipeline = ModelPipeline()
        
        result = pipeline.run_pipeline(
            data_source=sample_data_file,
            target_column='target',
            task_type='classification',
            model_types=['random_forest', 'logistic_regression'],
            experiment_name='integration_training_test'
        )
        
        assert isinstance(result, dict)
        assert 'models' in result
        assert 'best_model' in result
        assert 'experiment_id' in result
        assert 'training_summary' in result
        
        assert len(result['models']) == 2
        assert result['best_model'] is not None
        assert hasattr(result['best_model'], 'predict')
        assert hasattr(result['best_model'], 'model_metrics')
    
    def test_model_comparison(self, sample_data_file):
        """Test model comparison functionality."""
        pipeline = ModelPipeline()
        
        result = pipeline.run_pipeline(
            data_source=sample_data_file,
            target_column='target',
            task_type='classification',
            model_types=['random_forest', 'logistic_regression', 'svm'],
            experiment_name='model_comparison_test'
        )
        
        assert isinstance(result, dict)
        assert 'model_comparison' in result
        
        comparison = result['model_comparison']
        assert len(comparison) == 3
        
        for model_info in comparison:
            assert 'model_name' in model_info
            assert 'metrics' in model_info
            assert 'training_time' in model_info
    
    def test_hyperparameter_tuning(self, sample_data_file):
        """Test hyperparameter tuning integration."""
        pipeline = ModelPipeline()
        
        result = pipeline.run_pipeline(
            data_source=sample_data_file,
            target_column='target',
            task_type='classification',
            model_types=['random_forest'],
            hyperparameter_tuning=True,
            hyperparameter_config={
                'random_forest': {
                    'n_estimators': [10, 50, 100],
                    'max_depth': [3, 5, 10]
                }
            },
            experiment_name='hyperparameter_tuning_test'
        )
        
        assert isinstance(result, dict)
        assert 'best_model' in result
        assert 'hyperparameter_results' in result
        
        best_model = result['best_model']
        assert hasattr(best_model, 'model_params')
        assert best_model.model_params is not None
    
    def test_model_retraining(self, sample_data_file):
        """Test model retraining functionality."""
        pipeline = ModelPipeline()
        
        initial_result = pipeline.run_pipeline(
            data_source=sample_data_file,
            target_column='target',
            task_type='classification',
            model_types=['random_forest'],
            experiment_name='retraining_test'
        )
        
        initial_model = initial_result['best_model']
        initial_performance = initial_model.model_metrics.get('accuracy', 0)
        
        retrain_result = pipeline.retrain_model(
            model_id=initial_model.model_id,
            new_data_source=sample_data_file,
            target_column='target'
        )
        
        assert isinstance(retrain_result, dict)
        assert 'new_model' in retrain_result
        assert 'performance_comparison' in retrain_result
        
        new_model = retrain_result['new_model']
        assert hasattr(new_model, 'predict')
        assert hasattr(new_model, 'model_metrics')
        
        comparison = retrain_result['performance_comparison']
        assert 'initial_performance' in comparison
        assert 'new_performance' in comparison


@pytest.mark.integration
class TestModelRegistryIntegration:
    """Integration tests for model registry."""
    
    @pytest.fixture
    def registry(self, temp_directory):
        """Create model registry."""
        registry = ModelRegistry()
        registry.models_path = temp_directory
        return registry
    
    def test_model_save_load_cycle(self, registry, sample_data):
        """Test complete model save/load cycle."""
        from aureon.models.classification_models import ClassificationModelFactory
        
        factory = ClassificationModelFactory()
        model = factory.create_model('random_forest')
        
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        model.fit(X, y)
        
        model_id = registry.save_model(
            model,
            'integration_test_model',
            'classification',
            {'n_estimators': 100},
            {'accuracy': 0.95}
        )
        
        assert isinstance(model_id, int)
        assert model_id > 0
        
        loaded_model = registry.load_model(model_id)
        assert loaded_model is not None
        assert hasattr(loaded_model, 'predict')
        
        predictions = loaded_model.predict(X.iloc[:3])
        assert len(predictions) == 3
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_model_versioning(self, registry, sample_data):
        """Test model versioning functionality."""
        from aureon.models.classification_models import ClassificationModelFactory
        
        factory = ClassificationModelFactory()
        model1 = factory.create_model('random_forest')
        model2 = factory.create_model('logistic_regression')
        
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        model1.fit(X, y)
        model2.fit(X, y)
        
        model_id1 = registry.save_model(
            model1,
            'versioned_model',
            'classification',
            {'n_estimators': 50},
            {'accuracy': 0.90}
        )
        
        model_id2 = registry.save_model(
            model2,
            'versioned_model',
            'classification',
            {'C': 1.0},
            {'accuracy': 0.88}
        )
        
        models = registry.list_models(model_name='versioned_model')
        assert len(models) == 2
        
        versions = registry.get_model_versions('versioned_model')
        assert len(versions) == 2
    
    def test_model_metadata_management(self, registry, sample_data):
        """Test model metadata management."""
        from aureon.models.classification_models import ClassificationModelFactory
        
        factory = ClassificationModelFactory()
        model = factory.create_model('random_forest')
        
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        model.fit(X, y)
        
        metadata = {
            'description': 'Integration test model',
            'author': 'Test User',
            'tags': ['test', 'integration'],
            'dataset_info': {
                'rows': len(sample_data),
                'features': len(sample_data.columns) - 1
            }
        }
        
        model_id = registry.save_model(
            model,
            'metadata_test_model',
            'classification',
            {'n_estimators': 100},
            {'accuracy': 0.95},
            metadata=metadata
        )
        
        model_info = registry.get_model_info(model_id)
        assert model_info is not None
        assert model_info['metadata'] == metadata
        
        registry.update_model_metadata(model_id, {'new_tag': 'updated'})
        updated_info = registry.get_model_info(model_id)
        assert 'new_tag' in updated_info['metadata']
    
    def test_model_deletion_and_cleanup(self, registry, sample_data):
        """Test model deletion and cleanup."""
        from aureon.models.classification_models import ClassificationModelFactory
        
        factory = ClassificationModelFactory()
        model = factory.create_model('random_forest')
        
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        model.fit(X, y)
        
        model_id = registry.save_model(
            model,
            'deletion_test_model',
            'classification',
            {'n_estimators': 100},
            {'accuracy': 0.95}
        )
        
        models_before = registry.list_models()
        assert len(models_before) == 1
        
        result = registry.delete_model(model_id)
        assert result is True
        
        models_after = registry.list_models()
        assert len(models_after) == 0
        
        with pytest.raises(ValueError):
            registry.load_model(model_id)


@pytest.mark.integration
class TestEndToEndWorkflow:
    """End-to-end integration tests."""
    
    def test_complete_ml_workflow(self, sample_data, temp_directory):
        """Test complete ML workflow from data ingestion to prediction."""
        temp_file = os.path.join(temp_directory, 'workflow_test.csv')
        sample_data.to_csv(temp_file, index=False)
        
        db_url = f"sqlite:///{temp_directory}/workflow_test.db"
        db_service = DatabaseService(db_url)
        
        pipeline = ModelPipeline()
        
        training_result = pipeline.run_pipeline(
            data_source=temp_file,
            target_column='target',
            task_type='classification',
            model_types=['random_forest', 'logistic_regression'],
            experiment_name='complete_workflow_test'
        )
        
        assert isinstance(training_result, dict)
        assert 'best_model' in training_result
        
        best_model = training_result['best_model']
        experiment_id = training_result['experiment_id']
        
        experiment = db_service.get_experiment(experiment_id)
        assert experiment is not None
        assert experiment['name'] == 'complete_workflow_test'
        
        test_data = sample_data.drop('target', axis=1).iloc[:5]
        predictions = best_model.predict(test_data)
        
        assert len(predictions) == 5
        assert all(pred in [0, 1] for pred in predictions)
        
        if hasattr(best_model, 'predict_proba'):
            probabilities = best_model.predict_proba(test_data)
            assert probabilities.shape == (5, 2)
            assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_model_performance_monitoring_workflow(self, sample_data, temp_directory):
        """Test model performance monitoring workflow."""
        temp_file = os.path.join(temp_directory, 'monitoring_test.csv')
        sample_data.to_csv(temp_file, index=False)
        
        pipeline = ModelPipeline()
        
        training_result = pipeline.run_pipeline(
            data_source=temp_file,
            target_column='target',
            task_type='classification',
            model_types=['random_forest'],
            experiment_name='monitoring_test'
        )
        
        best_model = training_result['best_model']
        
        from aureon.services.monitoring import SystemMonitor
        monitor = SystemMonitor()
        
        test_data = sample_data.drop('target', axis=1).iloc[:10]
        test_targets = sample_data['target'].iloc[:10]
        
        performance = monitor.monitor_model_performance(
            best_model,
            test_data,
            test_targets
        )
        
        assert isinstance(performance, dict)
        assert 'performance_score' in performance
        assert 'timestamp' in performance
    
    def test_model_explanation_workflow(self, sample_data, temp_directory):
        """Test model explanation workflow."""
        temp_file = os.path.join(temp_directory, 'explanation_test.csv')
        sample_data.to_csv(temp_file, index=False)
        
        pipeline = ModelPipeline()
        
        training_result = pipeline.run_pipeline(
            data_source=temp_file,
            target_column='target',
            task_type='classification',
            model_types=['random_forest'],
            experiment_name='explanation_test'
        )
        
        best_model = training_result['best_model']
        
        from aureon.services.explainability import ExplainabilityService
        explainer = ExplainabilityService()
        
        test_data = sample_data.drop('target', axis=1).iloc[:3]
        
        explanation = explainer.generate_explanation(
            best_model,
            test_data,
            method='shap'
        )
        
        assert isinstance(explanation, dict)
        assert 'feature_importance' in explanation
        assert 'sample_explanations' in explanation
    
    def test_model_reporting_workflow(self, sample_data, temp_directory):
        """Test model reporting workflow."""
        temp_file = os.path.join(temp_directory, 'reporting_test.csv')
        sample_data.to_csv(temp_file, index=False)
        
        pipeline = ModelPipeline()
        
        training_result = pipeline.run_pipeline(
            data_source=temp_file,
            target_column='target',
            task_type='classification',
            model_types=['random_forest', 'logistic_regression'],
            experiment_name='reporting_test'
        )
        
        from aureon.services.reporting import ReportingService
        reporter = ReportingService()
        
        report = reporter.generate_experiment_report(
            training_result['models'],
            output_format='html'
        )
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert '<html>' in report.lower() or 'html' in report.lower()
