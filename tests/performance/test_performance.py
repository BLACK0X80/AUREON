import pytest
import pandas as pd
import numpy as np
import time
import tempfile
import os
from unittest.mock import patch, MagicMock
import asyncio
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc

from aureon.pipeline.model_pipeline import ModelPipeline
from aureon.models.model_registry import ModelRegistry
from aureon.services.database import DatabaseService
from aureon.api.main import app
from fastapi.testclient import TestClient


@pytest.mark.performance
class TestModelTrainingPerformance:
    """Performance tests for model training."""
    
    @pytest.fixture
    def large_dataset(self):
        """Create large dataset for performance testing."""
        np.random.seed(42)
        n_samples = 10000
        n_features = 50
        
        data = {
            f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)
        }
        data['target'] = np.random.randint(0, 2, n_samples)
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def large_data_file(self, large_dataset):
        """Create large data file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            large_dataset.to_csv(f.name, index=False)
            temp_path = f.name
        
        yield temp_path
        
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def test_random_forest_training_performance(self, large_data_file):
        """Test Random Forest training performance."""
        pipeline = ModelPipeline()
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        result = pipeline.run_pipeline(
            data_source=large_data_file,
            target_column='target',
            task_type='classification',
            model_types=['random_forest'],
            experiment_name='performance_test_rf'
        )
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        training_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        assert training_time < 60  # Should complete within 60 seconds
        assert memory_usage < 1000  # Should use less than 1GB additional memory
        assert result['best_model'] is not None
        assert 'accuracy' in result['best_model'].model_metrics
    
    def test_logistic_regression_training_performance(self, large_data_file):
        """Test Logistic Regression training performance."""
        pipeline = ModelPipeline()
        
        start_time = time.time()
        
        result = pipeline.run_pipeline(
            data_source=large_data_file,
            target_column='target',
            task_type='classification',
            model_types=['logistic_regression'],
            experiment_name='performance_test_lr'
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        assert training_time < 30  # Should complete within 30 seconds
        assert result['best_model'] is not None
    
    def test_multiple_model_training_performance(self, large_data_file):
        """Test multiple model training performance."""
        pipeline = ModelPipeline()
        
        start_time = time.time()
        
        result = pipeline.run_pipeline(
            data_source=large_data_file,
            target_column='target',
            task_type='classification',
            model_types=['random_forest', 'logistic_regression', 'svm'],
            experiment_name='performance_test_multi'
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        assert training_time < 120  # Should complete within 2 minutes
        assert len(result['models']) == 3
        assert result['best_model'] is not None
    
    def test_hyperparameter_tuning_performance(self, large_data_file):
        """Test hyperparameter tuning performance."""
        pipeline = ModelPipeline()
        
        start_time = time.time()
        
        result = pipeline.run_pipeline(
            data_source=large_data_file,
            target_column='target',
            task_type='classification',
            model_types=['random_forest'],
            hyperparameter_tuning=True,
            hyperparameter_config={
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15]
                }
            },
            experiment_name='performance_test_hp'
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        assert training_time < 300  # Should complete within 5 minutes
        assert result['best_model'] is not None
        assert 'hyperparameter_results' in result


@pytest.mark.performance
class TestPredictionPerformance:
    """Performance tests for model predictions."""
    
    @pytest.fixture
    def trained_model(self, sample_data):
        """Create trained model for prediction testing."""
        pipeline = ModelPipeline()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            result = pipeline.run_pipeline(
                data_source=temp_path,
                target_column='target',
                task_type='classification',
                model_types=['random_forest'],
                experiment_name='prediction_perf_test'
            )
            
            return result['best_model']
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_batch_prediction_performance(self, trained_model):
        """Test batch prediction performance."""
        np.random.seed(42)
        n_samples = 10000
        n_features = 3
        
        test_data = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples)
        })
        
        start_time = time.time()
        predictions = trained_model.predict(test_data)
        end_time = time.time()
        
        prediction_time = end_time - start_time
        throughput = n_samples / prediction_time
        
        assert prediction_time < 5  # Should complete within 5 seconds
        assert throughput > 1000  # Should process at least 1000 samples/second
        assert len(predictions) == n_samples
    
    def test_single_prediction_performance(self, trained_model):
        """Test single prediction performance."""
        test_sample = pd.DataFrame({
            'feature1': [1.0],
            'feature2': [2.0],
            'feature3': [0.5]
        })
        
        times = []
        for _ in range(100):
            start_time = time.time()
            prediction = trained_model.predict(test_sample)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        p95_time = np.percentile(times, 95)
        
        assert avg_time < 0.01  # Average should be less than 10ms
        assert p95_time < 0.05  # 95th percentile should be less than 50ms
    
    def test_probability_prediction_performance(self, trained_model):
        """Test probability prediction performance."""
        np.random.seed(42)
        n_samples = 5000
        
        test_data = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples)
        })
        
        if hasattr(trained_model, 'predict_proba'):
            start_time = time.time()
            probabilities = trained_model.predict_proba(test_data)
            end_time = time.time()
            
            prediction_time = end_time - start_time
            throughput = n_samples / prediction_time
            
            assert prediction_time < 3  # Should complete within 3 seconds
            assert throughput > 1000  # Should process at least 1000 samples/second
            assert probabilities.shape == (n_samples, 2)


@pytest.mark.performance
class TestDatabasePerformance:
    """Performance tests for database operations."""
    
    @pytest.fixture
    def db_service(self, temp_directory):
        """Create database service."""
        db_url = f"sqlite:///{temp_directory}/perf_test.db"
        return DatabaseService(db_url)
    
    def test_bulk_experiment_creation(self, db_service):
        """Test bulk experiment creation performance."""
        n_experiments = 1000
        
        start_time = time.time()
        
        experiment_ids = []
        for i in range(n_experiments):
            experiment_id = db_service.create_experiment(
                f'perf_experiment_{i}',
                f'Performance test experiment {i}',
                'classification',
                {'param1': f'value_{i}'}
            )
            experiment_ids.append(experiment_id)
        
        end_time = time.time()
        creation_time = end_time - start_time
        
        throughput = n_experiments / creation_time
        
        assert creation_time < 30  # Should complete within 30 seconds
        assert throughput > 30  # Should create at least 30 experiments/second
        assert len(experiment_ids) == n_experiments
    
    def test_bulk_experiment_retrieval(self, db_service):
        """Test bulk experiment retrieval performance."""
        n_experiments = 100
        
        experiment_ids = []
        for i in range(n_experiments):
            experiment_id = db_service.create_experiment(
                f'retrieval_test_{i}',
                f'Retrieval test experiment {i}',
                'classification',
                {}
            )
            experiment_ids.append(experiment_id)
        
        start_time = time.time()
        
        experiments = []
        for experiment_id in experiment_ids:
            experiment = db_service.get_experiment(experiment_id)
            experiments.append(experiment)
        
        end_time = time.time()
        retrieval_time = end_time - start_time
        
        throughput = n_experiments / retrieval_time
        
        assert retrieval_time < 10  # Should complete within 10 seconds
        assert throughput > 10  # Should retrieve at least 10 experiments/second
        assert len(experiments) == n_experiments
    
    def test_experiment_listing_performance(self, db_service):
        """Test experiment listing performance."""
        n_experiments = 500
        
        for i in range(n_experiments):
            db_service.create_experiment(
                f'listing_test_{i}',
                f'Listing test experiment {i}',
                'classification',
                {}
            )
        
        start_time = time.time()
        experiments = db_service.list_experiments()
        end_time = time.time()
        
        listing_time = end_time - start_time
        
        assert listing_time < 5  # Should complete within 5 seconds
        assert len(experiments) == n_experiments
    
    def test_model_run_creation_performance(self, db_service):
        """Test model run creation performance."""
        experiment_id = db_service.create_experiment(
            'model_run_perf_test',
            'Model run performance test',
            'classification',
            {}
        )
        
        n_runs = 1000
        
        start_time = time.time()
        
        run_ids = []
        for i in range(n_runs):
            run_id = db_service.create_model_run(
                experiment_id,
                f'Model_{i}',
                'random_forest',
                {'n_estimators': 100}
            )
            run_ids.append(run_id)
        
        end_time = time.time()
        creation_time = end_time - start_time
        
        throughput = n_runs / creation_time
        
        assert creation_time < 20  # Should complete within 20 seconds
        assert throughput > 50  # Should create at least 50 runs/second
        assert len(run_ids) == n_runs


@pytest.mark.performance
class TestAPIPerformance:
    """Performance tests for API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_health_endpoint_performance(self, client):
        """Test health endpoint performance."""
        times = []
        
        for _ in range(100):
            start_time = time.time()
            response = client.get("/health")
            end_time = time.time()
            
            assert response.status_code == 200
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        p95_time = np.percentile(times, 95)
        p99_time = np.percentile(times, 99)
        
        assert avg_time < 0.1  # Average should be less than 100ms
        assert p95_time < 0.2  # 95th percentile should be less than 200ms
        assert p99_time < 0.5  # 99th percentile should be less than 500ms
    
    def test_models_list_performance(self, client):
        """Test models list endpoint performance."""
        times = []
        
        for _ in range(50):
            start_time = time.time()
            response = client.get("/api/v1/models")
            end_time = time.time()
            
            assert response.status_code == 200
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        p95_time = np.percentile(times, 95)
        
        assert avg_time < 0.2  # Average should be less than 200ms
        assert p95_time < 0.5  # 95th percentile should be less than 500ms
    
    def test_concurrent_requests_performance(self, client):
        """Test concurrent requests performance."""
        def make_request():
            response = client.get("/health")
            return response.status_code == 200
        
        n_requests = 100
        n_workers = 10
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(make_request) for _ in range(n_requests)]
            results = [future.result() for future in futures]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        throughput = n_requests / total_time
        
        assert all(results)  # All requests should succeed
        assert total_time < 10  # Should complete within 10 seconds
        assert throughput > 10  # Should handle at least 10 requests/second


@pytest.mark.performance
class TestMemoryPerformance:
    """Performance tests for memory usage."""
    
    def test_model_training_memory_usage(self, sample_data):
        """Test memory usage during model training."""
        pipeline = ModelPipeline()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            gc.collect()  # Clean up before test
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            result = pipeline.run_pipeline(
                data_source=temp_path,
                target_column='target',
                task_type='classification',
                model_types=['random_forest', 'logistic_regression', 'svm'],
                experiment_name='memory_test'
            )
            
            peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory
            
            gc.collect()  # Clean up after test
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            assert memory_increase < 500  # Should use less than 500MB additional memory
            assert final_memory < initial_memory + 100  # Should clean up most memory
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_large_dataset_memory_usage(self):
        """Test memory usage with large dataset."""
        np.random.seed(42)
        n_samples = 50000
        n_features = 100
        
        data = {
            f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)
        }
        data['target'] = np.random.randint(0, 2, n_samples)
        
        large_dataset = pd.DataFrame(data)
        
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            large_dataset.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            pipeline = ModelPipeline()
            
            result = pipeline.run_pipeline(
                data_source=temp_path,
                target_column='target',
                task_type='classification',
                model_types=['random_forest'],
                experiment_name='large_dataset_memory_test'
            )
            
            peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory
            
            assert memory_increase < 2000  # Should use less than 2GB additional memory
            assert result['best_model'] is not None
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
            del large_dataset
            gc.collect()
    
    def test_model_registry_memory_usage(self, sample_data, temp_directory):
        """Test memory usage in model registry."""
        registry = ModelRegistry()
        registry.models_path = temp_directory
        
        from aureon.models.classification_models import ClassificationModelFactory
        
        factory = ClassificationModelFactory()
        
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        model_ids = []
        for i in range(10):
            model = factory.create_model('random_forest')
            model.fit(X, y)
            
            model_id = registry.save_model(
                model,
                f'memory_test_model_{i}',
                'classification',
                {'n_estimators': 100},
                {'accuracy': 0.95}
            )
            model_ids.append(model_id)
        
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        assert memory_increase < 200  # Should use less than 200MB additional memory
        
        for model_id in model_ids:
            loaded_model = registry.load_model(model_id)
            assert loaded_model is not None
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        assert final_memory < initial_memory + 300  # Should not leak excessive memory


@pytest.mark.performance
class TestScalabilityPerformance:
    """Performance tests for scalability."""
    
    def test_dataset_size_scalability(self):
        """Test performance scaling with dataset size."""
        dataset_sizes = [1000, 5000, 10000]
        training_times = []
        
        for size in dataset_sizes:
            np.random.seed(42)
            data = {
                'feature1': np.random.randn(size),
                'feature2': np.random.randn(size),
                'feature3': np.random.randn(size),
                'target': np.random.randint(0, 2, size)
            }
            
            dataset = pd.DataFrame(data)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                dataset.to_csv(f.name, index=False)
                temp_path = f.name
            
            try:
                pipeline = ModelPipeline()
                
                start_time = time.time()
                result = pipeline.run_pipeline(
                    data_source=temp_path,
                    target_column='target',
                    task_type='classification',
                    model_types=['random_forest'],
                    experiment_name=f'scalability_test_{size}'
                )
                end_time = time.time()
                
                training_times.append(end_time - start_time)
                
                assert result['best_model'] is not None
            
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        # Training time should scale sub-linearly with dataset size
        time_ratio = training_times[-1] / training_times[0]
        size_ratio = dataset_sizes[-1] / dataset_sizes[0]
        
        assert time_ratio < size_ratio  # Should scale better than linearly
    
    def test_feature_count_scalability(self):
        """Test performance scaling with feature count."""
        feature_counts = [10, 50, 100]
        training_times = []
        
        for n_features in feature_counts:
            np.random.seed(42)
            data = {
                f'feature_{i}': np.random.randn(5000) for i in range(n_features)
            }
            data['target'] = np.random.randint(0, 2, 5000)
            
            dataset = pd.DataFrame(data)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                dataset.to_csv(f.name, index=False)
                temp_path = f.name
            
            try:
                pipeline = ModelPipeline()
                
                start_time = time.time()
                result = pipeline.run_pipeline(
                    data_source=temp_path,
                    target_column='target',
                    task_type='classification',
                    model_types=['random_forest'],
                    experiment_name=f'feature_scalability_test_{n_features}'
                )
                end_time = time.time()
                
                training_times.append(end_time - start_time)
                
                assert result['best_model'] is not None
            
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        # Training time should scale reasonably with feature count
        time_ratio = training_times[-1] / training_times[0]
        feature_ratio = feature_counts[-1] / feature_counts[0]
        
        assert time_ratio < feature_ratio * 2  # Should scale better than quadratically
    
    def test_model_count_scalability(self, sample_data):
        """Test performance scaling with number of models."""
        model_counts = [1, 3, 5]
        training_times = []
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            for n_models in model_counts:
                model_types = ['random_forest', 'logistic_regression', 'svm', 'gradient_boosting', 'naive_bayes'][:n_models]
                
                pipeline = ModelPipeline()
                
                start_time = time.time()
                result = pipeline.run_pipeline(
                    data_source=temp_path,
                    target_column='target',
                    task_type='classification',
                    model_types=model_types,
                    experiment_name=f'model_count_test_{n_models}'
                )
                end_time = time.time()
                
                training_times.append(end_time - start_time)
                
                assert len(result['models']) == n_models
                assert result['best_model'] is not None
            
            # Training time should scale roughly linearly with model count
            time_ratio = training_times[-1] / training_times[0]
            model_ratio = model_counts[-1] / model_counts[0]
            
            assert time_ratio <= model_ratio * 1.5  # Should scale roughly linearly
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
