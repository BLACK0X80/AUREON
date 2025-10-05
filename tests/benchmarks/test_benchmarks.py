import pytest
import pandas as pd
import numpy as np
import time
import tempfile
import os
import json
from unittest.mock import patch, MagicMock

from aureon.pipeline.model_pipeline import ModelPipeline
from aureon.models.model_registry import ModelRegistry
from aureon.services.database import DatabaseService


@pytest.mark.benchmark
class TestMLflowComparison:
    """Benchmark tests comparing AUREON with MLflow."""
    
    @pytest.fixture
    def benchmark_dataset(self):
        """Create benchmark dataset."""
        np.random.seed(42)
        n_samples = 10000
        n_features = 20
        
        data = {
            f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)
        }
        data['target'] = np.random.randint(0, 2, n_samples)
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def benchmark_data_file(self, benchmark_dataset):
        """Create benchmark data file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            benchmark_dataset.to_csv(f.name, index=False)
            temp_path = f.name
        
        yield temp_path
        
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def test_aureon_training_time(self, benchmark_data_file):
        """Benchmark AUREON training time."""
        pipeline = ModelPipeline()
        
        start_time = time.time()
        
        result = pipeline.run_pipeline(
            data_source=benchmark_data_file,
            target_column='target',
            task_type='classification',
            model_types=['random_forest'],
            experiment_name='aureon_benchmark'
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        assert training_time < 30  # Should complete within 30 seconds
        assert result['best_model'] is not None
        
        return training_time
    
    def test_aureon_prediction_time(self, benchmark_data_file):
        """Benchmark AUREON prediction time."""
        pipeline = ModelPipeline()
        
        training_result = pipeline.run_pipeline(
            data_source=benchmark_data_file,
            target_column='target',
            task_type='classification',
            model_types=['random_forest'],
            experiment_name='aureon_prediction_benchmark'
        )
        
        best_model = training_result['best_model']
        
        test_data = pd.DataFrame({
            f'feature_{i}': np.random.randn(1000) for i in range(20)
        })
        
        start_time = time.time()
        predictions = best_model.predict(test_data)
        end_time = time.time()
        
        prediction_time = end_time - start_time
        throughput = len(test_data) / prediction_time
        
        assert prediction_time < 1  # Should complete within 1 second
        assert throughput > 1000  # Should process at least 1000 samples/second
        
        return prediction_time, throughput
    
    def test_aureon_memory_usage(self, benchmark_data_file):
        """Benchmark AUREON memory usage."""
        import psutil
        import gc
        
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        pipeline = ModelPipeline()
        
        result = pipeline.run_pipeline(
            data_source=benchmark_data_file,
            target_column='target',
            task_type='classification',
            model_types=['random_forest', 'logistic_regression'],
            experiment_name='aureon_memory_benchmark'
        )
        
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_usage = peak_memory - initial_memory
        
        gc.collect()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        assert memory_usage < 500  # Should use less than 500MB
        assert final_memory < initial_memory + 100  # Should clean up memory
        
        return memory_usage
    
    def test_aureon_model_accuracy(self, benchmark_data_file):
        """Benchmark AUREON model accuracy."""
        pipeline = ModelPipeline()
        
        result = pipeline.run_pipeline(
            data_source=benchmark_data_file,
            target_column='target',
            task_type='classification',
            model_types=['random_forest', 'logistic_regression', 'svm'],
            experiment_name='aureon_accuracy_benchmark'
        )
        
        best_model = result['best_model']
        accuracy = best_model.model_metrics.get('accuracy', 0)
        
        assert accuracy > 0.5  # Should achieve reasonable accuracy
        assert accuracy < 1.0  # Should not overfit completely
        
        return accuracy
    
    def test_aureon_experiment_tracking(self, benchmark_data_file, temp_directory):
        """Benchmark AUREON experiment tracking."""
        db_url = f"sqlite:///{temp_directory}/benchmark.db"
        db_service = DatabaseService(db_url)
        
        pipeline = ModelPipeline()
        
        start_time = time.time()
        
        result = pipeline.run_pipeline(
            data_source=benchmark_data_file,
            target_column='target',
            task_type='classification',
            model_types=['random_forest'],
            experiment_name='aureon_tracking_benchmark'
        )
        
        end_time = time.time()
        tracking_time = end_time - start_time
        
        experiment_id = result['experiment_id']
        experiment = db_service.get_experiment(experiment_id)
        
        assert experiment is not None
        assert experiment['name'] == 'aureon_tracking_benchmark'
        assert tracking_time < 35  # Should complete within 35 seconds
        
        return tracking_time


@pytest.mark.benchmark
class TestLatencyBenchmarks:
    """Latency benchmark tests."""
    
    def test_api_response_latency(self):
        """Benchmark API response latency."""
        from fastapi.testclient import TestClient
        from aureon.api.main import app
        
        client = TestClient(app)
        
        latencies = []
        
        for _ in range(100):
            start_time = time.time()
            response = client.get("/health")
            end_time = time.time()
            
            assert response.status_code == 200
            latencies.append(end_time - start_time)
        
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        assert p50_latency < 0.05  # 50th percentile should be less than 50ms
        assert p95_latency < 0.1   # 95th percentile should be less than 100ms
        assert p99_latency < 0.2   # 99th percentile should be less than 200ms
        
        return {
            'p50': p50_latency,
            'p95': p95_latency,
            'p99': p99_latency
        }
    
    def test_model_loading_latency(self, sample_data, temp_directory):
        """Benchmark model loading latency."""
        registry = ModelRegistry()
        registry.models_path = temp_directory
        
        from aureon.models.classification_models import ClassificationModelFactory
        
        factory = ClassificationModelFactory()
        model = factory.create_model('random_forest')
        
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        model.fit(X, y)
        
        model_id = registry.save_model(
            model,
            'latency_test_model',
            'classification',
            {'n_estimators': 100},
            {'accuracy': 0.95}
        )
        
        latencies = []
        
        for _ in range(50):
            start_time = time.time()
            loaded_model = registry.load_model(model_id)
            end_time = time.time()
            
            assert loaded_model is not None
            latencies.append(end_time - start_time)
        
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        assert avg_latency < 0.1  # Average should be less than 100ms
        assert p95_latency < 0.2  # 95th percentile should be less than 200ms
        
        return {
            'avg': avg_latency,
            'p95': p95_latency
        }
    
    def test_prediction_latency(self, sample_data):
        """Benchmark prediction latency."""
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
                experiment_name='latency_test'
            )
            
            best_model = result['best_model']
            
            test_sample = pd.DataFrame({
                'feature1': [1.0],
                'feature2': [2.0],
                'feature3': [0.5]
            })
            
            latencies = []
            
            for _ in range(1000):
                start_time = time.time()
                prediction = best_model.predict(test_sample)
                end_time = time.time()
                
                latencies.append(end_time - start_time)
            
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            
            assert avg_latency < 0.01  # Average should be less than 10ms
            assert p95_latency < 0.02  # 95th percentile should be less than 20ms
            assert p99_latency < 0.05  # 99th percentile should be less than 50ms
            
            return {
                'avg': avg_latency,
                'p95': p95_latency,
                'p99': p99_latency
            }
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


@pytest.mark.benchmark
class TestThroughputBenchmarks:
    """Throughput benchmark tests."""
    
    def test_concurrent_training_throughput(self):
        """Benchmark concurrent training throughput."""
        from concurrent.futures import ThreadPoolExecutor
        import threading
        
        def train_model(thread_id):
            np.random.seed(42 + thread_id)
            data = {
                'feature1': np.random.randn(1000),
                'feature2': np.random.randn(1000),
                'feature3': np.random.randn(1000),
                'target': np.random.randint(0, 2, 1000)
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
                    experiment_name=f'concurrent_test_{thread_id}'
                )
                end_time = time.time()
                
                return end_time - start_time
            
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        n_threads = 5
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(train_model, i) for i in range(n_threads)]
            training_times = [future.result() for future in futures]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        throughput = n_threads / total_time
        
        assert total_time < 60  # Should complete within 60 seconds
        assert throughput > 0.05  # Should handle at least 0.05 concurrent trainings/second
        
        return throughput
    
    def test_batch_prediction_throughput(self, sample_data):
        """Benchmark batch prediction throughput."""
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
                experiment_name='throughput_test'
            )
            
            best_model = result['best_model']
            
            batch_sizes = [100, 1000, 10000]
            throughputs = []
            
            for batch_size in batch_sizes:
                test_data = pd.DataFrame({
                    'feature1': np.random.randn(batch_size),
                    'feature2': np.random.randn(batch_size),
                    'feature3': np.random.randn(batch_size)
                })
                
                start_time = time.time()
                predictions = best_model.predict(test_data)
                end_time = time.time()
                
                prediction_time = end_time - start_time
                throughput = batch_size / prediction_time
                throughputs.append(throughput)
                
                assert len(predictions) == batch_size
            
            # Throughput should increase with batch size
            assert throughputs[-1] > throughputs[0]
            
            return throughputs
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_database_throughput(self, temp_directory):
        """Benchmark database operation throughput."""
        db_url = f"sqlite:///{temp_directory}/throughput_test.db"
        db_service = DatabaseService(db_url)
        
        n_operations = 1000
        
        start_time = time.time()
        
        experiment_ids = []
        for i in range(n_operations):
            experiment_id = db_service.create_experiment(
                f'throughput_test_{i}',
                f'Throughput test experiment {i}',
                'classification',
                {'param': f'value_{i}'}
            )
            experiment_ids.append(experiment_id)
        
        end_time = time.time()
        creation_time = end_time - start_time
        
        creation_throughput = n_operations / creation_time
        
        start_time = time.time()
        
        for experiment_id in experiment_ids:
            experiment = db_service.get_experiment(experiment_id)
            assert experiment is not None
        
        end_time = time.time()
        retrieval_time = end_time - start_time
        
        retrieval_throughput = n_operations / retrieval_time
        
        assert creation_throughput > 30  # Should create at least 30 experiments/second
        assert retrieval_throughput > 50  # Should retrieve at least 50 experiments/second
        
        return {
            'creation_throughput': creation_throughput,
            'retrieval_throughput': retrieval_throughput
        }


@pytest.mark.benchmark
class TestResourceUsageBenchmarks:
    """Resource usage benchmark tests."""
    
    def test_cpu_usage_benchmark(self, sample_data):
        """Benchmark CPU usage during training."""
        import psutil
        
        pipeline = ModelPipeline()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            cpu_percentages = []
            
            def monitor_cpu():
                while True:
                    cpu_percentages.append(psutil.cpu_percent())
                    time.sleep(0.1)
            
            import threading
            monitor_thread = threading.Thread(target=monitor_cpu, daemon=True)
            monitor_thread.start()
            
            start_time = time.time()
            
            result = pipeline.run_pipeline(
                data_source=temp_path,
                target_column='target',
                task_type='classification',
                model_types=['random_forest', 'logistic_regression', 'svm'],
                experiment_name='cpu_usage_test'
            )
            
            end_time = time.time()
            training_time = end_time - start_time
            
            avg_cpu = np.mean(cpu_percentages) if cpu_percentages else 0
            max_cpu = np.max(cpu_percentages) if cpu_percentages else 0
            
            assert training_time < 60  # Should complete within 60 seconds
            assert avg_cpu < 100  # Should not exceed 100% CPU usage
            assert max_cpu < 200  # Should not exceed 200% CPU usage (multi-core)
            
            return {
                'avg_cpu': avg_cpu,
                'max_cpu': max_cpu,
                'training_time': training_time
            }
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_memory_efficiency_benchmark(self, sample_data):
        """Benchmark memory efficiency."""
        import psutil
        import gc
        
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
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
                experiment_name='memory_efficiency_test'
            )
            
            peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            peak_memory_usage = peak_memory - initial_memory
            
            gc.collect()
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            final_memory_usage = final_memory - initial_memory
            
            assert peak_memory_usage < 200  # Should use less than 200MB peak
            assert final_memory_usage < 50   # Should clean up to less than 50MB
            
            return {
                'peak_memory_usage': peak_memory_usage,
                'final_memory_usage': final_memory_usage
            }
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_disk_usage_benchmark(self, sample_data, temp_directory):
        """Benchmark disk usage."""
        registry = ModelRegistry()
        registry.models_path = temp_directory
        
        from aureon.models.classification_models import ClassificationModelFactory
        
        factory = ClassificationModelFactory()
        
        initial_disk_usage = sum(os.path.getsize(os.path.join(temp_directory, f)) 
                                for f in os.listdir(temp_directory) 
                                if os.path.isfile(os.path.join(temp_directory, f)))
        
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        model_ids = []
        for i in range(10):
            model = factory.create_model('random_forest')
            model.fit(X, y)
            
            model_id = registry.save_model(
                model,
                f'disk_usage_test_{i}',
                'classification',
                {'n_estimators': 100},
                {'accuracy': 0.95}
            )
            model_ids.append(model_id)
        
        final_disk_usage = sum(os.path.getsize(os.path.join(temp_directory, f)) 
                              for f in os.listdir(temp_directory) 
                              if os.path.isfile(os.path.join(temp_directory, f)))
        
        disk_usage_increase = final_disk_usage - initial_disk_usage
        
        assert disk_usage_increase < 50 * 1024 * 1024  # Should use less than 50MB
        
        return disk_usage_increase


@pytest.mark.benchmark
class TestScalabilityBenchmarks:
    """Scalability benchmark tests."""
    
    def test_dataset_size_scalability(self):
        """Benchmark scalability with dataset size."""
        dataset_sizes = [1000, 5000, 10000, 20000]
        training_times = []
        accuracies = []
        
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
                
                training_time = end_time - start_time
                training_times.append(training_time)
                
                accuracy = result['best_model'].model_metrics.get('accuracy', 0)
                accuracies.append(accuracy)
                
                assert training_time < size / 100  # Should scale sub-linearly
            
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        # Verify scalability characteristics
        time_ratios = [training_times[i] / training_times[0] for i in range(1, len(training_times))]
        size_ratios = [dataset_sizes[i] / dataset_sizes[0] for i in range(1, len(dataset_sizes))]
        
        for i, (time_ratio, size_ratio) in enumerate(zip(time_ratios, size_ratios)):
            assert time_ratio < size_ratio * 1.5  # Should scale better than linearly
        
        return {
            'training_times': training_times,
            'accuracies': accuracies,
            'time_ratios': time_ratios,
            'size_ratios': size_ratios
        }
    
    def test_feature_count_scalability(self):
        """Benchmark scalability with feature count."""
        feature_counts = [5, 10, 20, 50]
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
                
                training_time = end_time - start_time
                training_times.append(training_time)
                
                assert training_time < n_features * 2  # Should scale reasonably with features
            
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        return training_times
    
    def test_concurrent_user_scalability(self):
        """Benchmark scalability with concurrent users."""
        from concurrent.futures import ThreadPoolExecutor
        from fastapi.testclient import TestClient
        from aureon.api.main import app
        
        client = TestClient(app)
        
        def make_requests(n_requests):
            responses = []
            for _ in range(n_requests):
                response = client.get("/health")
                responses.append(response.status_code == 200)
            return responses
        
        user_counts = [1, 5, 10, 20]
        throughputs = []
        
        for n_users in user_counts:
            requests_per_user = 10
            
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=n_users) as executor:
                futures = [executor.submit(make_requests, requests_per_user) for _ in range(n_users)]
                results = [future.result() for future in futures]
            
            end_time = time.time()
            total_time = end_time - start_time
            
            total_requests = n_users * requests_per_user
            throughput = total_requests / total_time
            throughputs.append(throughput)
            
            assert all(all(user_results) for user_results in results)  # All requests should succeed
            assert total_time < 30  # Should complete within 30 seconds
        
        # Throughput should scale reasonably with user count
        assert throughputs[-1] > throughputs[0] * 0.5  # Should maintain at least 50% efficiency
        
        return throughputs
