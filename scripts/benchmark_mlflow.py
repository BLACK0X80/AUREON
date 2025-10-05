#!/usr/bin/env python3
"""
Benchmark script to compare AUREON performance with MLflow.
This script runs comprehensive benchmarks and generates comparison reports.
"""

import argparse
import json
import time
import tempfile
import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aureon.pipeline.model_pipeline import ModelPipeline
from aureon.models.model_registry import ModelRegistry
from aureon.services.database import DatabaseService


def create_benchmark_dataset(n_samples=10000, n_features=20, random_state=42):
    """Create a benchmark dataset for testing."""
    np.random.seed(random_state)
    
    data = {
        f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)
    }
    data['target'] = np.random.randint(0, 2, n_samples)
    
    return pd.DataFrame(data)


def benchmark_aureon_training(dataset, model_types=['random_forest'], n_runs=3):
    """Benchmark AUREON training performance."""
    results = []
    
    for run in range(n_runs):
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
                model_types=model_types,
                experiment_name=f'aureon_benchmark_run_{run}'
            )
            end_time = time.time()
            
            training_time = end_time - start_time
            accuracy = result['best_model'].model_metrics.get('accuracy', 0)
            
            results.append({
                'run': run,
                'training_time': training_time,
                'accuracy': accuracy,
                'model_type': result['best_model'].model_name
            })
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    return results


def benchmark_aureon_prediction(dataset, model_types=['random_forest'], n_predictions=1000):
    """Benchmark AUREON prediction performance."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        dataset.to_csv(f.name, index=False)
        temp_path = f.name
    
    try:
        pipeline = ModelPipeline()
        
        training_result = pipeline.run_pipeline(
            data_source=temp_path,
            target_column='target',
            task_type='classification',
            model_types=model_types,
            experiment_name='aureon_prediction_benchmark'
        )
        
        best_model = training_result['best_model']
        
        test_data = dataset.drop('target', axis=1).iloc[:n_predictions]
        
        start_time = time.time()
        predictions = best_model.predict(test_data)
        end_time = time.time()
        
        prediction_time = end_time - start_time
        throughput = n_predictions / prediction_time
        
        return {
            'prediction_time': prediction_time,
            'throughput': throughput,
            'n_predictions': n_predictions
        }
    
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def benchmark_aureon_experiment_tracking(dataset, n_experiments=100):
    """Benchmark AUREON experiment tracking performance."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_url = f"sqlite:///{temp_dir}/benchmark.db"
        db_service = DatabaseService(db_url)
        
        start_time = time.time()
        
        experiment_ids = []
        for i in range(n_experiments):
            experiment_id = db_service.create_experiment(
                f'benchmark_experiment_{i}',
                f'Benchmark experiment {i}',
                'classification',
                {'param1': f'value_{i}'}
            )
            experiment_ids.append(experiment_id)
        
        end_time = time.time()
        creation_time = end_time - start_time
        
        start_time = time.time()
        
        experiments = []
        for experiment_id in experiment_ids:
            experiment = db_service.get_experiment(experiment_id)
            experiments.append(experiment)
        
        end_time = time.time()
        retrieval_time = end_time - start_time
        
        return {
            'creation_time': creation_time,
            'retrieval_time': retrieval_time,
            'creation_throughput': n_experiments / creation_time,
            'retrieval_throughput': n_experiments / retrieval_time,
            'n_experiments': n_experiments
        }


def benchmark_aureon_model_registry(dataset, n_models=50):
    """Benchmark AUREON model registry performance."""
    with tempfile.TemporaryDirectory() as temp_dir:
        registry = ModelRegistry()
        registry.models_path = temp_dir
        
        from aureon.models.classification_models import ClassificationModelFactory
        
        factory = ClassificationModelFactory()
        
        start_time = time.time()
        
        model_ids = []
        for i in range(n_models):
            model = factory.create_model('random_forest')
            
            X = dataset.drop('target', axis=1)
            y = dataset['target']
            
            model.fit(X, y)
            
            model_id = registry.save_model(
                model,
                f'benchmark_model_{i}',
                'classification',
                {'n_estimators': 100},
                {'accuracy': 0.95}
            )
            model_ids.append(model_id)
        
        end_time = time.time()
        save_time = end_time - start_time
        
        start_time = time.time()
        
        loaded_models = []
        for model_id in model_ids:
            loaded_model = registry.load_model(model_id)
            loaded_models.append(loaded_model)
        
        end_time = time.time()
        load_time = end_time - start_time
        
        return {
            'save_time': save_time,
            'load_time': load_time,
            'save_throughput': n_models / save_time,
            'load_throughput': n_models / load_time,
            'n_models': n_models
        }


def benchmark_memory_usage(dataset, model_types=['random_forest']):
    """Benchmark memory usage during training."""
    import psutil
    import gc
    
    gc.collect()
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        dataset.to_csv(f.name, index=False)
        temp_path = f.name
    
    try:
        pipeline = ModelPipeline()
        
        result = pipeline.run_pipeline(
            data_source=temp_path,
            target_column='target',
            task_type='classification',
            model_types=model_types,
            experiment_name='memory_benchmark'
        )
        
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        peak_memory_usage = peak_memory - initial_memory
        
        gc.collect()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        final_memory_usage = final_memory - initial_memory
        
        return {
            'peak_memory_usage': peak_memory_usage,
            'final_memory_usage': final_memory_usage,
            'memory_cleanup': peak_memory_usage - final_memory_usage
        }
    
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def run_comprehensive_benchmark(dataset_sizes=[1000, 5000, 10000], 
                               feature_counts=[5, 10, 20],
                               model_types=['random_forest', 'logistic_regression']):
    """Run comprehensive benchmark across different configurations."""
    results = {
        'training_performance': {},
        'prediction_performance': {},
        'experiment_tracking': {},
        'model_registry': {},
        'memory_usage': {},
        'scalability': {}
    }
    
    print("Running comprehensive AUREON benchmarks...")
    
    # Training performance benchmarks
    print("Benchmarking training performance...")
    for size in dataset_sizes:
        dataset = create_benchmark_dataset(n_samples=size)
        training_results = benchmark_aureon_training(dataset, model_types)
        
        avg_training_time = np.mean([r['training_time'] for r in training_results])
        avg_accuracy = np.mean([r['accuracy'] for r in training_results])
        
        results['training_performance'][f'dataset_size_{size}'] = {
            'avg_training_time': avg_training_time,
            'avg_accuracy': avg_accuracy,
            'throughput': size / avg_training_time
        }
    
    # Prediction performance benchmarks
    print("Benchmarking prediction performance...")
    dataset = create_benchmark_dataset()
    prediction_results = benchmark_aureon_prediction(dataset)
    results['prediction_performance'] = prediction_results
    
    # Experiment tracking benchmarks
    print("Benchmarking experiment tracking...")
    tracking_results = benchmark_aureon_experiment_tracking(dataset)
    results['experiment_tracking'] = tracking_results
    
    # Model registry benchmarks
    print("Benchmarking model registry...")
    registry_results = benchmark_aureon_model_registry(dataset)
    results['model_registry'] = registry_results
    
    # Memory usage benchmarks
    print("Benchmarking memory usage...")
    memory_results = benchmark_memory_usage(dataset)
    results['memory_usage'] = memory_results
    
    # Scalability benchmarks
    print("Benchmarking scalability...")
    scalability_results = {}
    
    for n_features in feature_counts:
        dataset = create_benchmark_dataset(n_features=n_features)
        training_results = benchmark_aureon_training(dataset, ['random_forest'])
        
        avg_training_time = np.mean([r['training_time'] for r in training_results])
        
        scalability_results[f'features_{n_features}'] = {
            'avg_training_time': avg_training_time,
            'throughput': 5000 / avg_training_time
        }
    
    results['scalability'] = scalability_results
    
    return results


def generate_report(results, output_file):
    """Generate a comprehensive benchmark report."""
    report = {
        'benchmark_info': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'version': '2.0.0',
            'description': 'AUREON Performance Benchmark Report'
        },
        'results': results,
        'summary': {
            'training_performance': {
                'best_throughput': max(
                    results['training_performance'][k]['throughput'] 
                    for k in results['training_performance']
                ),
                'avg_accuracy': np.mean([
                    results['training_performance'][k]['avg_accuracy'] 
                    for k in results['training_performance']
                ])
            },
            'prediction_performance': {
                'throughput': results['prediction_performance']['throughput'],
                'latency': results['prediction_performance']['prediction_time']
            },
            'experiment_tracking': {
                'creation_throughput': results['experiment_tracking']['creation_throughput'],
                'retrieval_throughput': results['experiment_tracking']['retrieval_throughput']
            },
            'model_registry': {
                'save_throughput': results['model_registry']['save_throughput'],
                'load_throughput': results['model_registry']['load_throughput']
            },
            'memory_usage': {
                'peak_memory_mb': results['memory_usage']['peak_memory_usage'],
                'final_memory_mb': results['memory_usage']['final_memory_usage']
            }
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Benchmark report saved to {output_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    
    print(f"Training Performance:")
    print(f"  Best Throughput: {report['summary']['training_performance']['best_throughput']:.2f} samples/sec")
    print(f"  Average Accuracy: {report['summary']['training_performance']['avg_accuracy']:.3f}")
    
    print(f"\nPrediction Performance:")
    print(f"  Throughput: {report['summary']['prediction_performance']['throughput']:.2f} predictions/sec")
    print(f"  Latency: {report['summary']['prediction_performance']['latency']:.4f} seconds")
    
    print(f"\nExperiment Tracking:")
    print(f"  Creation Throughput: {report['summary']['experiment_tracking']['creation_throughput']:.2f} experiments/sec")
    print(f"  Retrieval Throughput: {report['summary']['experiment_tracking']['retrieval_throughput']:.2f} experiments/sec")
    
    print(f"\nModel Registry:")
    print(f"  Save Throughput: {report['summary']['model_registry']['save_throughput']:.2f} models/sec")
    print(f"  Load Throughput: {report['summary']['model_registry']['load_throughput']:.2f} models/sec")
    
    print(f"\nMemory Usage:")
    print(f"  Peak Memory: {report['summary']['memory_usage']['peak_memory_mb']:.2f} MB")
    print(f"  Final Memory: {report['summary']['memory_usage']['final_memory_mb']:.2f} MB")
    
    print("="*50)


def main():
    """Main function to run benchmarks."""
    parser = argparse.ArgumentParser(description='Run AUREON performance benchmarks')
    parser.add_argument('--output', '-o', default='aureon_benchmark_report.json',
                       help='Output file for benchmark report')
    parser.add_argument('--dataset-sizes', nargs='+', type=int, 
                       default=[1000, 5000, 10000],
                       help='Dataset sizes to test')
    parser.add_argument('--feature-counts', nargs='+', type=int,
                       default=[5, 10, 20],
                       help='Feature counts to test')
    parser.add_argument('--model-types', nargs='+', 
                       default=['random_forest', 'logistic_regression'],
                       help='Model types to test')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick benchmark with smaller datasets')
    
    args = parser.parse_args()
    
    if args.quick:
        dataset_sizes = [1000, 2000]
        feature_counts = [5, 10]
        print("Running quick benchmark...")
    else:
        dataset_sizes = args.dataset_sizes
        feature_counts = args.feature_counts
        print("Running comprehensive benchmark...")
    
    try:
        results = run_comprehensive_benchmark(
            dataset_sizes=dataset_sizes,
            feature_counts=feature_counts,
            model_types=args.model_types
        )
        
        generate_report(results, args.output)
        
        print(f"\nBenchmark completed successfully!")
        print(f"Report saved to: {args.output}")
        
    except Exception as e:
        print(f"Benchmark failed with error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
