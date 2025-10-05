#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import sys

sys.path.append(str(Path(__file__).parent.parent))

from aureon.pipeline.data_pipeline import DataPipeline
from aureon.pipeline.model_pipeline import ModelPipeline
from aureon.models.model_registry import ModelRegistry
from aureon.services.reporting import ReportGenerator
from aureon.services.monitoring import DriftDetector
from aureon.services.explainability import ModelInterpretability


def generate_demo_data():
    np.random.seed(42)

    n_samples = 1000
    X = np.random.randn(n_samples, 5)
    y = (X[:, 0] + X[:, 1] + np.random.normal(0, 0.1, n_samples) > 0).astype(int)

    df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(5)])
    df["target"] = y

    return df


def run_demo():
    print("🚀 AUREON Demo - AI/ML Pipeline Management System")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        print("\n1. Generating demo data...")
        demo_data = generate_demo_data()
        data_file = temp_path / "demo_data.csv"
        demo_data.to_csv(data_file, index=False)
        print(
            f"   Generated {len(demo_data)} samples with {len(demo_data.columns)-1} features"
        )

        print("\n2. Running data pipeline...")
        data_pipeline = DataPipeline()
        processed_data = data_pipeline.run_pipeline(str(data_file), "target")

        print(f"   Raw data shape: {processed_data['raw_data'].shape}")
        print(f"   Processed data shape: {processed_data['processed_data'].shape}")

        if processed_data["splits"]:
            X_train, X_test, y_train, y_test = processed_data["splits"]
            print(f"   Training set: {X_train.shape[0]} samples")
            print(f"   Test set: {X_test.shape[0]} samples")

        print("\n3. Training models...")
        model_pipeline = ModelPipeline("classification", "demo_experiment")
        model_pipeline.configure_training(
            {
                "model_types": ["random_forest", "logistic_regression"],
                "hyperparameter_search": {
                    "enabled": True,
                    "models": ["random_forest"],
                    "param_grids": {
                        "random_forest": {"n_estimators": [10, 50], "max_depth": [3, 5]}
                    },
                    "search_type": "grid",
                    "cv": 3,
                },
            }
        )

        results = model_pipeline.train_models(X_train, y_train, X_test, y_test)
        print(f"   Trained {len(results['trained_models'])} models")
        print(f"   Best model: {results['best_model']}")
        print(f"   Best score: {results['best_score']:.4f}")

        print("\n4. Registering models...")
        registry = ModelRegistry()
        model_ids = model_pipeline.register_all_models(
            version="1.0", description="Demo experiment models"
        )
        print(f"   Registered {len(model_ids)} models")

        print("\n5. Model evaluation...")
        best_model = registry.get_model(model_ids[0])
        metrics = best_model.evaluate(X_test, y_test, "classification")
        print("   Model metrics:")
        for metric, value in metrics.items():
            print(f"     {metric}: {value:.4f}")

        print("\n6. Drift detection...")
        drift_detector = DriftDetector()
        reference_data = X_train.copy()
        reference_data["target"] = y_train
        current_data = X_test.copy()
        current_data["target"] = y_test

        drift_results = drift_detector.detect_drift(reference_data, current_data)
        print(f"   Drift detected: {drift_results['drift_detected']}")
        print(f"   Drift score: {drift_results['drift_score']:.4f}")

        print("\n7. Model explainability...")
        interpreter = ModelInterpretability()
        analysis = interpreter.comprehensive_analysis(best_model, X_test, y_test)
        print(
            f"   Generated explanations for {len(analysis['sample_explanations'])} samples"
        )

        if (
            analysis["global_importance"]
            and "feature_importance" in analysis["global_importance"]
        ):
            importance = analysis["global_importance"]["feature_importance"]
            top_features = sorted(
                importance.items(), key=lambda x: abs(x[1]), reverse=True
            )[:3]
            print("   Top 3 most important features:")
            for feature, importance_val in top_features:
                print(f"     {feature}: {importance_val:.4f}")

        print("\n8. Generating report...")
        report_generator = ReportGenerator()
        model_report = report_generator.generate_model_report(best_model)
        report_file = temp_path / "demo_report.json"

        import json

        with open(report_file, "w") as f:
            json.dump(model_report, f, indent=2, default=str)
        print(f"   Report saved to: {report_file}")

        print("\n9. Model registry statistics...")
        stats = registry.get_model_statistics()
        print(f"   Total models: {stats['total_models']}")
        print(f"   Active models: {stats['active_models']}")
        print(f"   Models by task type: {stats['models_by_task_type']}")

        print("\n10. Listing all models...")
        models = registry.list_models()
        print(f"   Found {len(models)} models in registry:")
        for model in models:
            print(
                f"     ID: {model['id']}, Name: {model['model_name']}, "
                f"Type: {model['task_type']}, Created: {model['created_at'][:19]}"
            )

    print("\n" + "=" * 60)
    print("🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("1. Try the CLI: aureon --help")
    print("2. Start the API: aureon serve")
    print("3. Check the documentation: README.md")
    print("4. Run tests: python scripts/run_tests.py")


if __name__ == "__main__":
    run_demo()
