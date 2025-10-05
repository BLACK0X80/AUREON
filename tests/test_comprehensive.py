import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

from aureon.data.cleaning import DataCleaner
from aureon.data.preprocessing import DataPreprocessor
from aureon.data.feature_engineering import FeatureEngineer
from aureon.data.split import DataSplitter
from aureon.models.classification_models import ClassificationModelFactory
from aureon.models.regression_models import RegressionModelFactory
from aureon.models.model_trainer import ModelTrainer
from aureon.models.model_registry import ModelRegistry
from aureon.pipeline.data_pipeline import DataPipeline
from aureon.pipeline.model_pipeline import ModelPipeline
from aureon.services.database import DatabaseService
from aureon.services.monitoring import SystemMonitor
from aureon.services.explainability import ExplainabilityService
from aureon.services.reporting import ReportingService


class TestDataCleaner:
    """Test cases for DataCleaner class."""
    
    def test_clean_data_basic(self, sample_data):
        """Test basic data cleaning functionality."""
        cleaner = DataCleaner()
        cleaned_data = cleaner.clean_data(sample_data)
        
        assert isinstance(cleaned_data, pd.DataFrame)
        assert len(cleaned_data) == len(sample_data)
        assert not cleaned_data.empty
    
    def test_handle_missing_values(self, sample_data_with_missing):
        """Test handling of missing values."""
        cleaner = DataCleaner()
        cleaned_data = cleaner.handle_missing_values(sample_data_with_missing, strategy='mean')
        
        assert not cleaned_data.isnull().any().any()
        assert len(cleaned_data) == len(sample_data_with_missing)
    
    def test_remove_duplicates(self, sample_data_with_duplicates):
        """Test duplicate removal."""
        cleaner = DataCleaner()
        cleaned_data = cleaner.remove_duplicates(sample_data_with_duplicates)
        
        assert len(cleaned_data) < len(sample_data_with_duplicates)
        assert not cleaned_data.duplicated().any()
    
    def test_outlier_detection(self, sample_data):
        """Test outlier detection."""
        cleaner = DataCleaner()
        outliers = cleaner.detect_outliers(sample_data, columns=['feature1', 'feature2'])
        
        assert isinstance(outliers, pd.DataFrame)
        assert 'is_outlier' in outliers.columns
    
    def test_data_validation(self, sample_data):
        """Test data validation."""
        cleaner = DataCleaner()
        validation_result = cleaner.validate_data(sample_data)
        
        assert isinstance(validation_result, dict)
        assert 'is_valid' in validation_result
        assert 'issues' in validation_result


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    def test_preprocess_data(self, sample_data):
        """Test data preprocessing."""
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.preprocess_data(sample_data, target_column='target')
        
        assert isinstance(processed_data, dict)
        assert 'X_train' in processed_data
        assert 'X_test' in processed_data
        assert 'y_train' in processed_data
        assert 'y_test' in processed_data
    
    def test_encode_categorical_features(self, sample_data):
        """Test categorical feature encoding."""
        preprocessor = DataPreprocessor()
        encoded_data = preprocessor.encode_categorical_features(sample_data, ['category'])
        
        assert isinstance(encoded_data, pd.DataFrame)
        assert 'category' not in encoded_data.columns or encoded_data['category'].dtype != 'object'
    
    def test_scale_features(self, sample_data):
        """Test feature scaling."""
        preprocessor = DataPreprocessor()
        scaled_data = preprocessor.scale_features(sample_data, ['feature1', 'feature2', 'feature3'])
        
        assert isinstance(scaled_data, pd.DataFrame)
        assert scaled_data[['feature1', 'feature2', 'feature3']].std().mean() < 2.0
    
    def test_feature_selection(self, sample_data):
        """Test feature selection."""
        preprocessor = DataPreprocessor()
        selected_features = preprocessor.select_features(
            sample_data.drop('target', axis=1), 
            sample_data['target'], 
            method='mutual_info'
        )
        
        assert isinstance(selected_features, list)
        assert len(selected_features) > 0


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class."""
    
    def test_create_polynomial_features(self, sample_data):
        """Test polynomial feature creation."""
        engineer = FeatureEngineer()
        poly_features = engineer.create_polynomial_features(
            sample_data[['feature1', 'feature2']], 
            degree=2
        )
        
        assert isinstance(poly_features, pd.DataFrame)
        assert poly_features.shape[1] > 2
    
    def test_create_interaction_features(self, sample_data):
        """Test interaction feature creation."""
        engineer = FeatureEngineer()
        interaction_features = engineer.create_interaction_features(
            sample_data[['feature1', 'feature2']]
        )
        
        assert isinstance(interaction_features, pd.DataFrame)
        assert 'feature1_x_feature2' in interaction_features.columns
    
    def test_create_time_features(self):
        """Test time-based feature creation."""
        engineer = FeatureEngineer()
        time_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='D'),
            'value': range(10)
        })
        
        time_features = engineer.create_time_features(time_data, 'timestamp')
        
        assert isinstance(time_features, pd.DataFrame)
        assert 'hour' in time_features.columns or 'day' in time_features.columns


class TestDataSplitter:
    """Test cases for DataSplitter class."""
    
    def test_split_data(self, sample_data):
        """Test data splitting."""
        splitter = DataSplitter()
        splits = splitter.split_data(sample_data, target_column='target', test_size=0.2)
        
        assert isinstance(splits, dict)
        assert 'X_train' in splits
        assert 'X_test' in splits
        assert 'y_train' in splits
        assert 'y_test' in splits
        
        total_samples = len(splits['X_train']) + len(splits['X_test'])
        assert total_samples == len(sample_data)
    
    def test_stratified_split(self, sample_data):
        """Test stratified data splitting."""
        splitter = DataSplitter()
        splits = splitter.split_data(
            sample_data, 
            target_column='target', 
            test_size=0.2, 
            stratify=True
        )
        
        train_prop = splits['y_train'].mean()
        test_prop = splits['y_test'].mean()
        
        assert abs(train_prop - test_prop) < 0.1
    
    def test_cross_validation_split(self, sample_data):
        """Test cross-validation splitting."""
        splitter = DataSplitter()
        cv_splits = splitter.cross_validation_split(
            sample_data.drop('target', axis=1),
            sample_data['target'],
            n_splits=5
        )
        
        assert len(cv_splits) == 5
        for split in cv_splits:
            assert 'train' in split
            assert 'test' in split


class TestClassificationModelFactory:
    """Test cases for ClassificationModelFactory class."""
    
    def test_create_random_forest(self):
        """Test Random Forest model creation."""
        factory = ClassificationModelFactory()
        model = factory.create_model('random_forest')
        
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    def test_create_logistic_regression(self):
        """Test Logistic Regression model creation."""
        factory = ClassificationModelFactory()
        model = factory.create_model('logistic_regression')
        
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    def test_create_svm(self):
        """Test SVM model creation."""
        factory = ClassificationModelFactory()
        model = factory.create_model('svm')
        
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
    
    def test_invalid_model_type(self):
        """Test invalid model type handling."""
        factory = ClassificationModelFactory()
        
        with pytest.raises(ValueError):
            factory.create_model('invalid_model')


class TestRegressionModelFactory:
    """Test cases for RegressionModelFactory class."""
    
    def test_create_linear_regression(self):
        """Test Linear Regression model creation."""
        factory = RegressionModelFactory()
        model = factory.create_model('linear_regression')
        
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
    
    def test_create_random_forest_regressor(self):
        """Test Random Forest Regressor model creation."""
        factory = RegressionModelFactory()
        model = factory.create_model('random_forest')
        
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
    
    def test_create_gradient_boosting(self):
        """Test Gradient Boosting model creation."""
        factory = RegressionModelFactory()
        model = factory.create_model('gradient_boosting')
        
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')


class TestModelTrainer:
    """Test cases for ModelTrainer class."""
    
    def test_train_classification_model(self, sample_data):
        """Test classification model training."""
        trainer = ModelTrainer()
        
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        model_info = trainer.train_model(
            X, y, 
            model_type='random_forest',
            task_type='classification'
        )
        
        assert isinstance(model_info, dict)
        assert 'model' in model_info
        assert 'metrics' in model_info
        assert 'training_time' in model_info
    
    def test_train_regression_model(self, sample_regression_data):
        """Test regression model training."""
        trainer = ModelTrainer()
        
        X = sample_regression_data.drop('target', axis=1)
        y = sample_regression_data['target']
        
        model_info = trainer.train_model(
            X, y,
            model_type='linear_regression',
            task_type='regression'
        )
        
        assert isinstance(model_info, dict)
        assert 'model' in model_info
        assert 'metrics' in model_info
    
    def test_hyperparameter_tuning(self, sample_data):
        """Test hyperparameter tuning."""
        trainer = ModelTrainer()
        
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        best_model = trainer.tune_hyperparameters(
            X, y,
            model_type='random_forest',
            task_type='classification',
            param_grid={'n_estimators': [10, 50], 'max_depth': [3, 5]}
        )
        
        assert best_model is not None
        assert hasattr(best_model, 'fit')
        assert hasattr(best_model, 'predict')


class TestModelRegistry:
    """Test cases for ModelRegistry class."""
    
    def test_save_model(self, mock_model, temp_directory):
        """Test model saving."""
        registry = ModelRegistry()
        registry.models_path = temp_directory
        
        model_id = registry.save_model(mock_model, 'test_model', 'classification')
        
        assert isinstance(model_id, int)
        assert model_id > 0
    
    def test_load_model(self, mock_model, temp_directory):
        """Test model loading."""
        registry = ModelRegistry()
        registry.models_path = temp_directory
        
        model_id = registry.save_model(mock_model, 'test_model', 'classification')
        loaded_model = registry.load_model(model_id)
        
        assert loaded_model is not None
        assert hasattr(loaded_model, 'predict')
    
    def test_list_models(self, temp_directory):
        """Test model listing."""
        registry = ModelRegistry()
        registry.models_path = temp_directory
        
        models = registry.list_models()
        
        assert isinstance(models, list)
    
    def test_delete_model(self, mock_model, temp_directory):
        """Test model deletion."""
        registry = ModelRegistry()
        registry.models_path = temp_directory
        
        model_id = registry.save_model(mock_model, 'test_model', 'classification')
        result = registry.delete_model(model_id)
        
        assert result is True


class TestDataPipeline:
    """Test cases for DataPipeline class."""
    
    def test_run_pipeline(self, sample_data, temp_csv_file):
        """Test complete data pipeline execution."""
        pipeline = DataPipeline()
        
        result = pipeline.run_pipeline(
            data_source=temp_csv_file,
            target_column='target',
            task_type='classification'
        )
        
        assert isinstance(result, dict)
        assert 'processed_data' in result
        assert 'preprocessing_info' in result
    
    def test_pipeline_with_cleaning(self, sample_data_with_missing, temp_directory):
        """Test pipeline with data cleaning."""
        pipeline = DataPipeline()
        
        temp_file = os.path.join(temp_directory, 'test.csv')
        sample_data_with_missing.to_csv(temp_file, index=False)
        
        result = pipeline.run_pipeline(
            data_source=temp_file,
            target_column='target',
            task_type='classification',
            cleaning_config={'handle_missing': True, 'remove_duplicates': True}
        )
        
        assert isinstance(result, dict)
        assert 'processed_data' in result


class TestModelPipeline:
    """Test cases for ModelPipeline class."""
    
    def test_run_model_pipeline(self, sample_data, temp_csv_file):
        """Test complete model pipeline execution."""
        pipeline = ModelPipeline()
        
        result = pipeline.run_pipeline(
            data_source=temp_csv_file,
            target_column='target',
            task_type='classification',
            model_types=['random_forest', 'logistic_regression']
        )
        
        assert isinstance(result, dict)
        assert 'models' in result
        assert 'best_model' in result
        assert 'experiment_id' in result
    
    def test_pipeline_with_hyperparameter_tuning(self, sample_data, temp_csv_file):
        """Test pipeline with hyperparameter tuning."""
        pipeline = ModelPipeline()
        
        result = pipeline.run_pipeline(
            data_source=temp_csv_file,
            target_column='target',
            task_type='classification',
            model_types=['random_forest'],
            hyperparameter_tuning=True
        )
        
        assert isinstance(result, dict)
        assert 'models' in result


class TestDatabaseService:
    """Test cases for DatabaseService class."""
    
    def test_create_experiment(self, temp_directory):
        """Test experiment creation."""
        db_url = f"sqlite:///{temp_directory}/test.db"
        db_service = DatabaseService(db_url)
        
        experiment_id = db_service.create_experiment(
            'test_experiment',
            'Test experiment description',
            'classification',
            {'param1': 'value1'}
        )
        
        assert isinstance(experiment_id, int)
        assert experiment_id > 0
    
    def test_get_experiment(self, temp_directory):
        """Test experiment retrieval."""
        db_url = f"sqlite:///{temp_directory}/test.db"
        db_service = DatabaseService(db_url)
        
        experiment_id = db_service.create_experiment(
            'test_experiment',
            'Test experiment description',
            'classification',
            {'param1': 'value1'}
        )
        
        experiment = db_service.get_experiment(experiment_id)
        
        assert experiment is not None
        assert experiment['name'] == 'test_experiment'
        assert experiment['task_type'] == 'classification'
    
    def test_list_experiments(self, temp_directory):
        """Test experiment listing."""
        db_url = f"sqlite:///{temp_directory}/test.db"
        db_service = DatabaseService(db_url)
        
        db_service.create_experiment(
            'test_experiment1',
            'Test experiment 1',
            'classification',
            {}
        )
        
        db_service.create_experiment(
            'test_experiment2',
            'Test experiment 2',
            'regression',
            {}
        )
        
        experiments = db_service.list_experiments()
        
        assert len(experiments) == 2
        assert all('name' in exp for exp in experiments)
    
    def test_create_dataset(self, temp_directory):
        """Test dataset creation."""
        db_url = f"sqlite:///{temp_directory}/test.db"
        db_service = DatabaseService(db_url)
        
        dataset_id = db_service.create_dataset(
            'test_dataset',
            '/path/to/data.csv',
            'csv',
            1000,
            100,
            5,
            'target'
        )
        
        assert isinstance(dataset_id, int)
        assert dataset_id > 0


class TestSystemMonitor:
    """Test cases for SystemMonitor class."""
    
    def test_get_system_metrics(self):
        """Test system metrics retrieval."""
        monitor = SystemMonitor()
        metrics = monitor.get_system_metrics()
        
        assert isinstance(metrics, dict)
        assert 'cpu_percent' in metrics
        assert 'memory_percent' in metrics
        assert 'disk_percent' in metrics
    
    def test_check_system_health(self):
        """Test system health check."""
        monitor = SystemMonitor()
        health = monitor.check_system_health()
        
        assert isinstance(health, dict)
        assert 'status' in health
        assert 'timestamp' in health
    
    def test_monitor_model_performance(self, mock_model):
        """Test model performance monitoring."""
        monitor = SystemMonitor()
        
        performance = monitor.monitor_model_performance(
            mock_model,
            pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [2, 4, 6]}),
            pd.Series([0, 1, 0])
        )
        
        assert isinstance(performance, dict)
        assert 'performance_score' in performance


class TestExplainabilityService:
    """Test cases for ExplainabilityService class."""
    
    def test_generate_shap_explanation(self, mock_model, sample_data):
        """Test SHAP explanation generation."""
        service = ExplainabilityService()
        
        explanation = service.generate_explanation(
            mock_model,
            sample_data.drop('target', axis=1).iloc[:5],
            method='shap'
        )
        
        assert isinstance(explanation, dict)
        assert 'feature_importance' in explanation
    
    def test_generate_lime_explanation(self, mock_model, sample_data):
        """Test LIME explanation generation."""
        service = ExplainabilityService()
        
        explanation = service.generate_explanation(
            mock_model,
            sample_data.drop('target', axis=1).iloc[:5],
            method='lime'
        )
        
        assert isinstance(explanation, dict)
        assert 'explanation' in explanation


class TestReportingService:
    """Test cases for ReportingService class."""
    
    def test_generate_model_report(self, mock_model, sample_data):
        """Test model report generation."""
        service = ReportingService()
        
        report = service.generate_model_report(
            mock_model,
            sample_data,
            'target',
            output_format='html'
        )
        
        assert isinstance(report, str)
        assert len(report) > 0
    
    def test_generate_experiment_report(self, mock_registry):
        """Test experiment report generation."""
        service = ReportingService()
        
        report = service.generate_experiment_report(
            mock_registry.list_models(),
            output_format='pdf'
        )
        
        assert isinstance(report, str)
        assert len(report) > 0


@pytest.mark.integration
class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_classification_pipeline(self, sample_data, temp_directory):
        """Test complete end-to-end classification pipeline."""
        temp_file = os.path.join(temp_directory, 'test.csv')
        sample_data.to_csv(temp_file, index=False)
        
        db_url = f"sqlite:///{temp_directory}/test.db"
        db_service = DatabaseService(db_url)
        
        pipeline = ModelPipeline()
        
        result = pipeline.run_pipeline(
            data_source=temp_file,
            target_column='target',
            task_type='classification',
            model_types=['random_forest'],
            experiment_name='integration_test'
        )
        
        assert isinstance(result, dict)
        assert 'models' in result
        assert 'best_model' in result
        assert 'experiment_id' in result
        
        experiment = db_service.get_experiment(result['experiment_id'])
        assert experiment is not None
        assert experiment['name'] == 'integration_test'
    
    def test_model_prediction_workflow(self, sample_data, temp_directory):
        """Test complete model prediction workflow."""
        temp_file = os.path.join(temp_directory, 'test.csv')
        sample_data.to_csv(temp_file, index=False)
        
        pipeline = ModelPipeline()
        
        training_result = pipeline.run_pipeline(
            data_source=temp_file,
            target_column='target',
            task_type='classification',
            model_types=['random_forest']
        )
        
        best_model = training_result['best_model']
        
        test_data = sample_data.drop('target', axis=1).iloc[:3]
        predictions = best_model.predict(test_data)
        
        assert len(predictions) == 3
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_model_retraining_workflow(self, sample_data, temp_directory):
        """Test model retraining workflow."""
        temp_file = os.path.join(temp_directory, 'test.csv')
        sample_data.to_csv(temp_file, index=False)
        
        pipeline = ModelPipeline()
        
        initial_result = pipeline.run_pipeline(
            data_source=temp_file,
            target_column='target',
            task_type='classification',
            model_types=['random_forest']
        )
        
        initial_model = initial_result['best_model']
        initial_performance = initial_model.model_metrics.get('accuracy', 0)
        
        retrain_result = pipeline.retrain_model(
            model_id=initial_result['best_model'].model_id,
            new_data_source=temp_file,
            target_column='target'
        )
        
        assert isinstance(retrain_result, dict)
        assert 'new_model' in retrain_result
        assert 'performance_comparison' in retrain_result
