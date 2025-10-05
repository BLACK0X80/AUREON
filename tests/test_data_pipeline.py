import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from aureon.data.ingestion import DataLoader
from aureon.data.cleaning import DataCleaner
from aureon.data.preprocessing import DataPreprocessor
from aureon.data.feature_engineering import FeatureEngineer
from aureon.data.split import DataSplitter
from aureon.pipeline.data_pipeline import DataPipeline


class TestDataLoader:
    def test_load_csv(self):
        loader = DataLoader()

        data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [10, 20, 30, 40, 50],
                "target": [0, 1, 0, 1, 0],
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            loaded_data = loader.load(temp_path, "csv")
            pd.testing.assert_frame_equal(data, loaded_data)
        finally:
            os.unlink(temp_path)

    def test_load_multiple_files(self):
        loader = DataLoader()

        data1 = pd.DataFrame({"feature1": [1, 2], "feature2": [10, 20]})
        data2 = pd.DataFrame({"feature1": [3, 4], "feature2": [30, 40]})

        temp_files = []
        try:
            for i, data in enumerate([data1, data2]):
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".csv", delete=False
                ) as f:
                    data.to_csv(f.name, index=False)
                    temp_files.append(f.name)

            combined_data = loader.load_multiple(temp_files, "csv")
            expected = pd.concat([data1, data2], ignore_index=True)
            pd.testing.assert_frame_equal(expected, combined_data)
        finally:
            for temp_file in temp_files:
                os.unlink(temp_file)

    def test_load_nonexistent_file(self):
        loader = DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent_file.csv")


class TestDataCleaner:
    def test_remove_duplicates(self):
        cleaner = DataCleaner()

        data = pd.DataFrame(
            {
                "feature1": [1, 2, 2, 3, 4],
                "feature2": [10, 20, 20, 30, 40],
                "target": [0, 1, 1, 0, 1],
            }
        )

        cleaned = cleaner.remove_duplicates(data)
        expected = data.drop_duplicates()
        pd.testing.assert_frame_equal(expected, cleaned)

    def test_handle_missing_values_mean(self):
        cleaner = DataCleaner()

        data = pd.DataFrame(
            {
                "feature1": [1, 2, np.nan, 4, 5],
                "feature2": [10, 20, 30, np.nan, 50],
                "target": [0, 1, 0, 1, 0],
            }
        )

        cleaned = cleaner.handle_missing_values(data, strategy="mean")

        assert not cleaned["feature1"].isnull().any()
        assert not cleaned["feature2"].isnull().any()
        assert cleaned["feature1"].mean() == 3.0
        assert cleaned["feature2"].mean() == 27.5

    def test_handle_missing_values_drop(self):
        cleaner = DataCleaner()

        data = pd.DataFrame(
            {
                "feature1": [1, 2, np.nan, 4, 5],
                "feature2": [10, 20, 30, 40, 50],
                "target": [0, 1, 0, 1, 0],
            }
        )

        cleaned = cleaner.handle_missing_values(data, strategy="drop")

        assert len(cleaned) == 4
        assert not cleaned.isnull().any().any()

    def test_encode_categorical_one_hot(self):
        cleaner = DataCleaner()

        data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "category": ["A", "B", "A", "C", "B"],
                "target": [0, 1, 0, 1, 0],
            }
        )

        encoded = cleaner.encode_categorical(data, ["category"], strategy="one_hot")

        assert "category_A" in encoded.columns
        assert "category_B" in encoded.columns
        assert "category_C" in encoded.columns
        assert "category" not in encoded.columns


class TestDataPreprocessor:
    def test_scale_features_standard(self):
        preprocessor = DataPreprocessor()

        data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [100, 200, 300, 400, 500],
                "target": [0, 1, 0, 1, 0],
            }
        )

        scaled = preprocessor.scale_features(data, strategy="standard")

        assert abs(scaled["feature1"].mean()) < 1e-10
        assert abs(scaled["feature2"].mean()) < 1e-10
        assert abs(scaled["feature1"].std() - 1.0) < 1e-10
        assert abs(scaled["feature2"].std() - 1.0) < 1e-10

    def test_scale_features_minmax(self):
        preprocessor = DataPreprocessor()

        data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [100, 200, 300, 400, 500],
                "target": [0, 1, 0, 1, 0],
            }
        )

        scaled = preprocessor.scale_features(data, strategy="minmax")

        assert scaled["feature1"].min() == 0.0
        assert scaled["feature1"].max() == 1.0
        assert scaled["feature2"].min() == 0.0
        assert scaled["feature2"].max() == 1.0

    def test_create_polynomial_features(self):
        preprocessor = DataPreprocessor()

        data = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "target": [0, 1, 0]}
        )

        poly_data = preprocessor.create_polynomial_features(data, degree=2)

        assert "feature1^2" in poly_data.columns
        assert "feature2^2" in poly_data.columns
        assert "feature1 feature2" in poly_data.columns


class TestFeatureEngineer:
    def test_create_date_features(self):
        engineer = FeatureEngineer()

        data = pd.DataFrame(
            {"date": ["2023-01-01", "2023-02-15", "2023-03-30"], "value": [10, 20, 30]}
        )

        engineered = engineer.create_date_features(data, "date")

        assert "date_year" in engineered.columns
        assert "date_month" in engineered.columns
        assert "date_day" in engineered.columns
        assert engineered["date_year"].iloc[0] == 2023

    def test_create_text_features(self):
        engineer = FeatureEngineer()

        data = pd.DataFrame(
            {
                "text": ["Hello World", "Python ML", "Data Science"],
                "value": [10, 20, 30],
            }
        )

        engineered = engineer.create_text_features(data, "text")

        assert "text_length" in engineered.columns
        assert "text_word_count" in engineered.columns
        assert engineered["text_length"].iloc[0] == 11
        assert engineered["text_word_count"].iloc[0] == 2

    def test_create_numerical_features(self):
        engineer = FeatureEngineer()

        data = pd.DataFrame(
            {
                "feature1": [1, 4, 9, 16],
                "feature2": [2, 8, 18, 32],
                "target": [0, 1, 0, 1],
            }
        )

        engineered = engineer.create_numerical_features(data, ["feature1", "feature2"])

        assert "feature1_log" in engineered.columns
        assert "feature1_sqrt" in engineered.columns
        assert "feature1_squared" in engineered.columns
        assert "feature2_log" in engineered.columns


class TestDataSplitter:
    def test_train_test_split(self):
        splitter = DataSplitter()

        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        X_train, X_test, y_train, y_test = splitter.train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        assert len(X_train) == 8
        assert len(X_test) == 2
        assert len(y_train) == 8
        assert len(y_test) == 2

    def test_train_val_test_split(self):
        splitter = DataSplitter()

        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        X_train, X_val, X_test, y_train, y_val, y_test = splitter.train_val_test_split(
            X, y, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42
        )

        assert len(X_train) == 6
        assert len(X_val) == 2
        assert len(X_test) == 2
        assert len(y_train) == 6
        assert len(y_val) == 2
        assert len(y_test) == 2


class TestDataPipeline:
    def test_run_pipeline(self):
        pipeline = DataPipeline()

        data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                "category": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
                "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            results = pipeline.run_pipeline(temp_path, "target", split_data=True)

            assert "raw_data" in results
            assert "processed_data" in results
            assert "X" in results
            assert "y" in results
            assert "splits" in results
            assert results["splits"] is not None

            X_train, X_test, y_train, y_test = results["splits"]
            assert len(X_train) > 0
            assert len(X_test) > 0
            assert len(y_train) > 0
            assert len(y_test) > 0

        finally:
            os.unlink(temp_path)

    def test_get_pipeline_stats(self):
        pipeline = DataPipeline()

        data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [10, 20, 30, 40, 50],
                "target": [0, 1, 0, 1, 0],
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            pipeline.run_pipeline(temp_path, "target")
            stats = pipeline.get_pipeline_stats()

            assert "raw_data_shape" in stats
            assert "processed_data_shape" in stats
            assert stats["raw_data_shape"] == (5, 3)

        finally:
            os.unlink(temp_path)
