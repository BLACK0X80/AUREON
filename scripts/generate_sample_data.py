import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def generate_classification_data(n_samples=1000, n_features=10, n_classes=2, noise=0.1):
    np.random.seed(42)

    X = np.random.randn(n_samples, n_features)

    if n_classes == 2:
        y = (X[:, 0] + X[:, 1] + np.random.normal(0, noise, n_samples) > 0).astype(int)
    else:
        y = X[:, 0] + X[:, 1] + np.random.normal(0, noise, n_samples)
        y = pd.cut(y, bins=n_classes, labels=False)

    feature_names = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    return df


def generate_regression_data(n_samples=1000, n_features=10, noise=0.1):
    np.random.seed(42)

    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] + 2 * X[:, 1] + 0.5 * X[:, 2] + np.random.normal(0, noise, n_samples)

    feature_names = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    return df


def generate_time_series_data(n_samples=1000, n_features=5):
    np.random.seed(42)

    dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")

    trend = np.linspace(0, 10, n_samples)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)
    noise = np.random.normal(0, 1, n_samples)

    y = trend + seasonal + noise

    X = np.random.randn(n_samples, n_features)
    feature_names = [f"feature_{i+1}" for i in range(n_features)]

    df = pd.DataFrame(X, columns=feature_names)
    df["date"] = dates
    df["target"] = y

    return df


def generate_categorical_data(n_samples=1000):
    np.random.seed(42)

    categories = ["A", "B", "C", "D"]
    regions = ["North", "South", "East", "West"]

    df = pd.DataFrame(
        {
            "category": np.random.choice(categories, n_samples),
            "region": np.random.choice(regions, n_samples),
            "feature_1": np.random.normal(0, 1, n_samples),
            "feature_2": np.random.normal(0, 1, n_samples),
            "feature_3": np.random.normal(0, 1, n_samples),
        }
    )

    df["target"] = (
        (df["category"] == "A").astype(int) * 2
        + (df["region"] == "North").astype(int) * 1.5
        + df["feature_1"] * 0.5
        + df["feature_2"] * 0.3
        + np.random.normal(0, 0.1, n_samples)
    )

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate sample datasets for AUREON")
    parser.add_argument("--output-dir", default="sample_data", help="Output directory")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--features", type=int, default=10, help="Number of features")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("Generating sample datasets...")

    classification_data = generate_classification_data(args.samples, args.features)
    classification_data.to_csv(output_dir / "classification_data.csv", index=False)
    print(f"Generated classification data: {output_dir / 'classification_data.csv'}")

    regression_data = generate_regression_data(args.samples, args.features)
    regression_data.to_csv(output_dir / "regression_data.csv", index=False)
    print(f"Generated regression data: {output_dir / 'regression_data.csv'}")

    time_series_data = generate_time_series_data(args.samples, args.features)
    time_series_data.to_csv(output_dir / "time_series_data.csv", index=False)
    print(f"Generated time series data: {output_dir / 'time_series_data.csv'}")

    categorical_data = generate_categorical_data(args.samples)
    categorical_data.to_csv(output_dir / "categorical_data.csv", index=False)
    print(f"Generated categorical data: {output_dir / 'categorical_data.csv'}")

    print("\nSample datasets generated successfully!")
    print(f"Output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
