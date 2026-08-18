import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import build_preprocessor, prepare_features_target


DATA_PATH = "data/employee_attrition.csv"


def load_sample_data():
    """Load a small reproducible sample for model validation tests."""
    df = pd.read_csv(DATA_PATH)

    return df.sample(
        n=800,
        random_state=42,
    )


def build_test_model(df):
    """Build a lightweight model pipeline for validation testing."""
    X, y = prepare_features_target(df)

    preprocessor = build_preprocessor(df)

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline, X, y


def test_model_prediction_shape():
    """Verify model predictions have the expected number of rows."""
    df = load_sample_data()

    pipeline, X, y = build_test_model(df)

    X_train, X_test, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    assert predictions.shape == (len(X_test),)


def test_model_prediction_type():
    """Verify predictions are a NumPy array containing binary classes."""
    df = load_sample_data()

    pipeline, X, y = build_test_model(df)

    X_train, X_test, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    assert isinstance(predictions, np.ndarray)
    assert set(predictions).issubset({0, 1})


def test_model_meets_minimum_performance_threshold():
    """Verify the model achieves the required minimum F1 performance."""
    df = load_sample_data()

    pipeline, X, y = build_test_model(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    f1 = f1_score(
        y_test,
        predictions,
    )

    assert f1 >= 0.30