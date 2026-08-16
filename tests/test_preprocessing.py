import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    validate_dataframe,
    get_feature_types,
    build_preprocessor,
    prepare_features_target,
)


def sample_dataframe():
    """Create a small sample dataset for preprocessing tests."""
    return pd.DataFrame(
        {
            "Age": [25, 35, np.nan, 45],
            "MonthlyIncome": [3000, np.nan, 5000, 6000],
            "JobRole": ["Sales", "Manager", None, "Sales"],
            "MaritalStatus": ["Single", "Married", "Single", None],
            "Attrition": ["No", "Yes", "No", "Yes"],
        }
    )


def test_validate_dataframe_accepts_valid_dataframe():
    """Valid DataFrames should pass validation."""
    df = sample_dataframe()

    assert validate_dataframe(df) is True


def test_validate_dataframe_rejects_non_dataframe():
    """Non-DataFrame input should raise TypeError."""
    with pytest.raises(TypeError):
        validate_dataframe([1, 2, 3])


def test_validate_dataframe_rejects_empty_dataframe():
    """Empty DataFrames should raise ValueError."""
    df = pd.DataFrame()

    with pytest.raises(ValueError):
        validate_dataframe(df)


def test_get_feature_types():
    """Numeric and categorical columns should be identified correctly."""
    df = sample_dataframe()

    numeric, categorical = get_feature_types(df)

    assert "Age" in numeric
    assert "MonthlyIncome" in numeric
    assert "JobRole" in categorical
    assert "MaritalStatus" in categorical


def test_preprocessor_handles_missing_values():
    """Preprocessing should remove all missing feature values."""
    df = sample_dataframe()

    X, _ = prepare_features_target(df)
    preprocessor = build_preprocessor(df)

    transformed = preprocessor.fit_transform(X)

    assert not np.isnan(transformed).any()


def test_preprocessor_encodes_categorical_variables():
    """Categorical variables should be converted to numeric features."""
    df = sample_dataframe()

    X, _ = prepare_features_target(df)
    preprocessor = build_preprocessor(df)

    transformed = preprocessor.fit_transform(X)

    assert np.issubdtype(transformed.dtype, np.number)
    assert transformed.shape[1] > X.select_dtypes(include=["number"]).shape[1]


def test_prepare_features_does_not_modify_original_dataframe():
    """Preparing features and target must not modify the original data."""
    df = sample_dataframe()
    original = df.copy(deep=True)

    prepare_features_target(df)

    pd.testing.assert_frame_equal(df, original)


def test_prepare_features_target_encodes_target():
    """Attrition target should be converted from Yes/No to 1/0."""
    df = sample_dataframe()

    _, y = prepare_features_target(df)

    assert set(y.unique()) == {0, 1}


def test_prepare_features_target_rejects_invalid_target():
    """Unexpected target values should raise ValueError."""
    df = sample_dataframe()
    df.loc[0, "Attrition"] = "Maybe"

    with pytest.raises(ValueError):
        prepare_features_target(df)


def test_missing_target_column_raises_error():
    """Missing target column should raise ValueError."""
    df = sample_dataframe().drop(columns=["Attrition"])

    with pytest.raises(ValueError):
        prepare_features_target(df)