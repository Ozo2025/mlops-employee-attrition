import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def validate_dataframe(df):
    """Validate that the input is a non-empty pandas DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("DataFrame cannot be empty.")

    return True


def get_feature_types(df, target_column="Attrition"):
    """Return lists of numeric and categorical feature columns."""
    validate_dataframe(df)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    features = df.drop(columns=[target_column])

    numeric_features = features.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = features.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    return numeric_features, categorical_features


def build_preprocessor(df, target_column="Attrition"):
    """
    Build a preprocessing pipeline for numeric and categorical features.

    Numeric:
        - Fill missing values with the median
        - Standardize values

    Categorical:
        - Fill missing values with the most frequent value
        - One-hot encode categories
    """
    numeric_features, categorical_features = get_feature_types(
        df, target_column
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


def prepare_features_target(df, target_column="Attrition"):
    """
    Separate features and target without modifying the original DataFrame.
    """
    validate_dataframe(df)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    data = df.copy()

    X = data.drop(columns=[target_column])
    y = data[target_column].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Target contains unexpected values.")

    return X, y