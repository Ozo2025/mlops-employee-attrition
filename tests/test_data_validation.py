import pandas as pd


DATA_PATH = "data/employee_attrition.csv"


def load_data():
    """Load the employee attrition dataset."""
    return pd.read_csv(DATA_PATH)


def test_expected_columns_present():
    """Dataset should contain required columns."""
    df = load_data()

    expected_columns = [
        "Age",
        "MonthlyIncome",
        "JobRole",
        "MaritalStatus",
        "Attrition",
    ]

    for column in expected_columns:
        assert column in df.columns


def test_target_contains_expected_values():
    """Attrition should contain only Yes and No values."""
    df = load_data()

    expected_values = {"Yes", "No"}
    actual_values = set(df["Attrition"].dropna().unique())

    assert actual_values.issubset(expected_values)


def test_age_within_expected_range():
    """Employee ages should fall within a reasonable range."""
    df = load_data()

    ages = df["Age"].dropna()

    assert ages.min() >= 18
    assert ages.max() <= 70


def test_monthly_income_non_negative():
    """Monthly income should never be negative."""
    df = load_data()

    incomes = df["MonthlyIncome"].dropna()

    assert (incomes >= 0).all()


def test_dataset_has_minimum_rows():
    """Dataset should satisfy the project requirement of 1,000+ rows."""
    df = load_data()

    assert len(df) >= 1000


def test_dataset_has_minimum_features():
    """Dataset should satisfy the project requirement of at least 8 features."""
    df = load_data()

    feature_count = len(df.columns) - 1

    assert feature_count >= 8


def test_dataset_contains_missing_values():
    """Dataset should contain missing values as required by the project."""
    df = load_data()

    assert df.isna().sum().sum() > 0