import os
import sys
import yaml
import numpy as np
import pandas as pd

from evidently import Report
from evidently.metrics import DriftedColumnsCount, ValueDrift
from evidently.presets import DataDriftPreset


CONFIG_PATH = "configs/config.yaml"
COLUMN_DRIFT_THRESHOLD = 0.10


def load_config(path=CONFIG_PATH):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def create_production_data(reference_df):
    """
    Create simulated production data with intentional drift.
    """
    production_df = reference_df.copy()

    rng = np.random.default_rng(42)

    production_df["Age"] = (
        production_df["Age"].fillna(
            production_df["Age"].median()
        )
        + rng.normal(10, 2, size=len(production_df))
    )

    production_df["MonthlyIncome"] = (
        production_df["MonthlyIncome"].fillna(
            production_df["MonthlyIncome"].median()
        )
        * 1.50
    )

    production_df.loc[
        production_df.sample(
            frac=0.60,
            random_state=42
        ).index,
        "JobRole"
    ] = "Sales Executive"

    production_df.loc[
        production_df.sample(
            frac=0.60,
            random_state=24
        ).index,
        "MaritalStatus"
    ] = "Single"

    return production_df


def monitor_drift():
    config = load_config()

    data_path = config["data"]["raw_path"]
    target_column = config["data"]["target"]
    drift_threshold = config["monitoring"]["drift_threshold"]
    report_path = config["monitoring"]["report_path"]

    df = pd.read_csv(data_path)

    reference_df = df.drop(
        columns=[target_column]
    ).copy()

    production_df = create_production_data(
        reference_df
    )

    feature_columns = reference_df.columns.tolist()

    metrics = [
        DriftedColumnsCount(
            drift_share=drift_threshold
        )
    ]

    metrics.extend(
        [
            ValueDrift(column=column)
            for column in feature_columns
        ]
    )

    report = Report(metrics)

    result = report.run(
        current_data=production_df,
        reference_data=reference_df,
    )

    visual_report = Report(
        [
            DataDriftPreset(
                drift_share=drift_threshold
            )
        ]
    )

    visual_result = visual_report.run(
        current_data=production_df,
        reference_data=reference_df,
    )

    os.makedirs(
        os.path.dirname(report_path),
        exist_ok=True
    )

    visual_result.save_html(report_path)

    result_dict = result.dict()

    drift_count = 0
    drift_share = 0.0
    drifted_features = []

    for metric in result_dict.get("metrics", []):
        metric_name = metric.get("metric_name", "")
        value = metric.get("value")
        metric_config = metric.get("config", {})

        if metric_name.startswith("DriftedColumnsCount"):
            drift_count = int(value["count"])
            drift_share = float(value["share"])

        elif metric_name.startswith("ValueDrift"):
            column = metric_config.get("column")

            if (
                column is not None
                and float(value) > COLUMN_DRIFT_THRESHOLD
            ):
                drifted_features.append(column)

    print("\nDrift Monitoring Summary")
    print("------------------------")
    print(f"Total features: {len(feature_columns)}")
    print(f"Drifted features: {drift_count}")
    print(f"Overall drift share: {drift_share:.2%}")

    if drifted_features:
        print(
            "Features showing drift: "
            + ", ".join(drifted_features)
        )
    else:
        print("Features showing drift: None")

    print(
        f"Configured drift threshold: "
        f"{drift_threshold:.2%}"
    )

    print(
        f"HTML report saved to: "
        f"{report_path}"
    )

    if drift_share > drift_threshold:
        print(
            "\nALERT: Drift exceeds the configured threshold."
        )
        print(
            "Recommended action: investigate and consider retraining."
        )
        sys.exit(1)

    print(
        "\nDrift is below the configured threshold."
    )
    print(
        "Recommended action: continue monitoring."
    )


if __name__ == "__main__":
    monitor_drift()