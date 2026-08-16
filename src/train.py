import os
import yaml
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import build_preprocessor, prepare_features_target


CONFIG_PATH = "configs/config.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def train_model():
    config = load_config()

    data_path = config["data"]["raw_path"]
    target_column = config["data"]["target"]
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]

    df = pd.read_csv(data_path)

    X, y = prepare_features_target(
        df,
        target_column=target_column
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor(
        df,
        target_column=target_column
    )

    model = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"],
        random_state=config["model"]["random_state"],
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    mlflow.set_experiment(
        config["mlflow"]["experiment_name"]
    )

    with mlflow.start_run():

        mlflow.log_param(
            "model_type",
            config["model"]["type"]
        )

        mlflow.log_param(
            "n_estimators",
            config["model"]["n_estimators"]
        )

        mlflow.log_param(
            "max_depth",
            config["model"]["max_depth"]
        )

        mlflow.log_param(
            "random_state",
            config["model"]["random_state"]
        )

        mlflow.log_param(
            "data_version",
            config["mlflow"]["data_version"]
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        f1 = f1_score(
            y_test,
            predictions
        )

        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "f1",
            f1
        )

        mlflow.log_metric(
            "roc_auc",
            roc_auc
        )

        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            skops_trusted_types=["numpy.dtype"],
        )

        os.makedirs(
            "models",
            exist_ok=True
        )

        joblib.dump(
            pipeline,
            "models/employee_attrition_model.joblib"
        )

        print("\nTraining Complete")
        print("-----------------")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC-AUC:  {roc_auc:.4f}")

        minimum_f1 = config["training"]["minimum_f1"]

        if f1 < minimum_f1:
            raise ValueError(
                f"Model F1 score {f1:.4f} "
                f"is below required threshold "
                f"{minimum_f1:.4f}"
            )

    return pipeline


if __name__ == "__main__":
    train_model()