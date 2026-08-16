import yaml
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.preprocess import prepare_features_target


CONFIG_PATH = "configs/config.yaml"
MODEL_PATH = "models/employee_attrition_model.joblib"


def load_config(path=CONFIG_PATH):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def evaluate_model():
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

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

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

    results = {
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": roc_auc,
    }

    print("\nModel Evaluation")
    print("----------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC:  {roc_auc:.4f}")

    return results


if __name__ == "__main__":
    evaluate_model()