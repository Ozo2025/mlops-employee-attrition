import mlflow


EXPERIMENT_NAME = "employee_attrition"
PRIMARY_METRIC = "metrics.f1"


def compare_experiments():
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise ValueError(
            f"Experiment '{EXPERIMENT_NAME}' was not found."
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"{PRIMARY_METRIC} DESC"],
    )

    if runs.empty:
        raise ValueError("No MLflow runs were found.")

    columns = [
        "run_id",
        "params.n_estimators",
        "params.max_depth",
        "metrics.accuracy",
        "metrics.f1",
        "metrics.roc_auc",
        "status",
    ]

    available_columns = [
        column for column in columns
        if column in runs.columns
    ]

    print("\nExperiment Comparison")
    print("---------------------")
    print(runs[available_columns].to_string(index=False))

    best_run = runs.iloc[0]

    print("\nBest Run")
    print("--------")
    print(f"Run ID: {best_run['run_id']}")
    print(
        f"n_estimators: "
        f"{best_run['params.n_estimators']}"
    )
    print(
        f"max_depth: "
        f"{best_run['params.max_depth']}"
    )
    print(
        f"Accuracy: "
        f"{best_run['metrics.accuracy']:.4f}"
    )
    print(
        f"F1 Score: "
        f"{best_run['metrics.f1']:.4f}"
    )
    print(
        f"ROC-AUC: "
        f"{best_run['metrics.roc_auc']:.4f}"
    )


if __name__ == "__main__":
    compare_experiments()