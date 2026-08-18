import copy

import yaml

from src.train import train_model


CONFIG_PATH = "configs/config.yaml"

EXPERIMENTS = [
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 10},
    {"n_estimators": 300, "max_depth": 10},
    {"n_estimators": 300, "max_depth": 15},
    {"n_estimators": 500, "max_depth": 15},
]


def run_experiments():
    """Run and log at least five model configurations with MLflow."""

    with open(CONFIG_PATH, "r") as file:
        original_config = yaml.safe_load(file)

    print("Running MLflow Experiments")
    print("=" * 50)

    for index, experiment in enumerate(EXPERIMENTS, start=1):
        config = copy.deepcopy(original_config)

        config["model"]["n_estimators"] = experiment["n_estimators"]
        config["model"]["max_depth"] = experiment["max_depth"]

        # Experiment comparison should log every configuration even when
        # its F1 score is below the production quality threshold.
        config["training"]["minimum_f1"] = 0.0

        with open(CONFIG_PATH, "w") as file:
            yaml.safe_dump(config, file, sort_keys=False)

        print(
            f"\nExperiment {index}/5: "
            f"n_estimators={experiment['n_estimators']}, "
            f"max_depth={experiment['max_depth']}"
        )

        train_model()

    # Restore the original project configuration after all experiments.
    with open(CONFIG_PATH, "w") as file:
        yaml.safe_dump(original_config, file, sort_keys=False)

    print("\n" + "=" * 50)
    print("Completed 5 MLflow experiment runs.")
    print("Run compare_experiments.py to compare results.")


if __name__ == "__main__":
    run_experiments()