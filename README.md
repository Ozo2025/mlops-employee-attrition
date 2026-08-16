# Employee Attrition MLOps Pipeline

## Project Overview

This project implements a complete end-to-end MLOps pipeline for an employee attrition classification model.

The goal of the project is not only to train a machine learning model, but to demonstrate the infrastructure and engineering practices required to manage a model throughout its lifecycle.

The pipeline includes:

- Git version control
- DVC data versioning
- YAML-based configuration
- Data preprocessing
- Model training and evaluation
- MLflow experiment tracking
- Automated testing with pytest
- CI/CD with GitHub Actions
- Data drift monitoring with Evidently

The model predicts whether an employee is likely to leave the company based on employee demographic, job, compensation, and workplace-related features.

---

## Dataset

This project uses the IBM HR Analytics Employee Attrition dataset.

The dataset contains:

- 1,470 employee records
- 35 columns
- Numeric and categorical features
- Binary target variable: `Attrition`
- Target values: `Yes` and `No`

Because the original dataset contains no missing values, missing values were intentionally introduced into the following features to satisfy the project requirements and test the preprocessing pipeline:

- `Age`
- `MonthlyIncome`
- `JobRole`
- `MaritalStatus`

The dataset is tracked using DVC rather than being committed directly to Git.

---

## Repository Structure

```text
mlops-employee-attrition/
│
├── .github/
│   └── workflows/
│       └── ml_pipeline.yml
│
├── configs/
│   └── config.yaml
│
├── data/
│   └── employee_attrition.csv.dvc
│
├── reports/
│   └── drift_report.html
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── monitor_drift.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_data_validation.py
│   └── test_model.py
│
├── compare_experiments.py
├── pytest.ini
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

Clone the repository:

```bash
git clone https://github.com/Ozo2025/mlops-employee-attrition.git
cd mlops-employee-attrition
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data Versioning with DVC

DVC is used to track the training dataset separately from Git.

Initialize DVC if necessary:

```bash
dvc init
```

The dataset is represented in Git by:

```text
data/employee_attrition.csv.dvc
```

The project uses a configured DVC remote for dataset storage.

To retrieve DVC-tracked data when the configured remote is accessible:

```bash
dvc pull
```

This keeps large/raw data artifacts outside the Git repository while allowing dataset versions to remain associated with source-code versions.

---

## Configuration

Training and monitoring settings are stored in:

```text
configs/config.yaml
```

The configuration controls settings including:

- Dataset path
- Target column
- Train/test split
- Random seed
- Model type
- Number of estimators
- Maximum tree depth
- Minimum F1 threshold
- MLflow experiment name
- Data version
- Drift threshold
- Drift report location

The training pipeline reads these values from the YAML file rather than hardcoding them in the training script.

---

## Model Training

The project uses a Random Forest classifier with preprocessing and model training combined in a scikit-learn pipeline.

Run training with:

```bash
python -m src.train
```

The training process:

1. Loads configuration values.
2. Loads the dataset.
3. Separates features and target.
4. Splits the data into training and testing sets.
5. Imputes missing values.
6. Encodes categorical features.
7. Scales numeric features.
8. Trains the Random Forest classifier.
9. Evaluates model performance.
10. Logs the experiment to MLflow.
11. Saves the trained model.
12. Enforces a minimum F1 quality threshold.

The primary model metric is F1 score because employee attrition is an imbalanced classification problem.

---

## Model Evaluation

Run evaluation with:

```bash
python -m src.evaluate
```

The evaluation script reports:

- Accuracy
- F1 score
- ROC-AUC

These metrics provide multiple views of model performance rather than relying only on accuracy.

---

## MLflow Experiment Tracking

MLflow is used to track model experiments.

Each training run records:

- Model type
- Number of estimators
- Maximum depth
- Random state
- Data version
- Accuracy
- F1 score
- ROC-AUC
- Trained model artifact

Multiple Random Forest configurations were evaluated.

Example configurations included:

| n_estimators | max_depth |
|---:|---:|
| 100 | 5 |
| 200 | 10 |
| 300 | 10 |
| 300 | 15 |
| 500 | 15 |

The best experiment based on F1 score can be identified programmatically with:

```bash
python compare_experiments.py
```

The experiment comparison uses `mlflow.search_runs()` and sorts the experiment history by the primary F1 metric.

During experimentation, the strongest observed run used 100 estimators and a maximum depth of 5.

---

## Automated Testing

The project contains tests at three levels.

### Preprocessing Unit Tests

The preprocessing suite verifies:

- Valid DataFrame handling
- Invalid input handling
- Empty DataFrame handling
- Numeric and categorical feature identification
- Missing-value imputation
- Categorical encoding
- Preservation of the original DataFrame
- Target encoding
- Invalid target detection
- Missing target detection

### Data Validation Tests

The data validation suite verifies:

- Required columns exist
- Target values are valid
- Employee ages are within expected ranges
- Monthly income is non-negative
- Dataset contains at least 1,000 rows
- Dataset contains at least eight features
- Missing values exist as required by the project

### Model Validation Tests

The model tests verify:

- Prediction shape and values are valid
- The model exceeds a minimum F1 performance threshold

Run the complete test suite with:

```bash
pytest tests/ -v
```

The complete suite currently contains 19 tests.

---

## CI/CD with GitHub Actions

The repository includes an automated GitHub Actions workflow located at:

```text
.github/workflows/ml_pipeline.yml
```

The workflow triggers on:

- Pushes to `main`
- Pull requests targeting `main`

The workflow contains two jobs:

### Test Job

The test job:

1. Checks out the repository.
2. Configures Python.
3. Installs project dependencies.
4. Prepares the dataset for the CI environment.
5. Runs the complete pytest suite.

### Training Job

The training job depends on the test job passing.

It:

1. Creates a fresh training environment.
2. Installs dependencies.
3. Prepares the training dataset.
4. Runs the training pipeline.
5. Applies the configured F1 performance quality gate.

If model performance falls below the configured threshold, training exits with a non-zero status and the CI pipeline fails.

A successful GitHub Actions pipeline has been completed with both the testing and training jobs passing.

---

## Drift Monitoring

Evidently is used to compare the reference training distribution against simulated production data.

Run monitoring with:

```bash
python -m src.monitor_drift
```

The monitoring script:

1. Loads the reference training data.
2. Generates simulated production data.
3. Introduces controlled distribution changes.
4. Runs drift detection across model features.
5. Identifies drifted features.
6. Calculates the overall drift share.
7. Saves an HTML report.
8. Compares the drift share with a configurable threshold.
9. Exits with code 1 if the configured drift threshold is exceeded.

The generated report is saved to:

```text
reports/drift_report.html
```

---

## Drift Analysis

The simulated production dataset intentionally changes several employee characteristics to demonstrate how the monitoring system responds to changing production data.

### Which features showed drift and why?

Four features showed detectable drift:

- `Age`
- `MonthlyIncome`
- `JobRole`
- `MaritalStatus`

The drift was expected because the production simulation intentionally changed the distributions of these variables. Age values were shifted, monthly income was increased, and the distributions of job role and marital status were altered.

The monitoring run detected 4 drifted features out of 34 total features, producing an overall drift share of approximately 11.76%.

### Would this drift likely affect model performance?

Potentially.

Age, income, job role, and marital status may contain useful predictive information for employee attrition. Significant changes in these distributions could cause production data to differ from the data on which the model was trained.

However, the overall drift share remained below the configured project threshold of 30%.

### Recommended Action

Continue monitoring.

Because approximately 11.76% of features drifted, the overall drift level remains below the 30% alert threshold. Immediate retraining is therefore not required.

If drift continues to increase, additional investigation should determine whether model performance is degrading. Retraining should be considered if drift exceeds the configured threshold or production performance metrics deteriorate.

---

## MLOps Workflow

The overall project workflow is:

```text
Raw Data
   ↓
DVC Versioning
   ↓
Preprocessing
   ↓
Model Training
   ↓
MLflow Experiment Tracking
   ↓
Model Evaluation
   ↓
pytest Validation
   ↓
GitHub Actions CI/CD
   ↓
Production Monitoring
   ↓
Evidently Drift Detection
```

This structure demonstrates how model development, testing, reproducibility, automation, and monitoring can be integrated into a complete MLOps workflow.

---

## Technologies

- Python 3.12
- pandas
- scikit-learn
- MLflow
- DVC
- pytest
- GitHub Actions
- Evidently
- PyYAML
- joblib

---

## Project Status

The project includes a complete MLOps workflow with version-controlled source code, DVC-tracked data, configurable model training, MLflow experiment tracking, automated tests, a successful CI/CD pipeline, model quality gates, and production data drift monitoring.