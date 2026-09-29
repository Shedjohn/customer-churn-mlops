from pathlib import Path

import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# PATHS
# ============================================================

PROCESSED_DATA_PATH = Path("data/processed")

MODEL_PATH = Path("models/random_forest.pkl")

REPORTS_PATH = Path("reports")

METRICS_PATH = REPORTS_PATH / "metrics.json"


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000"
)

EXPERIMENT_NAME = "Customer Churn Prediction"


# ============================================================
# MODEL PARAMETERS
# ============================================================

RANDOM_STATE = 42

N_ESTIMATORS = 100


# ============================================================
# LOAD PROCESSED DATA
# ============================================================

def load_processed_data():
    """Load training and testing data."""

    X_train_path = (
        PROCESSED_DATA_PATH / "X_train.csv"
    )

    X_test_path = (
        PROCESSED_DATA_PATH / "X_test.csv"
    )

    y_train_path = (
        PROCESSED_DATA_PATH / "y_train.csv"
    )

    y_test_path = (
        PROCESSED_DATA_PATH / "y_test.csv"
    )

    required_files = [
        X_train_path,
        X_test_path,
        y_train_path,
        y_test_path
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required processed file not found: "
                f"{file_path}"
            )

    X_train = pd.read_csv(
        X_train_path
    )

    X_test = pd.read_csv(
        X_test_path
    )

    y_train = pd.read_csv(
        y_train_path
    ).squeeze()

    y_test = pd.read_csv(
        y_test_path
    ).squeeze()

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# CREATE RANDOM FOREST
# ============================================================

def create_model():

    """
    Create the Random Forest model.
    """

    model = RandomForestClassifier(

        n_estimators=N_ESTIMATORS,

        random_state=RANDOM_STATE
    )

    return model


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    """
    Generate predictions and calculate
    evaluation metrics.
    """

    y_pred = model.predict(
        X_test
    )

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    metrics = {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1_score": f1,

        "roc_auc": roc_auc
    }

    return metrics


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model):

    """Save trained Random Forest."""

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"\nModel saved to: {MODEL_PATH}"
    )


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(metrics):

    """Save metrics to JSON."""

    import json

    REPORTS_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        METRICS_PATH,
        "w"
    ) as file:

        json.dump(
            {
                key: round(
                    float(value),
                    4
                )
                for key, value
                in metrics.items()
            },
            file,
            indent=4
        )

    print(
        f"Metrics saved to: {METRICS_PATH}"
    )


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():

    print("=" * 60)

    print(
        "RANDOM FOREST TRAINING WITH MLFLOW"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # CONNECT TO MLFLOW
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    print(
        f"\nMLflow tracking URI: "
        f"{mlflow.get_tracking_uri()}"
    )

    print(
        f"MLflow experiment: "
        f"{EXPERIMENT_NAME}"
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = load_processed_data()

    print(
        "\nProcessed data loaded."
    )

    print(
        f"X_train shape: {X_train.shape}"
    )

    print(
        f"X_test shape:  {X_test.shape}"
    )

    # --------------------------------------------------------
    # START MLFLOW RUN
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="Random Forest Baseline"
    ) as run:

        print(
            f"\nMLflow Run ID: "
            f"{run.info.run_id}"
        )

        # ----------------------------------------------------
        # CREATE MODEL
        # ----------------------------------------------------

        model = create_model()

        # ----------------------------------------------------
        # LOG PARAMETERS
        # ----------------------------------------------------

        mlflow.log_params({

            "model": "RandomForestClassifier",

            "n_estimators": N_ESTIMATORS,

            "random_state": RANDOM_STATE
        })

        # ----------------------------------------------------
        # TRAIN MODEL
        # ----------------------------------------------------

        print(
            "\nTraining Random Forest..."
        )

        model.fit(
            X_train,
            y_train
        )

        print(
            "Training completed."
        )

        # ----------------------------------------------------
        # EVALUATE
        # ----------------------------------------------------

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        # ----------------------------------------------------
        # LOG METRICS
        # ----------------------------------------------------

        mlflow.log_metrics(
            metrics
        )

        # ----------------------------------------------------
        # DISPLAY METRICS
        # ----------------------------------------------------

        print(
            "\nEvaluation Metrics:"
        )

        print("-" * 30)

        for name, value in metrics.items():

            print(
                f"{name}: {value:.4f}"
            )

        # ----------------------------------------------------
        # SAVE LOCAL MODEL
        # ----------------------------------------------------

        save_model(
            model
        )

        # ----------------------------------------------------
        # SAVE LOCAL METRICS
        # ----------------------------------------------------

        save_metrics(
            metrics
        )

        # ----------------------------------------------------
        # LOG MODEL TO MLFLOW
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            name="random_forest_model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        # ----------------------------------------------------
        # LOG METRICS FILE
        # ----------------------------------------------------

        mlflow.log_artifact(
            str(METRICS_PATH)
        )

        # ----------------------------------------------------
        # ADD TAGS
        # ----------------------------------------------------

        mlflow.set_tags({

            "project": "Customer Churn Prediction",

            "model_type": "Random Forest",

            "dataset": "Customer Churn Dataset",

            "stage": "baseline"
        })

        print(
            "\nMLflow logging completed."
        )

    print(
        "\n" + "=" * 60
    )

    print(
        "TRAINING AND MLFLOW RUN COMPLETED"
    )

    print(
        "=" * 60
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()