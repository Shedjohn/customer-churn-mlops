from pathlib import Path

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


# ============================================================
# MLflow SETTINGS
# ============================================================

MLFLOW_EXPERIMENT_NAME = "Customer Churn Prediction"


# ============================================================
# MODEL PARAMETERS
# ============================================================

RANDOM_STATE = 42
N_ESTIMATORS = 100


# ============================================================
# LOAD PROCESSED DATA
# ============================================================

def load_processed_data():
    """Load training and testing datasets."""

    X_train_path = PROCESSED_DATA_PATH / "X_train.csv"
    X_test_path = PROCESSED_DATA_PATH / "X_test.csv"
    y_train_path = PROCESSED_DATA_PATH / "y_train.csv"
    y_test_path = PROCESSED_DATA_PATH / "y_test.csv"

    required_files = [
        X_train_path,
        X_test_path,
        y_train_path,
        y_test_path
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required processed file not found: {file_path}"
            )

    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)

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
    """Create the Random Forest classifier."""

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE
    )

    return model


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(model, X_train, y_train):
    """Train Random Forest."""

    model.fit(
        X_train,
        y_train
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

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            y_pred
        ),

        "precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "f1_score": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_test,
            y_probability
        )
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
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("RANDOM FOREST TRAINING WITH MLFLOW")
    print("=" * 60)

    # --------------------------------------------------------
    # Configure MLflow experiment
    # --------------------------------------------------------

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT_NAME
    )

    # --------------------------------------------------------
    # Load processed data
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = load_processed_data()

    print("\nProcessed data loaded.")

    print(
        f"X_train shape: {X_train.shape}"
    )

    print(
        f"X_test shape: {X_test.shape}"
    )

    # --------------------------------------------------------
    # Start MLflow run
    # --------------------------------------------------------

    with mlflow.start_run():

        # ----------------------------------------------------
        # Create model
        # ----------------------------------------------------

        model = create_model()

        # ----------------------------------------------------
        # Log model parameters
        # ----------------------------------------------------

        mlflow.log_param(
            "model_type",
            "RandomForestClassifier"
        )

        mlflow.log_param(
            "n_estimators",
            N_ESTIMATORS
        )

        mlflow.log_param(
            "random_state",
            RANDOM_STATE
        )

        # ----------------------------------------------------
        # Train model
        # ----------------------------------------------------

        print("\nTraining Random Forest...")

        model = train_model(
            model,
            X_train,
            y_train
        )

        print("Training completed.")

        # ----------------------------------------------------
        # Evaluate model
        # ----------------------------------------------------

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        # ----------------------------------------------------
        # Log metrics to MLflow
        # ----------------------------------------------------

        mlflow.log_metrics(
            metrics
        )

        # ----------------------------------------------------
        # Print metrics
        # ----------------------------------------------------

        print("\nEvaluation Metrics:")
        print("-" * 30)

        print(
            f"Accuracy : {metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: {metrics['precision']:.4f}"
        )

        print(
            f"Recall   : {metrics['recall']:.4f}"
        )

        print(
            f"F1-score : {metrics['f1_score']:.4f}"
        )

        print(
            f"ROC-AUC  : {metrics['roc_auc']:.4f}"
        )

        # ----------------------------------------------------
        # Save model locally
        # ----------------------------------------------------

        save_model(model)

        # ----------------------------------------------------
        # Log model to MLflow
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            sk_model=model,
            name="random_forest_model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        # ----------------------------------------------------
        # Display run information
        # ----------------------------------------------------

        run_id = mlflow.active_run().info.run_id

        print("\nMLflow run completed.")

        print(
            f"Run ID: {run_id}"
        )

    print("\n" + "=" * 60)
    print("TRAINING + MLFLOW COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()

