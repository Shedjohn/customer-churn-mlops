from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PATHS
# ============================================================

PROCESSED_DATA_PATH = Path("data/processed")

MODEL_PATH = Path("models/random_forest.pkl")

REPORTS_PATH = Path("reports")

METRICS_PATH = REPORTS_PATH / "metrics.json"


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():
    """
    Load the testing features and target.

    The test data was created during preprocessing
    and has not been used to train the model.
    """

    X_test_path = PROCESSED_DATA_PATH / "X_test.csv"
    y_test_path = PROCESSED_DATA_PATH / "y_test.csv"

    # Check that files exist
    if not X_test_path.exists():
        raise FileNotFoundError(
            f"Test features not found: {X_test_path}"
        )

    if not y_test_path.exists():
        raise FileNotFoundError(
            f"Test target not found: {y_test_path}"
        )

    # Load test data
    X_test = pd.read_csv(X_test_path)

    y_test = pd.read_csv(
        y_test_path
    ).squeeze()

    return X_test, y_test


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model(path: Path):
    """
    Load the trained Random Forest model.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}"
        )

    model = joblib.load(path)

    return model


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(model, X_test):
    """
    Generate class predictions and churn probabilities.
    """

    # Predicted class
    y_pred = model.predict(X_test)

    # Probability of class 1 (churn)
    y_probability = model.predict_proba(X_test)[:, 1]

    return y_pred, y_probability


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(y_test, y_pred, y_probability):
    """
    Calculate classification evaluation metrics.
    """

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
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4)
    }

    return metrics


# ============================================================
# CONFUSION MATRIX
# ============================================================

def display_confusion_matrix(y_test, y_pred):
    """
    Display the confusion matrix.
    """

    matrix = confusion_matrix(
        y_test,
        y_pred
    )

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nConfusion Matrix Breakdown:")

    tn, fp, fn, tp = matrix.ravel()

    print(f"True Negatives : {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives : {tp}")

    return matrix


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

def display_classification_report(y_test, y_pred):
    """
    Display the detailed classification report.
    """

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    return report


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(metrics):
    """
    Save evaluation metrics as a JSON file.
    """

    REPORTS_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        METRICS_PATH,
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print(
        f"\nMetrics saved to: {METRICS_PATH}"
    )


# ============================================================
# MAIN EVALUATION PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("RANDOM FOREST MODEL EVALUATION")
    print("=" * 60)


    # --------------------------------------------------------
    # 1. Load test data
    # --------------------------------------------------------

    X_test, y_test = load_test_data()

    print("\nTest data loaded.")

    print(
        f"X_test shape: {X_test.shape}"
    )

    print(
        f"y_test shape: {y_test.shape}"
    )


    # --------------------------------------------------------
    # 2. Load trained model
    # --------------------------------------------------------

    model = load_model(
        MODEL_PATH
    )

    print(
        "\nRandom Forest model loaded successfully."
    )


    # --------------------------------------------------------
    # 3. Generate predictions
    # --------------------------------------------------------

    y_pred, y_probability = generate_predictions(
        model,
        X_test
    )

    print(
        "\nPredictions generated."
    )


    # --------------------------------------------------------
    # 4. Calculate metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_test,
        y_pred,
        y_probability
    )


    # --------------------------------------------------------
    # 5. Display metrics
    # --------------------------------------------------------

    print("\nEvaluation Metrics:")
    print("-" * 30)

    print(
        f"Accuracy : {metrics['accuracy']}"
    )

    print(
        f"Precision: {metrics['precision']}"
    )

    print(
        f"Recall   : {metrics['recall']}"
    )

    print(
        f"F1-score : {metrics['f1_score']}"
    )

    print(
        f"ROC-AUC  : {metrics['roc_auc']}"
    )


    # --------------------------------------------------------
    # 6. Confusion matrix
    # --------------------------------------------------------

    display_confusion_matrix(
        y_test,
        y_pred
    )


    # --------------------------------------------------------
    # 7. Classification report
    # --------------------------------------------------------

    display_classification_report(
        y_test,
        y_pred
    )


    # --------------------------------------------------------
    # 8. Save metrics
    # --------------------------------------------------------

    save_metrics(
        metrics
    )


    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()