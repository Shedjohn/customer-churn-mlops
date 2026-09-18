from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# PATHS
# ============================================================

PROCESSED_DATA_PATH = Path("data/processed")

PREPROCESSOR_PATH = Path("models/preprocessor.pkl")

MODEL_PATH = Path("models/random_forest.pkl")


# ============================================================
# MODEL PARAMETERS
# ============================================================

RANDOM_STATE = 42

N_ESTIMATORS = 100


# ============================================================
# LOAD PROCESSED DATA
# ============================================================

def load_processed_data():
    """
    Load the training and testing datasets created
    during preprocessing.
    """

    X_train_path = PROCESSED_DATA_PATH / "X_train.csv"
    X_test_path = PROCESSED_DATA_PATH / "X_test.csv"
    y_train_path = PROCESSED_DATA_PATH / "y_train.csv"
    y_test_path = PROCESSED_DATA_PATH / "y_test.csv"

    # Check that all required files exist
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

    # Load datasets
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)

    y_train = pd.read_csv(y_train_path).squeeze()
    y_test = pd.read_csv(y_test_path).squeeze()

    return X_train, X_test, y_train, y_test


# ============================================================
# CREATE RANDOM FOREST
# ============================================================

def create_model():
    """
    Create the Random Forest classifier.

    Random Forest was selected as our final model
    from the models evaluated during the Colab stage.
    """

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE
    )

    return model


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(model, X_train, y_train):
    """
    Train the Random Forest model using the training data.
    """

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model):
    """
    Save the trained Random Forest model to disk.
    """

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
# MAIN TRAINING PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("RANDOM FOREST MODEL TRAINING")
    print("=" * 60)


    # --------------------------------------------------------
    # 1. Load processed data
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
        f"X_test shape:  {X_test.shape}"
    )

    print(
        f"y_train shape: {y_train.shape}"
    )

    print(
        f"y_test shape:  {y_test.shape}"
    )


    # --------------------------------------------------------
    # 2. Create Random Forest
    # --------------------------------------------------------

    model = create_model()

    print("\nRandom Forest configuration:")

    print(
        f"Number of trees: {N_ESTIMATORS}"
    )

    print(
        f"Random state: {RANDOM_STATE}"
    )


    # --------------------------------------------------------
    # 3. Train model
    # --------------------------------------------------------

    print("\nTraining Random Forest...")

    model = train_model(
        model,
        X_train,
        y_train
    )

    print("Training completed.")


    # --------------------------------------------------------
    # 4. Save model
    # --------------------------------------------------------

    save_model(model)


    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()