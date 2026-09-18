from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

RAW_DATA_PATH = Path("data/raw/churn.csv")
PROCESSED_DATA_PATH = Path("data/processed")
PREPROCESSOR_PATH = Path("models/preprocessor.pkl")


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "Churn"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# LOAD DATA
# ============================================================

def load_data(path: Path) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    return pd.read_csv(path)


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    return df


# ============================================================
# VALIDATE DATA
# ============================================================

def validate_data(df: pd.DataFrame) -> None:

    if df.empty:
        raise ValueError("Dataset is empty.")

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found."
        )

    if df[TARGET].isna().any():
        raise ValueError(
            "Target column contains missing values."
        )

    print("Dataset validation passed.")


# ============================================================
# CHECK MISSING VALUES
# ============================================================

def check_missing_values(df: pd.DataFrame) -> None:

    missing_values = df.isnull().sum()

    print("\nMissing values:")
    print(missing_values)

    total_missing = missing_values.sum()

    if total_missing == 0:
        print("\nNo missing values found.")
    else:
        print(
            f"\nTotal missing values: {total_missing}"
        )


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:

    duplicate_count = df.duplicated().sum()

    print(
        f"\nDuplicate rows found: {duplicate_count}"
    )

    df = df.drop_duplicates().reset_index(drop=True)

    print(
        f"Rows after removing duplicates: {len(df)}"
    )

    return df


# ============================================================
# SPLIT FEATURES AND TARGET
# ============================================================

def split_features_target(df: pd.DataFrame):

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return X, y


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

def save_processed_data(
    X_train,
    X_test,
    y_train,
    y_test
):

    PROCESSED_DATA_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    X_train.to_csv(
        PROCESSED_DATA_PATH / "X_train.csv",
        index=False
    )

    X_test.to_csv(
        PROCESSED_DATA_PATH / "X_test.csv",
        index=False
    )

    y_train.to_csv(
        PROCESSED_DATA_PATH / "y_train.csv",
        index=False
    )

    y_test.to_csv(
        PROCESSED_DATA_PATH / "y_test.csv",
        index=False
    )

    print(
        "\nProcessed datasets saved successfully."
    )


# ============================================================
# SAVE PREPROCESSING CONFIGURATION
# ============================================================

def save_preprocessor(X_train):

    # We don't need StandardScaler or ColumnTransformer
    # for our Random Forest model.
    #
    # Instead, we save the preprocessing configuration
    # used to create the training data.

    preprocessor = {
        "target": TARGET,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "feature_columns": X_train.columns.tolist(),
        "column_cleaning": {
            "strip_whitespace": True,
            "collapse_multiple_spaces": True
        },
        "duplicate_removal": True,
        "scaling": False,
        "encoding": False
    }

    PREPROCESSOR_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH
    )

    print(
        f"\nPreprocessing configuration saved to: "
        f"{PREPROCESSOR_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    df = load_data(RAW_DATA_PATH)

    print(
        f"\nOriginal dataset shape: {df.shape}"
    )

    # --------------------------------------------------------
    # 2. Clean column names
    # --------------------------------------------------------

    df = clean_column_names(df)

    print("\nCleaned columns:")

    print(df.columns.tolist())

    # --------------------------------------------------------
    # 3. Validate data
    # --------------------------------------------------------

    validate_data(df)

    # --------------------------------------------------------
    # 4. Check missing values
    # --------------------------------------------------------

    check_missing_values(df)

    # --------------------------------------------------------
    # 5. Remove duplicates
    # --------------------------------------------------------

    df = remove_duplicates(df)

    print(
        f"\nDataset shape after cleaning: {df.shape}"
    )

    # --------------------------------------------------------
    # 6. Churn distribution
    # --------------------------------------------------------

    print("\nChurn distribution:")

    print(
        df[TARGET].value_counts()
    )

    print("\nChurn distribution (%):")

    print(
        df[TARGET]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    # --------------------------------------------------------
    # 7. Separate features and target
    # --------------------------------------------------------

    X, y = split_features_target(df)

    print(
        f"\nFeatures shape: {X.shape}"
    )

    print(
        f"Target shape: {y.shape}"
    )

    # --------------------------------------------------------
    # 8. Train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    print("\nTrain/Test split:")

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"X_test:  {X_test.shape}"
    )

    print(
        f"y_train: {y_train.shape}"
    )

    print(
        f"y_test:  {y_test.shape}"
    )

    # --------------------------------------------------------
    # 9. Save processed data
    # --------------------------------------------------------

    save_processed_data(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # --------------------------------------------------------
    # 10. Save preprocessing artifact
    # --------------------------------------------------------

    save_preprocessor(X_train)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()