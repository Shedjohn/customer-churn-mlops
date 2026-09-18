from pathlib import Path
import pandas as pd


# Path to the raw dataset
DATA_PATH = Path("data/raw/churn.csv")


def load_data(path: Path) -> pd.DataFrame:
    """
    Load the raw churn dataset.

    This function is responsible only for reading
    the dataset. Data cleaning and preprocessing
    will happen in a separate stage.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}"
        )

    df = pd.read_csv(path)

    return df


def main():
    """Load the dataset and display basic information."""

    df = load_data(DATA_PATH)

    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    return df


if __name__ == "__main__":
    main()