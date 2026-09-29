import pandas as pd

from src.preprocessing import clean_column_names


def test_clean_column_names():

    df = pd.DataFrame({
        "Call  Failure": [1],
        "Subscription  Length": [20],
        "Customer Value": [100]
    })

    cleaned = clean_column_names(df)

    assert "Call Failure" in cleaned.columns

    assert "Subscription Length" in cleaned.columns

    assert "Customer Value" in cleaned.columns