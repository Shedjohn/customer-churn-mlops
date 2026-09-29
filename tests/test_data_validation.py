import pandas as pd

from src.preprocessing import validate_data


def test_validate_data():

    df = pd.DataFrame({
        "Complains": [0],
        "Tariff Plan": [1],
        "Status": [1],
        "Age Group": [3],
        "Charge Amount": [2],
        "Age": [30],
        "Call Failure": [1],
        "Subscription Length": [20],
        "Seconds of Use": [500],
        "Frequency of use": [10],
        "Frequency of SMS": [5],
        "Distinct Called Numbers": [8],
        "Customer Value": [100]
        ,
        "Churn": [0]
    })

    validate_data(df)