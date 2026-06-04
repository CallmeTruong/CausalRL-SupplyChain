import pandas as pd

def create_features(df):

    df = df.copy()

    df["lag_1"] = df["demand"].shift(1)

    df["lag_7"] = df["demand"].shift(7)

    df["lag_14"] = df["demand"].shift(14)

    df["lag_28"] = df["demand"].shift(28)

    df["rolling_mean_7"] = (
        df["demand"]
        .rolling(7)
        .mean()
    )

    df["rolling_mean_28"] = (
        df["demand"]
        .rolling(28)
        .mean()
    )

    df["day_of_week"] = df["date"].dt.dayofweek

    df["month"] = df["date"].dt.month

    return df.dropna()