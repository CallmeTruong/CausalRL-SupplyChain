import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy().sort_values("date").reset_index(drop=True)

    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = df["demand"].shift(lag)

    for w in [7, 28]:
        df[f"rolling_mean_{w}"] = df["demand"].shift(1).rolling(w).mean()

    df["day_of_week"]  = df["date"].dt.dayofweek
    df["month"]        = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    for col in ["snap", "is_holiday"]:
        if col not in df.columns:
            df[col] = 0

    return df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

FEATURE_COLS = [
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_28",
    "day_of_week", "month", "week_of_year",
    "snap", "is_holiday",
]