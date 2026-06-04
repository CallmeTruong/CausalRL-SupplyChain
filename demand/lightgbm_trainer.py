import pickle
import numpy as np
import pandas as pd
import yaml
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from feature_engineering import create_features, FEATURE_COLS


def load_m5_multi(sales_path, calendar_path, n_items=50, store_id="CA_1"):

    sales    = pd.read_csv(sales_path)
    calendar = pd.read_csv(calendar_path, parse_dates=["date"])

    # get n_items of store_id
    rows = sales[sales["store_id"] == store_id].head(n_items)

    day_cols = [c for c in sales.columns if c.startswith("d_")]
    n_days   = len(day_cols)
    dates    = calendar["date"].values[:n_days]
    snap     = calendar["snap_CA"].values[:n_days]
    holiday  = (calendar["event_name_1"].fillna("").values[:n_days] != "").astype(int)

    all_dfs = []
    for _, row in rows.iterrows():
        df = pd.DataFrame({
            "date":       dates,
            "demand":     row[day_cols].values.astype(float),
            "snap":       snap,
            "is_holiday": holiday,
            "item_id":    row["item_id"],
            "store_id":   row["store_id"],
        })
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


def load_m5_single(sales_path, calendar_path, item_id, store_id):
    sales    = pd.read_csv(sales_path)
    calendar = pd.read_csv(calendar_path, parse_dates=["date"])

    row      = sales[(sales["item_id"] == item_id) & (sales["store_id"] == store_id)]
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    n        = len(day_cols)

    return pd.DataFrame({
        "date":       calendar["date"].values[:n],
        "demand":     row[day_cols].values.flatten().astype(float),
        "snap":       calendar["snap_CA"].values[:n],
        "is_holiday": (calendar["event_name_1"].fillna("").values[:n] != "").astype(int),
    })


MULTI_FEATURE_COLS = FEATURE_COLS + ["item_encoded"]


def train(cfg: dict):
    d = cfg["demand"]

    print(f"Loading {d.get('n_items', 50)} items from {d['store_id']}")
    df = load_m5_multi(
        sales_path    = d["sales_path"],
        calendar_path = d["calendar_path"],
        n_items       = d.get("n_items", 50),
        store_id      = d["store_id"],
    )

    item_codes = {item: i for i, item in enumerate(df["item_id"].unique())}

    dfs = []

    for item_id, g in df.groupby("item_id"):
        tmp = create_features(g)
        tmp["item_encoded"] = item_codes[item_id]
        dfs.append(tmp)

    df = pd.concat(dfs, ignore_index=True)

    #train/val
    cutoff  = df["date"].quantile(0.8)
    X_train = df[df["date"] <= cutoff][MULTI_FEATURE_COLS]
    y_train = df[df["date"] <= cutoff]["demand"]
    X_val   = df[df["date"] >  cutoff][MULTI_FEATURE_COLS]
    y_val   = df[df["date"] >  cutoff]["demand"]

    model = lgb.LGBMRegressor(
        n_estimators  = 500,
        learning_rate = 0.05,
        num_leaves    = 63,
        random_state  = 42,
        verbosity     = -1,
    )
    model.fit(
        X_train, y_train,
        eval_set  = [(X_val, y_val)],
        callbacks = [lgb.early_stopping(50, verbose=False)],
    )

    residuals    = y_val.values - model.predict(X_val)
    residual_std = float(np.std(residuals))
    mae          = mean_absolute_error(y_val, model.predict(X_val))
    print(f"MAE: {mae:.2f} | Residual std: {residual_std:.2f}")
    print(f"Items trained: {len(item_codes)}")

    artifact = {
        "model":        model,
        "residual_std": residual_std,
        "item_codes":   item_codes,          # use in inference
        "feature_cols": MULTI_FEATURE_COLS,
    }
    with open(d["model_path"], "wb") as f:
        pickle.dump(artifact, f)
    print(f"Saved → {d['model_path']}")


if __name__ == "__main__":
    train(yaml.safe_load(open("configs/config.yaml")))