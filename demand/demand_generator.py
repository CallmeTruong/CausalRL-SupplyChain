import pickle
import numpy as np
import pandas as pd
from demand.feature_engineering import create_features


class DemandGenerator:

    def __init__(self, model_path: str, item_key: str, seed=None):
        with open(model_path, "rb") as f:
            artifact = pickle.load(f)

        self.model        = artifact["model"]
        self.residual_std = artifact["residual_std"]
        self.feature_cols = artifact["feature_cols"]
        self.item_encoded = artifact["item_codes"].get(item_key, 0)
        self.rng          = np.random.default_rng(seed)
        self._history     = []

        item_stats       = artifact["item_stats"].get(item_key, {})
        self.demand_mean = float(item_stats.get("mean", 50.0))
        self.demand_std  = float(item_stats.get("std",  20.0))
        self.demand_cv   = (self.demand_std / self.demand_mean
                            if self.demand_mean > 0 else 0.5)

        print(f"DemandGenerator: item_key={item_key}, encoded={self.item_encoded}, "
              f"mean={self.demand_mean:.1f}, cv={self.demand_cv:.2f}")

    def seed_history(self, df: pd.DataFrame):
        self._history = df[["date", "demand"]].to_dict("records")

    def forecast(self, date) -> float:
        row = self._build_row(date)
        if row is None:
            return self._mean_recent()
        return float(max(0.0, self.model.predict(row)[0]))

    def sample(self, date) -> int:
        pred  = self.forecast(date)
        noise = self.rng.normal(0, self.residual_std)
        return max(0, int(round(pred + noise)))

    def record(self, date, demand: int):
        self._history.append({"date": date, "demand": float(demand)})

    # ---------- private ----------

    def _build_row(self, date):
        if len(self._history) < 28:
            return None

        df = pd.DataFrame(self._history)
        df["date"] = pd.to_datetime(df["date"])
        new = pd.DataFrame([{"date": pd.to_datetime(date), "demand": np.nan}])
        df  = pd.concat([df, new], ignore_index=True)
        df  = create_features(df)

        row = df.tail(1).copy()
        row["item_encoded"] = self.item_encoded
        return row[self.feature_cols]

    def _mean_recent(self):
        if not self._history:
            return self.demand_mean
        return float(np.mean([r["demand"] for r in self._history[-28:]]))