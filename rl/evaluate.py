import yaml
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from env.supply_chain_env import SupplyChainEnv
from demand.lightgbm_trainer import load_m5_multi
from demand.feature_engineering import create_features


def get_item_df_map(cfg: dict) -> dict:
    d         = cfg["demand"]
    store_ids = d.get("store_ids", [d.get("store_id", "CA_1")])  # backward compatible

    df = load_m5_multi(
        sales_path    = d["sales_path"],
        calendar_path = d["calendar_path"],
        n_items       = d.get("n_items", 50),
        store_ids     = store_ids,
    )
    item_df_map = {}
    for (store_id, item_id), g in df.groupby(["store_id", "item_id"]):
        key = f"{store_id}__{item_id}"
        item_df_map[key] = create_features(g.copy()).reset_index(drop=True)
    return item_df_map

def heuristic_action(obs, env, s=150, S=400):
    inventory = env.engine.inventory
    if inventory < s:
        target = S - inventory
        idx = int(np.argmin(np.abs(env.order_levels - target)))
        return idx
    return 0

def run_episode(env, model=None) -> dict:
    obs, info = env.reset()
    item_id   = info.get("item_id", "unknown")
    done      = False
    metrics   = {"total_cost": 0.0, "service_levels": [],
                 "stockouts": [], "disruption_days": 0}

    while not done:
        if model is None:
            action = env.action_space.sample()
        elif model == "heuristic":
            action = heuristic_action(obs, env)
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, _, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        metrics["total_cost"]      += info["total_cost"]
        metrics["service_levels"].append(info["service_level"])
        metrics["stockouts"].append(info["stockout"] > 0)
        if info["dis_type"] != 0:
            metrics["disruption_days"] += 1

    return {
        "item_id":         item_id,
        "total_cost":      metrics["total_cost"],
        "service_level":   float(np.mean(metrics["service_levels"])),
        "stockout_rate":   float(np.mean(metrics["stockouts"])),
        "disruption_days": metrics["disruption_days"],
    }


def evaluate(cfg_path="configs/config.yaml", n_episodes=50):
    cfg         = yaml.safe_load(open(cfg_path))
    item_df_map = get_item_df_map(cfg)

    env = SupplyChainEnv(item_df_map=item_df_map, config=cfg, seed=0)

    policies = {
        "Random":        None,
        "Heuristic(s,S)": "heuristic",
        "PPO_Universal": PPO.load("models/best_model"),
    }

    print(f"\nEvaluating over {n_episodes} episodes each ({len(item_df_map)} items)\n")

    for name, model in policies.items():
        episodes = [run_episode(env, model) for _ in range(n_episodes)]
        df       = pd.DataFrame(episodes)

        print(f"{name} — overall:")
        print(f"  service_level : {df['service_level'].mean():.3f} ± {df['service_level'].std():.3f}")
        print(f"  stockout_rate : {df['stockout_rate'].mean():.3f} ± {df['stockout_rate'].std():.3f}")
        print(f"  total_cost    : {df['total_cost'].mean():.0f} ± {df['total_cost'].std():.0f}")

        # Breakdown by item
        item_summary = (df.groupby("item_id")[["service_level", "stockout_rate", "total_cost"]]
                          .mean().sort_values("service_level"))
        print(f"\n  Per-item breakdown (worst 5 service_level):")
        print(item_summary.head(5).to_string())
        print()


if __name__ == "__main__":
    evaluate()