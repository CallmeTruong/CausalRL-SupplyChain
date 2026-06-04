import yaml
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from env.supply_chain_env import SupplyChainEnv
from demand.lightgbm_trainer import load_m5_single
from demand.feature_engineering import create_features


def get_demand_series(cfg):
    d  = cfg["demand"]
    df = load_m5_single(d["sales_path"], d["calendar_path"],
                        d["item_id"], d["store_id"])
    return create_features(df)["demand"].values


def run_episode(env, model=None) -> dict:
    obs, _  = env.reset()
    done    = False
    metrics = {"total_cost": 0.0, "service_levels": [],
               "stockouts": [], "disruption_days": 0}

    while not done:
        if model is None:
            action = env.action_space.sample()
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
        "total_cost":      metrics["total_cost"],
        "service_level":   float(np.mean(metrics["service_levels"])),
        "stockout_rate":   float(np.mean(metrics["stockouts"])),
        "disruption_days": metrics["disruption_days"],
    }


def base_stock_action(env, target=600, reorder=200):
    e        = env.engine
    position = e.inventory + e.pipeline.total_pipeline_quantity()
    order    = max(0, target - position) if position < reorder else 0
    order    = min(order, env.max_order)
    return int(np.argmin(np.abs(env.order_levels - order)))


def evaluate(cfg_path="configs/config.yaml", n_episodes=50):
    cfg           = yaml.safe_load(open(cfg_path))
    demand_series = get_demand_series(cfg)
    env           = SupplyChainEnv(demand_series=demand_series, config=cfg)

    policies = {
        "Random":     None,
        "PPO_Causal": PPO.load("models/best_model"),
    }

    print(f"\nEvaluating over {n_episodes} episodes each\n")

    for name, model in policies.items():
        episodes = []
        for ep in range(n_episodes):
            env._seed = ep * 10
            if name == "Base Stock":
                obs, _ = env.reset()
                done   = False
                res    = {"total_cost": 0.0, "service_levels": [], "stockouts": []}
                while not done:
                    action = base_stock_action(env)
                    obs, _, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    res["total_cost"]      += info["total_cost"]
                    res["service_levels"].append(info["service_level"])
                    res["stockouts"].append(info["stockout"] > 0)
                episodes.append({
                    "total_cost":    res["total_cost"],
                    "service_level": float(np.mean(res["service_levels"])),
                    "stockout_rate": float(np.mean(res["stockouts"])),
                })
            else:
                episodes.append(run_episode(env, model))

        df = pd.DataFrame(episodes)
        print(f"{name}:")
        print(f"  service_level : {df['service_level'].mean():.3f} ± {df['service_level'].std():.3f}")
        print(f"  stockout_rate : {df['stockout_rate'].mean():.3f} ± {df['stockout_rate'].std():.3f}")
        print(f"  total_cost    : {df['total_cost'].mean():.0f} ± {df['total_cost'].std():.0f}\n")


if __name__ == "__main__":
    evaluate()