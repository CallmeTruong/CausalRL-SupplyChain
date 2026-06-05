import yaml
import numpy as np
from stable_baselines3 import PPO

from env.supply_chain_env import SupplyChainEnv
from demand.lightgbm_trainer import load_m5_multi
from demand.feature_engineering import create_features


def load_env(cfg, seed=0):
    d         = cfg["demand"]
    store_ids = d.get("store_ids", [d.get("store_id", "CA_1")])

    df = load_m5_multi(
        sales_path=d["sales_path"],
        calendar_path=d["calendar_path"],
        n_items=d.get("n_items", 50),
        store_ids=store_ids,
    )

    item_df_map = {}
    for (store_id, item_id), g in df.groupby(["store_id", "item_id"]):
        key = f"{store_id}__{item_id}"
        item_df_map[key] = create_features(g.copy()).reset_index(drop=True)

    return SupplyChainEnv(item_df_map=item_df_map, config=cfg, seed=seed)


def run_inference(model_path="models/best_model", seed=42):
    cfg   = yaml.safe_load(open("configs/config.yaml"))
    model = PPO.load(model_path)
    env   = load_env(cfg, seed=seed)

    obs, info = env.reset()
    done      = False

    total_cost     = 0.0
    service_levels = []
    log            = []

    print(f"Item: {info.get('item_id', 'unknown')}\n")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        total_cost += info["total_cost"]
        service_levels.append(info["service_level"])

        log.append({
            "day":         len(log) + 1,
            "inventory":   info["inventory"],
            "demand":      info["demand"],
            "order":       info["actual_order"],
            "stockout":    info["stockout"],
            "lead_time":   info["lead_time"],
            "disruption":  info["dis_type"],
            "service_lvl": info["service_level"],
            "cost":        info["total_cost"],
        })

        print(
            f"Day {log[-1]['day']:3d} | "
            f"Inv: {info['inventory']:5d} | "
            f"Demand: {info['demand']:4d} | "
            f"Order: {info['actual_order']:4d} | "
            f"Stockout: {info['stockout']:3d} | "
            f"Dis: {info['dis_type']} | "
            f"SvcLvl: {info['service_level']:.2f}"
        )

    print("\n=== Episode Summary ===")
    print(f"Total Cost    : {total_cost:,.1f}")
    print(f"Service Level : {np.mean(service_levels):.3f}")
    print(f"Stockout Rate : {np.mean([r['stockout'] > 0 for r in log]):.3f}")

    return log


if __name__ == "__main__":
    run_inference()