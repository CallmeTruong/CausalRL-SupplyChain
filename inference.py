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


def run_episode(env, model, item_key: str) -> list[dict]:
    obs, info = env.reset(options={"item_key": item_key})
    done = False
    log  = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        log.append({
            "day":         len(log) + 1,
            "inventory":   info["inventory"],
            "demand":      info["demand"],
            "order":       info["actual_order"],
            "stockout":    info["stockout"],
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

    return log


def print_summary(item_key: str, log: list[dict]):
    total_cost    = sum(r["cost"] for r in log)
    avg_svc       = np.mean([r["service_lvl"] for r in log])
    stockout_rate = np.mean([r["stockout"] > 0 for r in log])
    print(f"\n=== Summary: {item_key} ===")
    print(f"Total Cost    : {total_cost:,.1f}")
    print(f"Service Level : {avg_svc:.3f}")
    print(f"Stockout Rate : {stockout_rate:.3f}")


def run_inference(
    model_path: str = "models/best_model",
    seed: int = 42,
    items: list[str] | None = None,
):
    cfg   = yaml.safe_load(open("configs/config.yaml"))
    model = PPO.load(model_path)
    env   = load_env(cfg, seed=seed)

    all_keys    = list(env._item_df_map.keys())
    target_keys = items if items else all_keys

    all_results = {}

    for item_key in target_keys:
        print(f"\n{'='*60}")
        print(f"Item: {item_key}")
        print('='*60)

        log = run_episode(env, model, item_key)
        print_summary(item_key, log)
        all_results[item_key] = log

    print(f"\n{'='*60}")
    print("=== OVERALL SUMMARY ===")
    all_svc   = [r["service_lvl"] for log in all_results.values() for r in log]
    all_stk   = [r["stockout"] > 0 for log in all_results.values() for r in log]
    all_cost  = [sum(r["cost"] for r in log) for log in all_results.values()]
    print(f"Items tested  : {len(all_results)}")
    print(f"Avg Svc Level : {np.mean(all_svc):.3f}")
    print(f"Stockout Rate : {np.mean(all_stk):.3f}")
    print(f"Avg Cost/item : {np.mean(all_cost):,.1f}")
    print(f"Total Cost    : {sum(all_cost):,.1f}")

    return all_results


if __name__ == "__main__":
    # run for all item and store
    # run_inference()

    # run with specific item
    run_inference(items=["CA_1__HOBBIES_1_023", "CA_2__FOODS_1_001"])