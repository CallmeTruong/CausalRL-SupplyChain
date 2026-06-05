import yaml
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)

from env.supply_chain_env import SupplyChainEnv
from demand.lightgbm_trainer import load_m5_multi
from demand.feature_engineering import create_features


def get_item_df_map(cfg: dict) -> dict:
    d  = cfg["demand"]
    df = load_m5_multi(
        sales_path    = d["sales_path"],
        calendar_path = d["calendar_path"],
        n_items       = d.get("n_items", 50),
        store_id      = d["store_id"],
    )
    item_df_map = {}
    for item_id, g in df.groupby("item_id"):
        item_df_map[item_id] = create_features(g.copy()).reset_index(drop=True)
    return item_df_map


def make_env_fn(cfg, item_df_map, seed):
    def _init():
        return SupplyChainEnv(item_df_map=item_df_map, config=cfg, seed=seed)
    return _init


class MetricsCallback(BaseCallback):
    """Log metrics to TensorBoard: service_level, stockout_rate, total_cost."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._episode_metrics = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "service_level" in info:
                self._episode_metrics.append({
                    "service_level": info["service_level"],
                    "stockout":      float(info["stockout"] > 0),
                    "total_cost":    info["total_cost"],
                    "dis_active":    float(info["dis_type"] != 0),
                    "svc_demand_only": info["service_level"] if info["demand"] > 0 else None,
                })

        svc_demand = [m["svc_demand_only"] for m in self._episode_metrics
                      if m["svc_demand_only"] is not None]
        if svc_demand:
            self.logger.record("supply_chain/service_level_real",
                               float(np.mean(svc_demand)))

        if len(self._episode_metrics) >= 1000:
            df = pd.DataFrame(self._episode_metrics)
            self.logger.record("supply_chain/service_level", df["service_level"].mean())
            self.logger.record("supply_chain/stockout_rate", df["stockout"].mean())
            self.logger.record("supply_chain/avg_cost",      df["total_cost"].mean())
            self.logger.record("supply_chain/disruption_pct",df["dis_active"].mean())
            self._episode_metrics = []

        return True


def train(cfg_path="configs/config.yaml", resume_path=None):
    cfg          = yaml.safe_load(open(cfg_path))
    rl           = cfg["rl"]
    item_df_map  = get_item_df_map(cfg)

    print(f"Training universal agent over {len(item_df_map)} items.")

    train_env = make_vec_env(make_env_fn(cfg, item_df_map, seed=42),  n_envs=4)
    eval_env  = make_vec_env(make_env_fn(cfg, item_df_map, seed=99),  n_envs=1)

    eval_callback = EvalCallback(
        eval_env             = eval_env,
        best_model_save_path = "models/",
        log_path             = "logs/",
        eval_freq            = 10_000,
        n_eval_episodes      = 10,
        deterministic        = True,
        verbose              = 1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq   = 25_000,
        save_path   = "models/checkpoints/",
        name_prefix = "ppo_universal",
        verbose     = 1,
    )
    metrics_callback = MetricsCallback()
    callbacks = [eval_callback, checkpoint_callback, metrics_callback]

    if resume_path:
        print(f"Resuming from {resume_path}")
        model = PPO.load(
            resume_path,
            env             = train_env,
            tensorboard_log = "logs/tensorboard/",
        )
        steps_done      = model.num_timesteps
        steps_remaining = rl.get("total_timesteps", 1_000_000) - steps_done
        print(f"Already trained: {steps_done:,} | Remaining: {steps_remaining:,}")
    else:
        model = PPO(
            policy          = "MlpPolicy",
            env             = train_env,
            learning_rate   = rl.get("learning_rate", 3e-4),
            n_steps         = rl.get("n_steps", 2048),
            batch_size      = rl.get("batch_size", 64),
            n_epochs        = rl.get("n_epochs", 10),
            gamma           = rl.get("gamma", 0.99),
            gae_lambda      = rl.get("gae_lambda", 0.95),
            clip_range      = rl.get("clip_range", 0.2),
            ent_coef        = rl.get("ent_coef", 0.01),
            policy_kwargs   = dict(net_arch=[256, 256]),
            verbose         = 1,
            tensorboard_log = "logs/tensorboard/",
        )
        steps_remaining = rl.get("total_timesteps", 1_000_000)

    model.learn(
        total_timesteps     = steps_remaining,
        callback            = callbacks,
        reset_num_timesteps = resume_path is None,
        progress_bar        = True,
    )

    model.save("models/ppo_universal_final")
    print("Saved → models/ppo_universal_final.zip")


if __name__ == "__main__":
    import sys
    resume = sys.argv[1] if len(sys.argv) > 1 else None
    train(resume_path=resume)