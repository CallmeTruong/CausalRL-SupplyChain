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
from demand.lightgbm_trainer import load_m5_single
from demand.feature_engineering import create_features



def get_df(cfg):
    d = cfg["demand"]
    df = load_m5_single(d["sales_path"], d["calendar_path"],
                        d["item_id"], d["store_id"])
    return create_features(df)


def make_env_fn(cfg, seed):
    def _init():
        df = get_df(cfg)
        return SupplyChainEnv(df_history=df, config=cfg, seed=seed)
    return _init


class MetricsCallback(BaseCallback):
    """
    Log add metrics to TensorBoard:
    service_level, stockout_rate, total_cost, disruption_days.
    """

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
                })

        # Log every n steps
        if len(self._episode_metrics) >= 1000:
            df = pd.DataFrame(self._episode_metrics)
            self.logger.record("supply_chain/service_level", df["service_level"].mean())
            self.logger.record("supply_chain/stockout_rate", df["stockout"].mean())
            self.logger.record("supply_chain/avg_cost",      df["total_cost"].mean())
            self.logger.record("supply_chain/disruption_pct",df["dis_active"].mean())
            self._episode_metrics = []

        return True


def train(cfg_path="configs/config.yaml", resume_path=None):
    cfg = yaml.safe_load(open(cfg_path))
    rl  = cfg["rl"]
    
    train_env = make_vec_env(make_env_fn(cfg, seed=42), n_envs=4)
    eval_env  = make_vec_env(make_env_fn(cfg, seed=99), n_envs=1)

    # save best model
    eval_callback = EvalCallback(
        eval_env             = eval_env,
        best_model_save_path = "models/",
        log_path             = "logs/",
        eval_freq            = 10_000,
        n_eval_episodes      = 10,
        deterministic        = True,
        verbose              = 1,
    )

    # save checkpoint every n steps
    checkpoint_callback = CheckpointCallback(
        save_freq   = 100_000,
        save_path   = "models/checkpoints/",
        name_prefix = "ppo_causal",
        verbose     = 1,
    )

    metrics_callback = MetricsCallback()

    callbacks = [eval_callback, checkpoint_callback, metrics_callback]

    # Resume from checkpoint
    if resume_path:
        print(f"Resuming from {resume_path}")
        model = PPO.load(
            resume_path,
            env           = train_env,
            tensorboard_log = "logs/tensorboard/",
        )
        # calc remaining steps
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
        total_timesteps   = steps_remaining,
        callback          = callbacks,
        reset_num_timesteps = resume_path is None,  # False when resume
        progress_bar      = True,
    )

    model.save("models/ppo_causal_final")
    print("Saved → models/ppo_causal_final.zip")


if __name__ == "__main__":
    import sys
    # Train from start:  python rl/train.py
    # Resume:       python rl/train.py models/checkpoints/ppo_causal_100000_steps.zip
    resume = sys.argv[1] if len(sys.argv) > 1 else None
    train(resume_path=resume)