import os
import yaml
import pickle
import argparse
import pandas as pd
from pathlib import Path
from rl.train import train as train_rl
from rl.evaluate import evaluate
from demand.lightgbm_trainer import train as train_demand


def load_config():
    return yaml.safe_load(open("configs/config.yaml"))


def step1_train_demand(cfg):
    """Train LightGBM demand model"""
    model_path = cfg["demand"]["model_path"]

    if Path(model_path).exists():
        print(f"[1/3] Demand model found at {model_path}, skipping.")
        return

    print("[1/3] Training demand model...")
    train_demand(cfg)
    print("[1/3] Done.\n")


def step2_train_rl(cfg, resume_path=None):
    print("[2/3] Training RL agent...")
    train_rl(
        cfg_path="configs/config.yaml",
        resume_path=resume_path
    )
    print("[2/3] Done.\n")


def step3_evaluate(cfg):
    """Evaluate"""
    print("[3/3] Evaluating policies...")
    evaluate(cfg_path="configs/config.yaml", n_episodes=50)
    print("[3/3] Done.\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-demand", action="store_true",
                        help="Skip train demand model")
    parser.add_argument("--skip-train",  action="store_true",
                        help="Skip train RL, only evaluate")
    parser.add_argument("--resume",      type=str, default=None,
                        help="Path checkpoint for resume train RL")
    args = parser.parse_args()

    for d in ["models/checkpoints", "logs/tensorboard"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    cfg = load_config()

    if not args.skip_demand:
        step1_train_demand(cfg)

    if not args.skip_train:
        step2_train_rl(cfg, resume_path=args.resume)

    step3_evaluate(cfg)