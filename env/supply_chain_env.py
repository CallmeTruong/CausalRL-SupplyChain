import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from simulator.supply_chain_engine import SupplyChainEngine
from simulator.disruption_engine import DisruptionEngine
from demand.demand_generator import DemandGenerator
from causal.scm import SCM
from causal.counterfactual_engine import CounterfactualEngine


class SupplyChainEnv(gym.Env):

    def __init__(self, item_df_map: dict, config: dict, seed=None):
        """
        item_df_map: dict[item_id -> DataFrame]
        """
        super().__init__()

        sim = config["simulation"]
        rl  = config["rl"]
        dis = config["disruption"]

        self.episode_length = sim["episode_length"]
        self.max_order      = rl["max_order"]
        self.order_levels   = np.linspace(0, rl["max_order"], rl["n_order_levels"], dtype=int)

        self.action_space      = spaces.Discrete(rl["n_order_levels"])
        # 27 features
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(27,), dtype=np.float32)

        self._item_df_map = item_df_map          # dict item_id -> df
        self._item_ids    = list(item_df_map.keys())
        self._sim_cfg     = sim
        self._dis_cfg     = dis
        self._model_path  = config["demand"]["model_path"]
        self._seed        = seed
        self._rng         = np.random.default_rng(seed)

        scm = SCM(
            base_lead_time   = sim["base_lead_time"],
            max_capacity     = sim["max_supplier_capacity"],
            holding_cost     = sim["holding_cost"],
            stockout_penalty = sim["stockout_penalty"],
        )
        self._cf = CounterfactualEngine(
            scm          = scm,
            order_levels = self.order_levels,
            horizon      = rl["cf_horizon"],
        )

        self.engine         = None
        self._step_count    = 0
        self._last_forecast = 50.0
        self._current_item  = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Random choose item per episode
        self._current_item = self._rng.choice(self._item_ids)
        df_history         = self._item_df_map[self._current_item]
        _seed              = int(self._rng.integers(0, 2**31))

        gen = DemandGenerator(
            model_path = self._model_path,
            item_id    = self._current_item,
            seed       = _seed,
        )
        gen.seed_history(df_history.head(56))

        self.engine = SupplyChainEngine(
            demand_generator  = gen,
            dates             = df_history["date"].values,
            disruption_engine = DisruptionEngine(
                mean_inter_arrival = self._dis_cfg["mean_inter_arrival"],
                seed               = _seed,
            ),
            **{k: v for k, v in self._sim_cfg.items() if k != "episode_length"},
            seed = _seed,
        )
        self.engine.reset()
        self._step_count    = 0
        self._last_forecast = gen.demand_mean

        obs = self._get_obs(
            demand          = gen.demand_mean,
            demand_forecast = gen.demand_mean,
            lead_time       = float(self._sim_cfg["base_lead_time"]),
            dis_lead_delta  = 0.0,
            dis_demand_mult = 1.0,
            capacity_ratio  = 1.0,
        )
        return obs, {"item_id": self._current_item}

    def step(self, action: int):
        order = int(self.order_levels[action])
        info  = self.engine.step(order)

        self._last_forecast = info["demand_forecast"]

        obs = self._get_obs(
            demand          = float(info["demand"]),
            demand_forecast = float(info["demand_forecast"]),
            lead_time       = float(info["lead_time"]),
            dis_lead_delta  = float(info["dis_lead_delta"]),
            dis_demand_mult = float(info["dis_demand_mult"]),
            capacity_ratio  = info["capacity"] / self._sim_cfg["max_supplier_capacity"],
        )
        reward = self._reward(info)

        self._step_count += 1
        done = info["done"] or self._step_count >= self.episode_length

        return obs, reward, done, False, info

    # ---------- private ----------

    def _get_obs(self, demand, demand_forecast, lead_time,
                 dis_lead_delta, dis_demand_mult, capacity_ratio):
        e       = self.engine
        gen     = e.demand_generator
        max_inv = self._sim_cfg["initial_inventory"] * 3
        max_lt  = self._sim_cfg["base_lead_time"] + 20

        base = np.array([
            e.inventory / max_inv,
            e.backlog   / max_inv,
            e.pipeline.total_pipeline_quantity() / max_inv,
            (e.inventory + e.pipeline.total_pipeline_quantity() - e.backlog) / max_inv,
            demand          / max(demand_forecast * 3, 1),
            demand_forecast / max_inv,
            lead_time  / max_lt,
            dis_lead_delta / 20.0,
            self._step_count / self.episode_length,
            np.sin(2 * np.pi * self._step_count / 7),
            np.cos(2 * np.pi * self._step_count / 7),
        ], dtype=np.float32)

        dis_vec = e.disruption_engine.as_vector()

        cf_vec = self._cf.compute(
            inventory       = float(e.inventory),
            backlog         = float(e.backlog),
            lead_time       = lead_time,
            demand          = demand,
            demand_forecast = demand_forecast,
            dis_lead_delta  = dis_lead_delta,
            dis_demand_mult = dis_demand_mult,
            capacity_ratio  = capacity_ratio,
        )

        # Product context
        product_ctx = np.array([
            np.clip(gen.demand_cv, 0.0, 3.0),
            np.clip(gen.demand_mean / max_inv, 0.0, 1.0),
            np.clip(self._sim_cfg["base_lead_time"] / max_lt, 0.0, 1.0),  # lead time ratio
        ], dtype=np.float32)

        return np.concatenate([base, dis_vec, cf_vec, product_ctx])

    def _reward(self, info: dict) -> float:
        expected_daily_cost = (
            self._sim_cfg["initial_inventory"] * self._sim_cfg["holding_cost"]
        )
        service_bonus    = 5.0 * info["service_level"]
        cost_penalty     = 5.0 * (info["total_cost"] / expected_daily_cost)
        disruption_bonus = (1.0 if info["dis_type"] != 0
                            and info["service_level"] > 0.9 else 0.0)
        return float(np.clip(service_bonus - cost_penalty + disruption_bonus,
                             -10.0, 10.0))