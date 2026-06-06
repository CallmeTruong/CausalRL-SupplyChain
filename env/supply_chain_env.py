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
        super().__init__()

        sim = config["simulation"]
        rl  = config["rl"]
        dis = config["disruption"]

        self.episode_length = sim["episode_length"]
        self.max_order      = rl["max_order"]
        self.n_order_levels = rl["n_order_levels"]
        self.order_levels   = np.linspace(0, rl["max_order"], rl["n_order_levels"], dtype=int)

        self.action_space      = spaces.Discrete(rl["n_order_levels"])
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(27,), dtype=np.float32)

        self._item_df_map = item_df_map
        self._item_ids    = list(item_df_map.keys())
        self._sim_cfg     = sim
        self._dis_cfg     = dis
        self._model_path  = config["demand"]["model_path"]
        self._seed        = seed
        self._rng         = np.random.default_rng(seed)

        self._scm = SCM(
            base_lead_time      = sim["base_lead_time"],
            max_capacity        = sim["max_supplier_capacity"],
            holding_cost        = sim["holding_cost"],
            stockout_penalty    = sim["stockout_penalty"],
            backlog_cost        = sim.get("backlog_cost",        3.0),
            order_cost_fixed    = sim.get("order_cost_fixed",    2.0),
            order_cost_variable = sim.get("order_cost_variable", 0.05),
        )
        self._cf_horizon = rl["cf_horizon"]

        self._cf                = None
        self._item_order_levels = self.order_levels  # placeholder

        self.engine        = None
        self._step_count   = 0
        self._current_item = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._current_item = self._rng.choice(self._item_ids)
        df_history         = self._item_df_map[self._current_item]
        _seed              = int(self._rng.integers(0, 2**31))

        gen = DemandGenerator(
            model_path = self._model_path,
            item_key   = self._current_item,
            seed       = _seed,
        )
        gen.seed_history(df_history.head(56))

        demand_ref = max(gen.demand_mean, 1.0)

        item_initial_inv = max(int(demand_ref * self._sim_cfg["base_lead_time"] * 2), 5)

        item_max_order          = max(int(demand_ref * self._sim_cfg["base_lead_time"] * 10), 10)
        self._item_order_levels = np.linspace(0, item_max_order, self.n_order_levels, dtype=int)

        self._cf = CounterfactualEngine(
            scm          = self._scm,
            order_levels = self._item_order_levels,
            horizon      = self._cf_horizon,
        )

        # Override initial_inventory
        _engine_keys = {
            "initial_inventory", "base_lead_time", "max_supplier_capacity",
            "holding_cost", "stockout_penalty", "backlog_cost",
            "order_cost_fixed", "order_cost_variable",
        }
        sim_kwargs = {k: v for k, v in self._sim_cfg.items()
                    if k in _engine_keys}
        sim_kwargs["initial_inventory"] = item_initial_inv

        self.engine = SupplyChainEngine(
            demand_generator  = gen,
            dates             = df_history["date"].values,
            disruption_engine = DisruptionEngine(
                mean_inter_arrival = self._dis_cfg["mean_inter_arrival"],
                seed               = _seed,
            ),
            **sim_kwargs,
            seed = _seed,
        )
        self.engine.reset()
        self._step_count = 0

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
        order = int(self._item_order_levels[action])
        info  = self.engine.step(order)

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
        e          = self.engine
        gen        = e.demand_generator
        max_lt     = self._sim_cfg["base_lead_time"] + 20
        demand_ref = max(gen.demand_mean, 1.0)

        inv_ref = max(
            self._sim_cfg["base_lead_time"] * demand_ref * 10.0,
            float(max(self._item_order_levels)),
        )

        def log_norm(x):
            return np.log1p(max(float(x), 0.0)) / np.log1p(inv_ref)

        inv_pos = e.inventory + e.pipeline.total_pipeline_quantity() - e.backlog

        base = np.array([
            log_norm(e.inventory),
            log_norm(e.backlog),
            log_norm(e.pipeline.total_pipeline_quantity()),
            log_norm(max(inv_pos, 0)),
            demand          / max(demand_ref * 3.0, 1.0),
            demand_forecast / max(demand_ref * 3.0, 1.0),
            lead_time       / max_lt,
            dis_lead_delta  / 20.0,
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

        product_ctx = np.array([
            np.clip(gen.demand_cv, 0.0, 3.0),
            demand_ref / max(float(self.max_order), 1.0),
            np.clip(self._sim_cfg["base_lead_time"] / max_lt, 0.0, 1.0),
        ], dtype=np.float32)

        return np.concatenate([base, dis_vec, cf_vec, product_ctx])

    def _reward(self, info: dict) -> float:
        gen        = self.engine.demand_generator
        demand_ref = max(gen.demand_mean, 1.0)

        target_inv             = max(self._sim_cfg["base_lead_time"] * demand_ref * 2.0, 1.0)
        expected_daily_holding = target_inv * self._sim_cfg["holding_cost"]

        normalized_cost = info["total_cost"] / max(expected_daily_holding, 0.1)
        cost_penalty    = np.sqrt(normalized_cost)

        overstock         = max(0.0, self.engine.inventory - target_inv * 3.0)
        overstock_penalty = 0.01 * overstock / max(demand_ref, 1.0)

        service_score = 2.0 * info["service_level"]
        dis_bonus     = 0.5 if (info["dis_type"] != 0 and info["service_level"] > 0.9) else 0.0

        return float(np.clip(
            service_score - cost_penalty - overstock_penalty + dis_bonus,
            -30.0, 3.5
        ))