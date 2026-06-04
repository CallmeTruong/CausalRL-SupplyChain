import numpy as np
import gymnasium as gym
from gymnasium import spaces

from simulator.supply_chain_engine import SupplyChainEngine
from simulator.disruption_engine import DisruptionEngine
from causal.scm import SCM
from causal.counterfactual_engine import CounterfactualEngine


class SupplyChainEnv(gym.Env):
    """
    Observation: 23 dims = base(10) + disruption(5) + counterfactual(8)
    Action:      discrete, N level order from 0 to max_order
    Reward:      service_level_bonus - cost_penalty
    """

    def __init__(self, demand_series, config: dict, seed=None):
        super().__init__()

        sim = config["simulation"]
        rl  = config["rl"]
        dis = config["disruption"]

        self.episode_length = sim["episode_length"]
        self.max_order      = rl["max_order"]
        self.order_levels   = np.linspace(0, rl["max_order"], rl["n_order_levels"], dtype=int)

        self.action_space      = spaces.Discrete(rl["n_order_levels"])
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(23,), dtype=np.float32)

        self._demand_series = np.array(demand_series)
        self._sim_cfg       = sim
        self._dis_cfg       = dis
        self._seed          = seed

        scm = SCM(
            base_lead_time   = sim["base_lead_time"],
            max_capacity     = sim["max_supplier_capacity"],
            holding_cost     = sim["holding_cost"],
            stockout_penalty = sim["stockout_penalty"],
        )
        self._cf = CounterfactualEngine(
            scm          = scm,
            max_order    = rl["max_order"],
            n_candidates = rl["cf_n_candidates"],
            horizon      = rl["cf_horizon"],
        )

        self.engine       = None
        self._step_count  = 0
        self._recent_demands = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        _seed = seed or self._seed

        self.engine = SupplyChainEngine(
            demand_series         = self._demand_series,
            disruption_engine     = DisruptionEngine(
                mean_inter_arrival = self._dis_cfg["mean_inter_arrival"],
                seed               = _seed,
            ),
            **{k: v for k, v in self._sim_cfg.items() if k != "episode_length"},
            seed = _seed,
        )
        self.engine.reset()
        self._step_count     = 0
        self._recent_demands = []

        obs = self._get_obs(demand=50.0, lead_time=float(self._sim_cfg["base_lead_time"]),
                            dis_lead_delta=0.0, dis_demand_mult=1.0, capacity_ratio=1.0)
        return obs, {}

    def step(self, action: int):
        order  = int(self.order_levels[action])
        info   = self.engine.step(order)

        self._recent_demands.append(info["demand"])
        if len(self._recent_demands) > 7:
            self._recent_demands.pop(0)

        obs = self._get_obs(
            demand          = float(info["demand"]),
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

    def _get_obs(self, demand, lead_time, dis_lead_delta, dis_demand_mult, capacity_ratio):
        e           = self.engine
        max_inv     = self._sim_cfg["initial_inventory"] * 3
        max_lt      = self._sim_cfg["base_lead_time"] + 20
        demand_mean = float(np.mean(self._recent_demands)) if self._recent_demands else 50.0

        base = np.array([
            e.inventory / max_inv,
            e.backlog   / max_inv,
            e.pipeline.total_pipeline_quantity() / max_inv,
            demand / max(demand_mean * 3, 1),
            demand_mean / max_inv,
            lead_time / max_lt,
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
            demand_forecast = demand_mean,
            dis_lead_delta  = dis_lead_delta,
            dis_demand_mult = dis_demand_mult,
            capacity_ratio  = capacity_ratio,
        )

        return np.concatenate([base, dis_vec, cf_vec])

    def _reward(self, info: dict) -> float:
        service_bonus    =  5.0 * info["service_level"]
        cost_penalty     =  info["total_cost"] / 1000.0
        disruption_bonus = (1.0 if info["dis_type"] != 0
                            and info["service_level"] > 0.9 else 0.0)
        return float(np.clip(service_bonus - cost_penalty + disruption_bonus, -10.0, 10.0))