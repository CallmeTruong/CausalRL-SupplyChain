import numpy as np
from causal.scm import SCM


class CounterfactualEngine:
    """
    each step:
    1. Abduction  — infer noise from today observation
    2. Intervention — try N order quantity level
    3. Prediction — rollout SCM -> 8 features
    """

    def __init__(self, scm: SCM, order_levels, horizon=14):
        self.scm        = scm
        self.horizon    = horizon
        self.candidates = order_levels

    def compute(self, inventory, backlog, lead_time, demand,
            demand_forecast, dis_lead_delta, dis_demand_mult,
            capacity_ratio) -> np.ndarray:

        state = self.scm.abduct(
            observed_inventory  = inventory,
            observed_backlog    = backlog,
            observed_lead_time  = lead_time,
            observed_demand     = demand,
            demand_forecast     = demand_forecast,
            dis_lead_delta      = dis_lead_delta,
        )

        results = self.rollout(
            state           = state,
            order_levels    = self.candidates,
            dis_lead_delta  = dis_lead_delta,
            dis_demand_mult = dis_demand_mult,
            capacity_ratio  = capacity_ratio,
            horizon         = self.horizon,
        )

        stockout_rates = results["stockout_rate"]
        service_levels = results["avg_service_lvl"]
        total_costs    = results["total_cost"]
        avg_inventories= results["avg_inventory"]

        best_idx   = int(np.argmin(total_costs))
        cost_mean  = float(np.mean(total_costs))
        cost_sens  = float(np.std(total_costs) / cost_mean) if cost_mean > 0 else 0.0
        mid_idx    = len(self.candidates) // 2

        return np.clip(np.array([
            float(np.min(stockout_rates)),
            float(np.max(service_levels)),
            float(np.min(total_costs)) / 1000.0,
            float(stockout_rates[mid_idx]),
            float(avg_inventories[best_idx]) / 500.0,
            cost_sens,
            float(np.mean(stockout_rates > 0.1)),
            float(self.candidates[best_idx]) / float(max(self.candidates)),
        ], dtype=np.float32), -5.0, 5.0)