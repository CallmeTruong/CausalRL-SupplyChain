import numpy as np
from causal.scm import SCM


class CounterfactualEngine:
    """
    each step:
    1. Abduction  — infer noise from today observation
    2. Intervention — try N order quantity level
    3. Prediction — rollout SCM -> 8 features
    """

    def __init__(self, scm: SCM, max_order=500, n_candidates=10, horizon=14):
        self.scm        = scm
        self.horizon    = horizon
        self.candidates = np.linspace(0, max_order, n_candidates, dtype=int)

    def compute(
        self,
        inventory:       float,
        backlog:         float,
        lead_time:       float,
        demand:          float,
        demand_forecast: float,
        dis_lead_delta:  float,
        dis_demand_mult: float,
        capacity_ratio:  float,
    ) -> np.ndarray:
        """vector 8 dims."""

        # step 1: Abduction
        state = self.scm.abduct(
            observed_inventory  = inventory,
            observed_backlog    = backlog,
            observed_lead_time  = lead_time,
            observed_demand     = demand,
            demand_forecast     = demand_forecast,
            dis_lead_delta      = dis_lead_delta,
        )

        # step 2+3: Intervention + Prediction for each candidate
        results = [
            self.scm.rollout(
                state           = state,
                order_quantity  = int(q),
                dis_lead_delta  = dis_lead_delta,
                dis_demand_mult = dis_demand_mult,
                capacity_ratio  = capacity_ratio,
                horizon         = self.horizon,
            )
            for q in self.candidates
        ]

        stockout_rates = [r["stockout_rate"]   for r in results]
        service_levels = [r["avg_service_lvl"] for r in results]
        total_costs    = [r["total_cost"]       for r in results]
        avg_inventories= [r["avg_inventory"]    for r in results]

        best_idx   = int(np.argmin(total_costs))
        cost_mean  = float(np.mean(total_costs))
        cost_sens  = float(np.std(total_costs) / cost_mean) if cost_mean > 0 else 0.0
        mid_idx    = len(self.candidates) // 2

        return np.clip(np.array([
            min(stockout_rates),                         # best-case stockout rate
            max(service_levels),                         # best-case service level
            min(total_costs) / 1000.0,                   # best-case cost
            stockout_rates[mid_idx],                     # stockout if order mean
            avg_inventories[best_idx] / 500.0,           # inventory at optimal
            cost_sens,                                   # sens of cost/ action
            float(np.mean([s > 0.1 for s in stockout_rates])),  # % candidate with risk
            float(self.candidates[best_idx]) / float(max(self.candidates)),  # optimal order
        ], dtype=np.float32), -5.0, 5.0)