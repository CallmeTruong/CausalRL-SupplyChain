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

    def rollout_batch(self, state, order_levels, dis_lead_delta,
                  dis_demand_mult, capacity_ratio, horizon=7):
        n = len(order_levels)

        inventory  = np.full(n, state.inventory)
        backlog    = np.full(n, state.backlog)

        in_transit = np.zeros(n)

        lead_time = self.scm._lead_time(dis_lead_delta, state.noise_lead_time)
        demand    = self.scm._demand(state.demand_forecast,
                                    dis_demand_mult, state.noise_demand)
        actual_orders = np.minimum(order_levels,
                                    self.scm.max_capacity * capacity_ratio)

        stockout_counts = np.zeros(n)
        total_costs     = np.zeros(n)
        total_svcs      = np.zeros(n)
        inv_sum         = np.zeros(n)

        for step in range(horizon):
            # received
            received = np.where(step == int(lead_time), in_transit, 0.0)
            if step == 0:
                in_transit = actual_orders

            # get backlog
            bf        = np.minimum(backlog, received)
            backlog  -= bf
            inventory += (received - bf)

            # sale
            sales     = np.minimum(demand, inventory)
            inventory -= sales
            stockout   = np.maximum(0.0, demand - sales)
            backlog   += stockout

            # new order
            pipeline_order = actual_orders
            arrival        = step + int(lead_time)

            cost = (inventory * self.scm.holding_cost
                + stockout  * self.scm.stockout_penalty
                + np.where(pipeline_order > 0, 2.0, 0.0))
            svc  = np.where(demand == 0, 1.0,
                            np.maximum(0.0, 1.0 - stockout / max(demand, 1)))

            stockout_counts += (stockout > 0).astype(float)
            total_costs     += cost
            total_svcs      += svc
            inv_sum         += inventory

        return {
            "stockout_rate":   stockout_counts / horizon,
            "avg_inventory":   inv_sum         / horizon,
            "total_cost":      total_costs,
            "avg_service_lvl": total_svcs      / horizon,
        }
    

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

        results = self.rollout_batch(
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