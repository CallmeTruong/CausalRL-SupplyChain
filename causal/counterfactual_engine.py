import numpy as np
from causal.scm import SCM


class CounterfactualEngine:
    """
    At each environment step, generates 8 counterfactual features by:

    1. Abduction  — infer noise from today's observation
    2. Intervention — try N candidate order quantities (do(OrderQuantity = q))
    3. Prediction  — stochastic rollout of 7-day future trajectory
                    for each candidate, using real-world dynamics

    Key invariants maintained between SCM and real environment:
    - Cost parameters: holding_cost, stockout_penalty, backlog_cost,
      order_cost_fixed, order_cost_variable all match real engine
    - Demand is stochastic: drawn from N(forecast, residual_std) each step
    - Order is placed ONCE at step 0 only (the intervention under test)
    - Backlog is tracked across steps
    - Lead time is fixed after abduction (intervention does not change it)
    """

    def __init__(self, scm: SCM, order_levels, horizon=14, seed=None):
        self.scm        = scm
        self.horizon    = horizon
        self.candidates = order_levels
        self.rng        = np.random.default_rng(seed)

    def rollout_batch(self, state, order_levels, dis_lead_delta,
                      dis_demand_mult, capacity_ratio, horizon=7):
        n = len(order_levels)

        inventory  = np.full(n, float(state.inventory))
        backlog    = np.full(n, float(state.backlog))
        in_transit = np.zeros(n)

        # Lead time is fixed after abduction — intervention does not change it
        lead_time = self.scm._lead_time(dis_lead_delta, state.noise_lead_time)
        # Capacity is fixed after abduction — intervention does not change it
        actual_orders = np.minimum(
            order_levels,
            self.scm.max_capacity * capacity_ratio
        )

        stockout_counts = np.zeros(n)
        total_costs     = np.zeros(n)
        total_svcs      = np.zeros(n)
        inv_sum         = np.zeros(n)

        for step in range(horizon):
            # Demand is STOCHASTIC: draw fresh noise each step
            # This matches the real environment's DemandGenerator.sample()
            step_noise = self.rng.normal(0.0, state.residual_std, size=n)
            demand = np.maximum(
                0.0,
                state.demand_forecast * dis_demand_mult
                + state.noise_demand  # persistent deviation from forecast
                + step_noise           # new random shock each day
            )

            # Receive: orders placed at step 0 arrive after lead_time days
            received = np.where(step == int(lead_time), in_transit, 0.0)
            if step == 0:
                in_transit = actual_orders

            # Fulfill backlog first, then inventory
            bf          = np.minimum(backlog, received)
            backlog    -= bf
            inventory  += (received - bf)

            # Sales and stockout
            sales     = np.minimum(demand, inventory)
            inventory -= sales
            stockout   = np.maximum(0.0, demand - sales)
            backlog   += stockout

            # Cost: all components match real SupplyChainEngine
            cost = (
                inventory   * self.scm.holding_cost
              + stockout    * self.scm.stockout_penalty
              + backlog     * self.scm.backlog_cost
              + np.where(actual_orders > 0, self.scm.order_cost_fixed, 0.0)
              + actual_orders * self.scm.order_cost_variable
            )

            # Service level
            svc = np.where(
                demand == 0, 1.0,
                np.maximum(0.0, 1.0 - stockout / np.maximum(demand, 1))
            )

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
               capacity_ratio, residual_std=0.0) -> np.ndarray:

        state = self.scm.abduct(
            observed_inventory  = inventory,
            observed_backlog    = backlog,
            observed_lead_time  = lead_time,
            observed_demand     = demand,
            demand_forecast     = demand_forecast,
            dis_lead_delta      = dis_lead_delta,
            residual_std        = residual_std,
        )

        results = self.rollout_batch(
            state           = state,
            order_levels    = self.candidates,
            dis_lead_delta  = dis_lead_delta,
            dis_demand_mult = dis_demand_mult,
            capacity_ratio  = capacity_ratio,
            horizon         = self.horizon,
        )

        stockout_rates  = results["stockout_rate"]
        service_levels  = results["avg_service_lvl"]
        total_costs     = results["total_cost"]
        avg_inventories = results["avg_inventory"]

        best_idx   = int(np.argmin(total_costs))
        cost_mean  = float(np.mean(total_costs))
        cost_sens  = float(np.std(total_costs) / cost_mean) if cost_mean > 0 else 0.0
        mid_idx    = len(self.candidates) // 2

        return np.clip(np.array([
            float(np.min(stockout_rates)),
            float(np.max(service_levels)),
            float(stockout_rates[mid_idx]),
            np.log1p(float(np.min(total_costs))) / np.log1p(1000.0),
            np.log1p(float(avg_inventories[best_idx])) / np.log1p(500.0),
            cost_sens,
            float(np.mean(stockout_rates > 0.1)),
            float(self.candidates[best_idx]) / float(max(self.candidates)),
        ], dtype=np.float32), -5.0, 5.0)
