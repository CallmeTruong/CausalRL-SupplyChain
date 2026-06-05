import numpy as np
from dataclasses import dataclass
from causal.dag import DAG


@dataclass
class SCMState:
    """Noise terms"""
    inventory:        float
    backlog:          float
    demand_forecast:  float
    noise_lead_time:  float
    noise_demand:     float


class SCM:


    def __init__(
        self,
        base_lead_time    = 7,
        max_capacity      = 1000,
        holding_cost      = 0.5,
        stockout_penalty  = 10.0,
    ):
        self.base_lead_time   = base_lead_time
        self.max_capacity     = max_capacity
        self.holding_cost     = holding_cost
        self.stockout_penalty = stockout_penalty
        self.order_descendants = DAG.descendants("OrderQuantity")

    # ------------------------------------------------------------------
    # Structural equations
    # ------------------------------------------------------------------

    def _lead_time(self, dis_delta, noise):
        # LeadTime ← DisruptionEvent, WeatherNoise
        return max(1.0, self.base_lead_time + dis_delta + noise)

    def _actual_order(self, order, capacity_ratio):
        # ActualOrder ← OrderQuantity, CapacityRatio
        return min(order, self.max_capacity * capacity_ratio)

    def _demand(self, forecast, dis_mult, noise):
        # Demand ← DemandForecast, DemandNoise, DisruptionEvent
        return max(0.0, forecast * dis_mult + noise)

    def _received(self, actual_order, step, lead_time):
        # Received ← ActualOrder, LeadTime
        return actual_order if step == int(lead_time) else 0.0

    def _inventory_next(self, inventory, received, demand):
        # InventoryNext ← Received, Demand
        return max(0.0, inventory + received - demand)

    def _stockout(self, inventory, received, demand):
        # Stockout ← InventoryNext, Demand
        return max(0.0, demand - (inventory + received))

    def _total_cost(self, inventory, stockout, actual_order):
        # TotalCost ← HoldingCost, StockoutCost, OrderCost
        return (inventory  * self.holding_cost
              + stockout   * self.stockout_penalty
              + (2.0 if actual_order > 0 else 0.0))

    # ------------------------------------------------------------------
    # step 1: Abduction — infer noise from real observation
    # ------------------------------------------------------------------

    def abduct(
        self,
        observed_inventory:   float,
        observed_backlog:     float,
        observed_lead_time:   float,
        observed_demand:      float,
        demand_forecast:      float,
        dis_lead_delta:       float,
    ) -> SCMState:
        """
        Invert structural equations to solve noise.
        noise_lt = lead_time - base - dis_delta
        noise_d  = demand    - forecast
        """
        noise_lt = observed_lead_time - self.base_lead_time - dis_lead_delta
        noise_d  = observed_demand - demand_forecast

        return SCMState(
            inventory       = observed_inventory,
            backlog         = observed_backlog,
            demand_forecast = demand_forecast,
            noise_lead_time = float(np.clip(noise_lt, -3.0, 3.0)),
            noise_demand    = float(np.clip(noise_d, -30.0, 30.0)),
        )

    # ------------------------------------------------------------------
    # step 2+3: Intervention + Prediction
    # Nodes not in ORDER_DESCENDANTS → calc with noise fixed
    # Nodes in ORDER_DESCENDANTS       → calc with new order
    # ------------------------------------------------------------------

    def rollout(self, state, order_levels, dis_lead_delta,
                    dis_demand_mult, capacity_ratio, horizon=7):

        n = len(order_levels)

        inventory  = np.full(n, state.inventory)
        backlog    = np.full(n, state.backlog)
        # Pipeline đơn giản: mỗi candidate một giá trị in_transit
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