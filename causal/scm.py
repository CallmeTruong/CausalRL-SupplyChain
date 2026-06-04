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


# Precompute
ORDER_DESCENDANTS = DAG.descendants("OrderQuantity")


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

    def rollout(
        self,
        state:          SCMState,
        order_quantity: int,
        dis_lead_delta: float,
        dis_demand_mult:float,
        capacity_ratio: float,
        horizon:        int = 14,
    ) -> dict:
        inventory  = state.inventory
        backlog    = state.backlog
        in_transit = 0.0

        stockouts, inventories, costs, service_lvls = [], [], [], []

        for step in range(horizon):

            # if not descendants → keep noise from abduction (same world)
            lead_time = self._lead_time(dis_lead_delta, state.noise_lead_time)
            demand    = self._demand(state.demand_forecast, dis_demand_mult,
                                     state.noise_demand)

            # descendants → recalculate do(Order = order_quantity)
            actual_order = self._actual_order(order_quantity, capacity_ratio)
            received     = self._received(in_transit, step, lead_time)
            if step == 0:
                in_transit = actual_order

            stockout      = self._stockout(inventory, received, demand)
            inventory     = self._inventory_next(inventory, received, demand)
            backlog       = max(0.0, backlog + stockout)
            cost          = self._total_cost(inventory, stockout, actual_order)
            service_level = 1.0 if demand == 0 else max(0.0, 1.0 - stockout / demand)

            stockouts.append(stockout)
            inventories.append(inventory)
            costs.append(cost)
            service_lvls.append(service_level)

        return {
            "stockout_rate":   float(np.mean([s > 0 for s in stockouts])),
            "avg_inventory":   float(np.mean(inventories)),
            "total_cost":      float(np.sum(costs)),
            "avg_service_lvl": float(np.mean(service_lvls)),
        }