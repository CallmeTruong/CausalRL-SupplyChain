import numpy as np
from simulator.shipment_pipeline import ShipmentPipeline
from simulator.disruption_engine import DisruptionEngine


class SupplyChainEngine:

    def __init__(
        self,
        demand_generator,
        dates,
        initial_inventory    = 500,
        base_lead_time       = 7,
        max_supplier_capacity= 1000,
        holding_cost         = 0.5,
        stockout_penalty     = 10.0,
        order_cost_fixed     = 2.0,
        order_cost_variable  = 0.1,
        disruption_engine    = None,
        seed                 = None,
    ):
        self.demand_generator       = demand_generator
        self.dates                  = dates
        self.initial_inventory      = initial_inventory
        self.base_lead_time         = base_lead_time
        self.max_supplier_capacity  = max_supplier_capacity
        self.holding_cost           = holding_cost
        self.stockout_penalty       = stockout_penalty
        self.order_cost_fixed       = order_cost_fixed
        self.order_cost_variable    = order_cost_variable
        self.disruption_engine      = disruption_engine or DisruptionEngine(seed=seed)
        self.reset()

    def reset(self):
        self.inventory   = self.initial_inventory
        self.backlog     = 0
        self.current_day = 0
        self.pipeline    = ShipmentPipeline()
        self.disruption_engine.reset()

    def step(self, order_quantity: int) -> dict:
        dis = self.disruption_engine.step()

        lead_time = int(self.base_lead_time + dis.lead_time_delta)
        capacity  = int(self.max_supplier_capacity * dis.capacity_ratio)

        # received
        received        = self.pipeline.receive(self.current_day)
        self.inventory += received

        # Demand from LightGBM
        date   = self.dates[self.current_day]
        demand = int(self.demand_generator.sample(date) * dis.demand_mult)

        self.demand_generator.record(date, demand)

        sales           = min(demand, self.inventory)
        self.inventory -= sales
        stockout        = demand - sales
        self.backlog    = max(0, self.backlog + stockout)

        # Order
        actual_order = min(order_quantity, capacity)
        if actual_order > 0:
            self.pipeline.add_order(
                quantity    = actual_order,
                arrival_day = self.current_day + lead_time,
            )

        # Cost
        holding_cost  = self.inventory * self.holding_cost
        stockout_cost = stockout       * self.stockout_penalty
        order_cost    = (self.order_cost_fixed + self.order_cost_variable * actual_order
                        ) if actual_order > 0 else 0.0
        total_cost    = holding_cost + stockout_cost + order_cost

        service_level = 1.0 if demand == 0 else sales / demand

        # Forecast today
        demand_forecast = self.demand_generator.forecast(date)

        self.current_day += 1
        done = self.current_day >= len(self.dates)

        return {
            "demand":             demand,
            "demand_forecast":    demand_forecast,
            "inventory":          self.inventory,
            "backlog":            self.backlog,
            "received":           received,
            "stockout":           stockout,
            "actual_order":       actual_order,
            "pipeline_qty":       self.pipeline.total_pipeline_quantity(),
            "lead_time":          lead_time,
            "capacity":           capacity,
            "dis_type":           dis.dtype,
            "dis_days_remaining": dis.days_remaining,
            "dis_lead_delta":     dis.lead_time_delta,
            "dis_capacity_ratio": dis.capacity_ratio,
            "dis_demand_mult":    dis.demand_mult,
            "service_level":      service_level,
            "total_cost":         total_cost,
            "done":               done,
        }