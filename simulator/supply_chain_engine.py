from simulator.shipment_pipeline import ShipmentPipeline


class SupplyChainEngine:

    def __init__(
        self,
        demand_series,
        initial_inventory=500,
        lead_time=7,
        holding_cost=0.5,
        stockout_penalty=10,
        order_cost=2
    ):

        self.demand_series = demand_series

        self.inventory = initial_inventory

        self.backlog = 0

        self.current_day = 0

        self.lead_time = lead_time

        self.pipeline = ShipmentPipeline()

        self.holding_cost = holding_cost

        self.stockout_penalty = stockout_penalty

        self.order_cost = order_cost

    def place_order(
        self,
        quantity
    ):

        arrival_day = (
            self.current_day
            + self.lead_time
        )

        self.pipeline.add_order(
            quantity=quantity,
            arrival_day=arrival_day
        )

    def step(
        self,
        order_quantity
    ):

        demand = int(
            self.demand_series[
                self.current_day
            ]
        )

        self.place_order(
            order_quantity
        )

        received = self.pipeline.receive(
            self.current_day
        )

        self.inventory += received

        sales = min(
            demand,
            self.inventory
        )

        self.inventory -= sales

        stockout = max(
            0,
            demand - sales
        )

        self.backlog += stockout

        holding_cost = (
            self.inventory
            * self.holding_cost
        )

        stockout_cost = (
            stockout
            * self.stockout_penalty
        )

        ordering_cost = (
            self.order_cost
            if order_quantity > 0
            else 0
        )

        total_cost = (
            holding_cost
            + stockout_cost
            + ordering_cost
        )

        self.current_day += 1

        done = (
            self.current_day
            >= len(self.demand_series)
        )

        return {

            "day":
                self.current_day,

            "demand":
                demand,

            "inventory":
                self.inventory,

            "received":
                received,

            "sales":
                sales,

            "stockout":
                stockout,

            "pipeline":
                self.pipeline.total_pipeline_quantity(),

            "cost":
                total_cost,

            "done":
                done
        }