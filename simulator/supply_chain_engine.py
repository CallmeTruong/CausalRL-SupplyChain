import numpy as np

from simulator.shipment_pipeline import ShipmentPipeline


class SupplyChainEngine:

    def __init__(
        self,
        initial_inventory=500,
        lead_time=7
    ):

        self.inventory = initial_inventory

        self.backlog = 0

        self.current_day = 0

        self.lead_time = lead_time

        self.pipeline = ShipmentPipeline()

    def place_order(self, quantity):

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
        demand,
        order_quantity
    ):

        self.place_order(order_quantity)

        received = self.pipeline.receive(
            self.current_day
        )

        self.inventory += received

        effective_inventory = (
            self.inventory
        )

        sales = min(
            demand,
            effective_inventory
        )

        self.inventory -= sales

        stockout = max(
            0,
            demand - sales
        )

        self.backlog += stockout

        self.current_day += 1

        return {
            "day": self.current_day,
            "inventory": self.inventory,
            "received": received,
            "sales": sales,
            "stockout": stockout,
            "pipeline": self.pipeline.get_pipeline_quantity()
        }