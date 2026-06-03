class ShipmentPipeline:

    def __init__(self):
        self.orders = []

    def add_order(
        self,
        quantity: int,
        arrival_day: int
    ):

        self.orders.append(
            {
                "quantity": quantity,
                "arrival_day": arrival_day
            }
        )

    def receive(
        self,
        current_day: int
    ):

        arrived_quantity = 0

        remaining_orders = []

        for order in self.orders:

            if order["arrival_day"] <= current_day:

                arrived_quantity += order["quantity"]

            else:

                remaining_orders.append(order)

        self.orders = remaining_orders

        return arrived_quantity

    def total_pipeline_quantity(self):

        return sum(
            order["quantity"]
            for order in self.orders
        )