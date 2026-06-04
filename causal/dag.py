from typing import Dict, List, Set


class CausalDAG:
    """Definition of the causal relationship of the supply chain.
    Used to determine which nodes the intervention do(X=x) reaches."""

    # DAG
    STRUCTURE: Dict[str, List[str]] = {
        # Exogenous — No parent
        "DisruptionEvent":   [],
        "WeatherNoise":      [],
        "DemandNoise":       [],
        "DemandForecast":    [],
        "OrderQuantity":     [],   # action of agent

        # Supply side
        "LeadTime":          ["DisruptionEvent", "WeatherNoise"],
        "CapacityRatio":     ["DisruptionEvent"],
        "ActualOrder":       ["OrderQuantity", "CapacityRatio"],
        "Received":          ["ActualOrder", "LeadTime"],

        # Demand side
        "Demand":            ["DemandForecast", "DemandNoise", "DisruptionEvent"],

        # Inventory dynamics
        "InventoryNext":     ["Received", "Demand"],
        "Stockout":          ["InventoryNext", "Demand"],
        "ServiceLevel":      ["Stockout", "Demand"],

        # Cost
        "HoldingCost":       ["InventoryNext"],
        "StockoutCost":      ["Stockout"],
        "OrderCost":         ["ActualOrder"],
        "TotalCost":         ["HoldingCost", "StockoutCost", "OrderCost"],
    }

    def __init__(self):
        self._children = self._build_children()

    def _build_children(self) -> Dict[str, List[str]]:
        children = {n: [] for n in self.STRUCTURE}
        for node, parents in self.STRUCTURE.items():
            for p in parents:
                children[p].append(node)
        return children

    def descendants(self, node: str) -> Set[str]:
        """All nodes are affected when this node changes."""
        result, queue = set(), list(self._children[node])
        while queue:
            n = queue.pop(0)
            if n not in result:
                result.add(n)
                queue.extend(self._children[n])
        return result

    def topological_order(self) -> List[str]:
        """calc parent first"""
        in_degree = {n: len(p) for n, p in self.STRUCTURE.items()}
        queue     = [n for n, d in in_degree.items() if d == 0]
        order     = []
        while queue:
            n = queue.pop(0)
            order.append(n)
            for child in self._children[n]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        assert len(order) == len(self.STRUCTURE)
        return order

    def summary(self):
        print("=== Causal DAG ===")
        for n in self.topological_order():
            parents = self.STRUCTURE[n] or ["—"]
            print(f"  {n:20s} ← {', '.join(parents)}")


DAG = CausalDAG().summary()