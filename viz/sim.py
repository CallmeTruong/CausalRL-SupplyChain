"""
sim.py — Real-data supply chain simulation engine.
Uses the trained LightGBM demand model and M5 historical data to drive
the inventory simulation. No synthetic / simulated demand is used.
"""
import math, random, os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from env.supply_chain_env import SupplyChainEnv
from demand.lightgbm_trainer import load_m5_multi
from demand.feature_engineering import create_features

__all__ = ["get_item_keys", "make_state", "sim_step",
            "trigger_disruption", "Order", "VizState"]


# ── Config ────────────────────────────────────────────────────────────────────
try:
    _CFG = yaml.safe_load(open("configs/config.yaml"))
except Exception:
    _CFG = {"simulation": {}, "disruption": {}, "demand": {}}

_SIM      = _CFG.get("simulation", {})
_DIS_CFG  = _CFG.get("disruption", {})
_DEM_CFG  = _CFG.get("demand", {})

_SIM_DEFAULTS = {
    "holding_cost":        0.05,
    "stockout_penalty":    2.5,
    "backlog_cost":        0.20,
    "order_cost_fixed":   10.0,
    "order_cost_variable":  1.0,
    "base_lead_time":        7,
    "episode_length":      365,
}


def _sim_val(key, default):
    return _SIM.get(key, _SIM_DEFAULTS.get(key, default))


# ── Load real M5 data ──────────────────────────────────────────────────────────
_item_df_map: dict = {}
_item_keys: list[str] = []
_env_cls = SupplyChainEnv
_cfg_real: dict = {}


def _load_m5_data():
    """Load M5 sales history for configured stores/items."""
    global _item_df_map, _item_keys, _cfg_real

    d = _DEM_CFG
    sales_path    = d.get("sales_path",    "data/raw/sales_train_evaluation.csv")
    calendar_path = d.get("calendar_path", "data/raw/calendar.csv")
    store_ids     = d.get("store_ids", ["CA_1"])
    n_items       = d.get("n_items", 50)

    if not os.path.exists(sales_path):
        print(f"[sim] Data file not found: {sales_path}")
        return

    try:
        df = load_m5_multi(sales_path, calendar_path, n_items, store_ids)
        for (store_id, item_id), g in df.groupby(["store_id", "item_id"]):
            key = f"{store_id}__{item_id}"
            feat = create_features(g.copy()).reset_index(drop=True)
            _item_df_map[key] = feat
        _item_keys = sorted(_item_df_map.keys())
        _cfg_real = _CFG
        print(f"[sim] Loaded M5 data — {len(_item_keys)} item-store pairs")
    except Exception as e:
        print(f"[sim] Failed to load M5 data: {e}")


_load_m5_data()


# ── Shared data structures ───────────────────────────────────────────────────────
@dataclass
class Order:
    qty: int      # order quantity
    eta: int       # estimated arrival day


@dataclass
class VizState:
    item_key:       str = "—"
    store_id:       str = "—"
    item_id:        str = "—"

    day:            int = 0
    done:           bool = False

    inv:            int = 0
    backlog:        int = 0

    last_demand:    int = 0
    last_forecast: float = 0.0
    last_order:     int = 0
    last_received:  int = 0
    last_stockout:  int = 0
    last_action:    int = 0
    last_reward:  float = 0.0
    last_lead_time: int = 7

    dis_type:       int = 0
    dis_days:       int = 0
    dis_lead_delta: float = 0.0
    dis_cap_ratio:  float = 1.0
    dis_dem_mult:  float = 1.0

    svc_sum:        float = 0.0
    svc_days:       int = 0
    total_stockout: int = 0
    cum_hold_cost:  float = 0.0
    cum_stk_cost:   float = 0.0
    cum_ord_cost:   float = 0.0
    cum_bc_cost:    float = 0.0

    pipeline: list = field(default_factory=list)  # list of Order

    inv_hist: list = field(default_factory=list)
    svc_hist: list = field(default_factory=list)
    dem_hist: list = field(default_factory=list)
    fct_hist: list = field(default_factory=list)

    # real env reference
    _env: object = field(default=None, repr=False)
    _obs: object = field(default=None, repr=False)

    @property
    def avg_service_level(self) -> Optional[float]:
        return self.svc_sum / self.svc_days if self.svc_days else None

    @property
    def pipeline_qty(self) -> int:
        return sum(o.qty for o in self.pipeline)

    @property
    def cum_total_cost(self) -> float:
        return (self.cum_hold_cost + self.cum_stk_cost
                + self.cum_ord_cost + self.cum_bc_cost)

    @property
    def demand_mean(self) -> float:
        if self._env is not None:
            try:
                return self._env.engine.demand_generator.demand_mean
            except Exception:
                pass
        return 0.0

    @property
    def demand_cv(self) -> float:
        if self._env is not None:
            try:
                return self._env.engine.demand_generator.demand_cv
            except Exception:
                pass
        return 0.0


# ── Public API ────────────────────────────────────────────────────────────────
def get_item_keys() -> list[str]:
    return _item_keys


def make_state(item_key: str) -> VizState:
    """Create a fresh VizState for the given item-store key."""
    if not _item_keys:
        print("[sim] No item-store data loaded. Check data files.")
        return VizState(item_key=item_key, store_id="—", item_id="—")

    parts    = item_key.split("__")
    store_id = parts[0] if len(parts) > 1 else "—"
    item_id  = parts[1] if len(parts) > 1 else item_key

    from viz import config as vcfg
    vcfg.current_item_key = item_key
    vcfg.current_store_id = store_id
    vcfg.current_item_id  = item_id

    env = _env_cls(
        item_df_map={item_key: _item_df_map[item_key]},
        config=_cfg_real,
        seed=42,
    )
    obs, _ = env.reset()

    vs = VizState(
        item_key=item_key,
        store_id=store_id,
        item_id=item_id,
        inv=int(env.engine.inventory),
        _env=env,
        _obs=obs,
    )
    vs.inv_hist = [vs.inv]
    return vs


def sim_step(vs: VizState) -> dict:
    """Advance the simulation by one day. Returns a status dict."""
    if vs.done:
        return {"text": "Episode finished.", "level": ""}
    return _real_step(vs)


def trigger_disruption(vs: VizState):
    """Force a disruption event on the current state."""
    vs.dis_days = random.randint(5, 10)
    vs.dis_type = random.choice([2, 3])
    vs.dis_lead_delta = random.uniform(3, 8) if vs.dis_type == 2 else 0.0
    vs.dis_cap_ratio  = random.uniform(0.3, 0.6) if vs.dis_type == 2 else 1.0
    vs.dis_dem_mult   = random.uniform(2.0, 3.5) if vs.dis_type == 3 else 1.0

    if vs._env is not None:
        try:
            vs._env.engine.disruption_engine.force_disruption()
        except Exception:
            pass


# ── Simulation step ────────────────────────────────────────────────────────────
def _real_step(vs: VizState) -> dict:
    env    = vs._env
    engine = env.engine
    gen    = engine.demand_generator

    dis    = engine.disruption_engine.step()
    lead_time  = int(_sim_val("base_lead_time", 7) + dis.lead_time_delta)
    capacity   = int(_sim_val("max_supplier_capacity", 1000) * dis.capacity_ratio)
    today      = engine.current_day

    received = engine.pipeline.receive(today)

    backlog_fulfilled = min(vs.backlog, received)
    vs.backlog -= backlog_fulfilled
    vs.inv     += received - backlog_fulfilled

    date   = engine.dates[today]
    demand = int(gen.sample(date) * dis.demand_mult)
    gen.record(date, demand)

    sales    = min(demand, vs.inv)
    vs.inv  -= sales
    stockout = max(0, demand - sales)
    vs.backlog += stockout

    svc = 1.0 if demand == 0 else sales / demand
    vs.svc_sum  += svc
    vs.svc_days += 1
    vs.total_stockout += stockout

    forecast  = float(gen.forecast(date))
    order_qty = max(0, min(int(forecast * 1.2), capacity))

    HC  = _sim_val("holding_cost", 0.05)
    SP  = _sim_val("stockout_penalty", 2.5)
    BC  = _sim_val("backlog_cost", 0.20)
    OCF = _sim_val("order_cost_fixed", 10.0)
    OCV = _sim_val("order_cost_variable", 1.0)

    hc = vs.inv * HC
    sc = stockout * SP
    bc = vs.backlog * BC
    oc = (OCF if order_qty > 0 else 0) + order_qty * OCV

    vs.cum_hold_cost += hc
    vs.cum_stk_cost  += sc
    vs.cum_ord_cost  += oc
    vs.cum_bc_cost   += bc

    if order_qty > 0:
        engine.pipeline.add_order(quantity=order_qty, arrival_day=today + lead_time)

    vs.pipeline = [
        Order(qty=o["quantity"], eta=o["arrival_day"])
        for o in engine.pipeline.orders
    ]

    engine.current_day += 1
    if engine.current_day >= len(engine.dates):
        vs.done = True

    def _push(lst, v):
        lst.append(v)
        if len(lst) > 60:
            lst.pop(0)

    _push(vs.inv_hist, vs.inv)
    _push(vs.svc_hist, svc * 100)
    _push(vs.dem_hist, demand)
    _push(vs.fct_hist, forecast)

    vs.day             = engine.current_day
    vs.last_demand     = demand
    vs.last_forecast   = forecast
    vs.last_order     = order_qty
    vs.last_received  = received
    vs.last_stockout  = stockout
    vs.last_action    = order_qty
    vs.last_reward    = svc * 2.0
    vs.last_lead_time = lead_time
    vs.dis_type       = dis.dtype
    vs.dis_days       = dis.days_remaining
    vs.dis_lead_delta = dis.lead_time_delta
    vs.dis_cap_ratio  = dis.capacity_ratio
    vs.dis_dem_mult   = dis.demand_mult

    if stockout > 0:
        return {
            "text": (f"Day {vs.day} — Stockout! {stockout}u short "
                     f"| Demand {demand} | Inv {vs.inv}"),
            "level": "bad",
        }
    if dis.dtype > 0:
        names = {1: "Port closure", 2: "Supplier failure", 3: "Demand surge"}
        return {
            "text": (f"Day {vs.day} — {names.get(dis.dtype, 'Disruption')} "
                     f"| Demand {demand} | Ordered {order_qty}u"),
            "level": "warn",
        }
    if order_qty > 0:
        return {
            "text": (f"Day {vs.day} — Ordered {order_qty}u "
                     f"| Inv {vs.inv} | Demand {demand}"),
            "level": "ok",
        }
    return {
        "text": f"Day {vs.day} — Stable | Inv {vs.inv} | Demand {demand}",
        "level": "",
    }
