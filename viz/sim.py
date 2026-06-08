"""
sim.py — lightweight built-in sim that mirrors SupplyChainEnv logic.
If the real PPO model + env are importable, they are used instead.
"""
import math, random
from dataclasses import dataclass, field
from typing import Optional

# ── Try to load real env (optional) ──────────────────────────────
_REAL = False
_model = None
_env_cls = None
_item_df_map = {}
_item_keys_real = []
_cfg_real = {}

def _try_real():
    global _REAL, _model, _env_cls, _item_df_map, _item_keys_real, _cfg_real
    try:
        import yaml
        from stable_baselines3 import PPO
        from env.supply_chain_env import SupplyChainEnv
        from demand.lightgbm_trainer import load_m5_multi
        from demand.feature_engineering import create_features
        from viz import config as vcfg

        _cfg_real = yaml.safe_load(open("configs/config.yaml"))
        d = _cfg_real["demand"]
        store_ids = d.get("store_ids", [d.get("store_id", "CA_1")])
        df = load_m5_multi(d["sales_path"], d["calendar_path"],
                           d.get("n_items", 50), store_ids)
        for (sid, iid), g in df.groupby(["store_id", "item_id"]):
            key = f"{sid}__{iid}"
            _item_df_map[key] = create_features(g.copy()).reset_index(drop=True)
        _item_keys_real = sorted(_item_df_map.keys())
        _model   = PPO.load(vcfg.model_path)
        _env_cls = SupplyChainEnv
        _REAL    = True
        print(f"[sim] Real env — {len(_item_keys_real)} items")
    except Exception as e:
        print(f"[sim] Fallback sim ({e})")

# ── Built-in items ────────────────────────────────────────────────
_BUILTIN = [
    ("CA_1", "FOODS_3_090",    3.2, 1.05),
    ("CA_1", "HOBBIES_1_004",  1.5, 1.24),
    ("TX_1", "FOODS_1_001",    5.8, 0.90),
    ("CA_2", "HOUSEHOLD_1_01", 2.1, 1.10),
    ("TX_2", "FOODS_2_033",    4.4, 0.95),
]
_ORDER_LEVELS = [0, 4, 8, 12, 16, 20, 28, 41, 53]

# ── Shared state ──────────────────────────────────────────────────
@dataclass
class Order:
    qty: int
    eta: int   # arrival day

@dataclass
class VizState:
    item_key:    str   = "—"
    store_id:    str   = "—"
    item_id:     str   = "—"
    day:         int   = 0
    done:        bool  = False

    inv:         int   = 0
    backlog:     int   = 0

    last_demand:   int   = 0
    last_forecast: float = 0.0
    last_order:    int   = 0
    last_received: int   = 0
    last_stockout: int   = 0
    last_action:   int   = 0
    last_reward:   float = 0.0
    last_lead_time:int   = 7

    dis_type:      int   = 0
    dis_days:      int   = 0
    dis_lead_delta:float = 0.0
    dis_cap_ratio: float = 1.0
    dis_dem_mult:  float = 1.0

    svc_sum:        float = 0.0
    svc_days:       int   = 0
    total_stockout: int   = 0
    cum_hold_cost:  float = 0.0
    cum_stk_cost:   float = 0.0
    cum_ord_cost:   float = 0.0

    pipeline: list = field(default_factory=list)  # list of Order

    inv_hist: list = field(default_factory=list)
    svc_hist: list = field(default_factory=list)
    dem_hist: list = field(default_factory=list)
    fct_hist: list = field(default_factory=list)

    # built-in only
    _demand_mean: float = 1.1
    _demand_cv:   float = 1.0
    # real env only
    _env:  object = field(default=None, repr=False)
    _obs:  object = field(default=None, repr=False)

    @property
    def avg_service_level(self) -> Optional[float]:
        return self.svc_sum / self.svc_days if self.svc_days else None

    @property
    def pipeline_qty(self) -> int:
        return sum(o.qty for o in self.pipeline)

    @property
    def cum_total_cost(self) -> float:
        return self.cum_hold_cost + self.cum_stk_cost + self.cum_ord_cost

    @property
    def demand_mean(self) -> float:
        if _REAL and self._env:
            try: return self._env.engine.demand_generator.demand_mean
            except: pass
        return self._demand_mean

    @property
    def demand_cv(self) -> float:
        if _REAL and self._env:
            try: return self._env.engine.demand_generator.demand_cv
            except: pass
        return self._demand_cv


# ── Public API ────────────────────────────────────────────────────
def get_item_keys():
    if not _REAL:
        _try_real()
    if _REAL:
        return _item_keys_real
    return [f"{s}__{i}" for s, i, *_ in _BUILTIN]


def make_state(item_key: str) -> VizState:
    from viz import config as vcfg
    if not _item_df_map and not _REAL:
        _try_real()

    parts    = item_key.split("__")
    store_id = parts[0] if len(parts) > 1 else "—"
    item_id  = parts[1] if len(parts) > 1 else item_key

    vcfg.current_item_key = item_key
    vcfg.current_store_id = store_id
    vcfg.current_item_id  = item_id

    if _REAL:
        env = _env_cls(
            item_df_map={item_key: _item_df_map[item_key]},
            config=_cfg_real, seed=42)
        obs, _ = env.reset()
        vs = VizState(item_key=item_key, store_id=store_id, item_id=item_id,
                      inv=int(env.engine.inventory), _env=env, _obs=obs)
    else:
        row = next(((s,i,dm,cv) for s,i,dm,cv in _BUILTIN
                    if f"{s}__{i}" == item_key), ("—","—",1.1,1.0))
        init_inv = max(10, int(row[2] * 14))
        vs = VizState(item_key=item_key, store_id=store_id, item_id=item_id,
                      inv=init_inv, _demand_mean=row[2], _demand_cv=row[3])

    vs.inv_hist = [vs.inv]
    return vs


def sim_step(vs: VizState) -> dict:
    if vs.done:
        return {"text": "Episode finished.", "level": ""}
    if _REAL and vs._env is not None:
        return _real_step(vs)
    return _builtin_step(vs)


def trigger_disruption(vs: VizState):
    vs.dis_days = random.randint(5, 10)
    vs.dis_type = random.choice([2, 3])
    vs.dis_lead_delta = random.uniform(3, 8) if vs.dis_type == 2 else 0.0
    vs.dis_cap_ratio  = random.uniform(0.3, 0.6) if vs.dis_type == 2 else 1.0
    vs.dis_dem_mult   = random.uniform(2.0, 3.5) if vs.dis_type == 3 else 1.0
    if _REAL and vs._env:
        try:
            e = vs._env.engine.disruption_engine
            e.force_disruption()
        except Exception:
            pass


# ── Built-in step ─────────────────────────────────────────────────
def _builtin_agent(vs: VizState) -> int:
    pos    = vs.inv + vs.pipeline_qty - vs.backlog
    target = vs._demand_mean * 14 * (1.8 if vs.dis_days > 0 else 1.0)
    need   = target - pos
    if need <= 0: return 0
    for lvl in reversed(_ORDER_LEVELS):
        if lvl <= need: return lvl
    return _ORDER_LEVELS[-1]


def _builtin_step(vs: VizState) -> dict:
    dis = vs.dis_days > 0
    if dis: vs.dis_days -= 1

    dm = (vs.dis_dem_mult if vs.dis_type == 3 and dis
          else random.uniform(1.8, 2.5) if dis else 1.0)
    std     = vs._demand_mean * vs._demand_cv * 0.5
    demand  = max(0, round(vs._demand_mean * dm + random.gauss(0, std)))
    forecast = vs._demand_mean * dm

    received = sum(o.qty for o in vs.pipeline if o.eta == vs.day)
    vs.pipeline = [o for o in vs.pipeline if o.eta != vs.day]
    vs.last_received = received

    bl = min(vs.backlog, received)
    vs.backlog -= bl
    vs.inv     += received - bl

    sales    = min(demand, vs.inv)
    vs.inv  -= sales
    stockout = max(0, demand - sales)
    vs.backlog += stockout

    svc = 1.0 if demand == 0 else max(0.0, 1 - stockout / demand)
    vs.svc_sum        += svc
    vs.svc_days       += 1
    vs.total_stockout += stockout

    order = _builtin_agent(vs)
    vs.last_order    = order
    vs.last_action   = _ORDER_LEVELS.index(order) if order in _ORDER_LEVELS else 0
    vs.last_demand   = demand
    vs.last_forecast = forecast
    vs.last_stockout = stockout
    vs.last_reward   = svc * 2.0 - (order * 0.1 / max(vs._demand_mean, 1))

    HC, SP, BC, OCF, OCV = 0.05, 2.5, 0.20, 10.0, 0.10
    hc = vs.inv * HC
    sc = stockout * SP
    oc = (OCF if order > 0 else 0) + order * OCV + vs.backlog * BC
    vs.cum_hold_cost += hc
    vs.cum_stk_cost  += sc
    vs.cum_ord_cost  += oc

    if order > 0:
        lt = 7 + (2 if dis and vs.dis_type == 2 else 0)
        vs.pipeline.append(Order(qty=order, eta=vs.day + lt))

    vs.day += 1
    if vs.day > 365: vs.done = True

    def _push(lst, v):
        lst.append(v)
        if len(lst) > 60: lst.pop(0)

    _push(vs.inv_hist, vs.inv)
    _push(vs.svc_hist, svc * 100)
    _push(vs.dem_hist, demand)
    _push(vs.fct_hist, forecast)

    if stockout > 0:
        return {"text": f"Day {vs.day} — Stockout! {stockout}u short · Demand {demand} · Inv {vs.inv}", "level": "bad"}
    if dis:
        return {"text": f"Day {vs.day} — Disruption · Demand {demand} · Ordered {order}u · Inv {vs.inv}", "level": "warn"}
    if order > 0:
        return {"text": f"Day {vs.day} — Agent ordered {order}u · Inv {vs.inv} · Demand {demand}", "level": "ok"}
    return {"text": f"Day {vs.day} — Stable · Inv {vs.inv} · Demand {demand}", "level": ""}


# ── Real env step ─────────────────────────────────────────────────
def _real_step(vs: VizState) -> dict:
    env = vs._env
    action, _ = _model.predict(vs._obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(int(action))

    vs._obs = obs
    vs.done = terminated or truncated

    order    = int(info.get("actual_order", env.order_levels[action]))
    demand   = int(info.get("demand",    0))
    stockout = int(info.get("stockout",  0))
    received = int(info.get("received",  0))
    dis_type = int(info.get("dis_type",  0))
    dis_days = int(info.get("dis_days_remaining", 0))
    cost     = float(info.get("total_cost", 0.0))
    inv      = int(info.get("inventory", env.engine.inventory))

    hc = inv * 0.05
    sc = stockout * 2.5
    oc = max(0.0, cost - hc - sc)

    vs.day            = int(env._step_count)
    vs.inv            = inv
    vs.backlog        = int(info.get("backlog", 0))
    vs.last_demand    = demand
    vs.last_forecast  = float(info.get("demand_forecast", demand))
    vs.last_order     = order
    vs.last_received  = received
    vs.last_stockout  = stockout
    vs.last_action    = int(action)
    vs.last_reward    = float(reward)
    vs.last_lead_time = int(info.get("lead_time", 7))
    vs.dis_type       = dis_type
    vs.dis_days       = dis_days
    vs.dis_lead_delta = float(info.get("dis_lead_delta", 0.0))
    vs.dis_cap_ratio  = float(info.get("dis_capacity_ratio", 1.0))
    vs.dis_dem_mult   = float(info.get("dis_demand_mult", 1.0))

    vs.cum_hold_cost += hc
    vs.cum_stk_cost  += sc
    vs.cum_ord_cost  += oc
    vs.total_stockout += stockout

    svc = 1.0 if demand == 0 else max(0.0, 1 - stockout / demand)
    vs.svc_sum  += svc
    vs.svc_days += 1

    # Pipeline snapshot
    try:
        vs.pipeline = [Order(qty=o["quantity"], eta=o["arrival_day"])
                       for o in env.engine.pipeline.orders]
    except Exception:
        vs.pipeline = []

    def _push(lst, v):
        lst.append(v)
        if len(lst) > 60: lst.pop(0)
    _push(vs.inv_hist, vs.inv)
    _push(vs.svc_hist, svc * 100)
    _push(vs.dem_hist, demand)
    _push(vs.fct_hist, vs.last_forecast)

    if stockout > 0:
        return {"text": f"Day {vs.day} — Stockout! {stockout}u short · Demand {demand} · Inv {vs.inv}", "level": "bad"}
    if dis_type > 0:
        names = {1:"Port closure", 2:"Supplier failure", 3:"Demand surge"}
        return {"text": f"Day {vs.day} — {names.get(dis_type,'Disruption')} · Demand {demand} · Ordered {order}u", "level": "warn"}
    if order > 0:
        return {"text": f"Day {vs.day} — Agent ordered {order}u · Inv {vs.inv} · Demand {demand}", "level": "ok"}
    return {"text": f"Day {vs.day} — Stable · Inv {vs.inv} · Demand {demand}", "level": ""}