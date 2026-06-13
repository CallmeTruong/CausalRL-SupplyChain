"""
draw.py — all rendering. Reads VizState, writes to pygame surface.
"""
import math
import pygame
from viz import config as cfg

_ALPHA_SURF = None

def get_alpha_surf():
    global _ALPHA_SURF
    if _ALPHA_SURF is None:
        _ALPHA_SURF = pygame.Surface((cfg.W, cfg.H), pygame.SRCALPHA)
    _ALPHA_SURF.fill((0, 0, 0, 0))
    return _ALPHA_SURF

def r(surf, color, x, y, w, h, rd=8, border=None, bw=1):
    pygame.draw.rect(surf, color, (x, y, w, h), border_radius=rd)
    if border:
        pygame.draw.rect(surf, border, (x, y, w, h), bw, border_radius=rd)

def t(surf, s, font, color, x, y, align="left"):
    img = font.render(str(s), True, color)
    if align == "center": x -= img.get_width() // 2
    elif align == "right": x -= img.get_width()
    surf.blit(img, (x, y))

def _dashed(surf, col, x1, y1, x2, y2, dash=5, gap=4):
    dx, dy = x2-x1, y2-y1
    ln = math.hypot(dx, dy)
    if ln == 0: return
    for i in range(int(ln / (dash+gap))):
        t0 = i*(dash+gap)/ln
        t1 = min((i*(dash+gap)+dash)/ln, 1.0)
        pygame.draw.line(surf, col,
                         (int(x1+dx*t0), int(y1+dy*t0)),
                         (int(x1+dx*t1), int(y1+dy*t1)), 1)

def arrow(surf, color, x1, y1, x2, y2, alpha=200, dashed=False):
    tmp = get_alpha_surf()
    col = (*color, alpha)
    if dashed: _dashed(tmp, col, x1, y1, x2, y2)
    else: pygame.draw.line(tmp, col, (x1,y1), (x2,y2), 1)
    ang = math.atan2(y2-y1, x2-x1)
    for da in (-0.42, 0.42):
        pygame.draw.line(tmp, col, (x2,y2),
                         (int(x2-9*math.cos(ang-da)), int(y2-9*math.sin(ang-da))), 1)
    surf.blit(tmp, (0,0))

def flow_dots(surf, color, x1, y1, x2, y2, phase, active):
    if not active: return
    dist  = math.hypot(x2-x1, y2-y1)
    steps = max(1, int(dist/32))
    tmp   = get_alpha_surf()
    for i in range(steps):
        tt = ((i/steps) + (phase%32)/32) % 1.0
        px = int(x1+(x2-x1)*tt)
        py = int(y1+(y2-y1)*tt)
        pygame.draw.circle(tmp, (*color, 90), (px,py), 3)
    surf.blit(tmp, (0,0))

def _sparkline(surf, bx, by, bw, bh, data, color,
               ref=None, ref_col=None, data2=None, col2=None,
               ymin=None, ymax=None):
    if len(data) < 2: return
    pl,pr,pt,pb = 8,6,4,8
    cx,cy = bx+pl, by+pt
    cw,ch = bw-pl-pr, bh-pt-pb
    n = len(data)
    lo = ymin if ymin is not None else min(min(data), min(data2) if data2 else 9e9)
    hi = ymax if ymax is not None else max(max(data), max(data2) if data2 else -9e9, lo+1)

    def px(i): return int(cx+(i/(n-1))*cw)
    def py(v): return int(cy+ch-((v-lo)/(hi-lo))*ch)

    if ref is not None:
        ry = py(ref)
        _dashed(surf, (*(ref_col or cfg.C["amber"]), 130), cx, ry, cx+cw, ry, dash=3, gap=4)

    pts = [(px(i), py(v)) for i,v in enumerate(data)]
    poly = pts + [(px(n-1), cy+ch), (px(0), cy+ch)]
    tmp = get_alpha_surf()
    local = [(p[0]-bx, p[1]-by) for p in poly]
    pygame.draw.polygon(tmp, (*color, 35), local)
    surf.blit(tmp, (bx, by))

    if data2 and len(data2) >= 2:
        n2 = len(data2)
        def px2(i): return int(cx+(i/(n2-1))*cw)
        for i in range(n2-1):
            _dashed(surf, (*(col2 or cfg.C["blue"]), 150),
                    px2(i), py(data2[i]), px2(i+1), py(data2[i+1]), dash=4, gap=3)

    pygame.draw.lines(surf, color, False, pts, 2)
    lv = data[-1]
    dc = cfg.C["red"] if lv < 5 else cfg.C["amber"] if lv < 15 else color
    pygame.draw.circle(surf, dc, (px(n-1), py(lv)), 4)


_KPI_Y  = 65
_MAP_Y  = 135
_PAN_Y  = 395
_SVC_Y  = 515
_LOG_Y  = 610
_LEG_Y  = 695

def draw_header(surf, vs):
    t(surf, vs.item_id,  cfg.FONT_LG,      cfg.C["t1"], 290, 10)
    t(surf, vs.store_id, cfg.FONT_MONO_SM, cfg.C["t2"], 290, 32)
    t(surf, f"mean {vs.demand_mean:.1f} u/day   CV {vs.demand_cv:.2f}", cfg.FONT_MONO_SM, cfg.C["t3"], 360, 32)

def draw_kpis(surf, vs):
    kpis = [
        ("Day",         str(vs.day),                   None),
        ("Inventory",   str(vs.inv),                   _ic(vs.inv)),
        ("Service Lvl", _st(vs),                       _sc(vs)),
        ("Stockout",    str(vs.last_stockout),         cfg.C["red"] if vs.last_stockout else None),
        ("Total Stk",   str(vs.total_stockout),        cfg.C["amber"] if vs.total_stockout else None),
        ("Last Order",  f"{vs.last_order}u ", cfg.C["blue"] if vs.last_order else None),
        ("Total Cost",  f"${vs.cum_total_cost:,.0f}",  cfg.C["t1"]),
        ("Reward",      f"{vs.last_reward:+.2f}",      _rc(vs.last_reward)),
    ]
    n  = len(kpis)
    kw = (cfg.W - 20 - (n-1)*5) // n
    kh = 60
    x  = 10
    for label, value, color in kpis:
        r(surf, cfg.C["card"], x, _KPI_Y, kw, kh, border=cfg.C["border"])
        t(surf, label.upper(), cfg.FONT_SM,      cfg.C["t2"],          x+8, _KPI_Y+8)
        t(surf, value,         cfg.FONT_MONO_LG, color or cfg.C["t1"], x+8, _KPI_Y+28)
        x += kw+5

def _ic(inv):
    return cfg.C["red"] if inv < 5 else cfg.C["amber"] if inv < 15 else cfg.C["teal"]

def _st(vs):
    s = vs.avg_service_level
    return f"{s*100:.1f}%" if s is not None else "—"

def _sc(vs):
    s = vs.avg_service_level
    if s is None:  return cfg.C["t1"]
    if s >= 0.95:  return cfg.C["teal"]
    if s >= 0.80:  return cfg.C["amber"]
    return cfg.C["red"]

def _rc(rw):
    if rw >= 1.0:  return cfg.C["teal"]
    if rw >= 0.0:  return cfg.C["t1"]
    if rw >= -2.0: return cfg.C["amber"]
    return cfg.C["red"]

def draw_agent_node(surf, vs, pulse):
    n   = cfg.NODES["agent"]
    col = cfg.C["blue"]
    if pulse > 0:
        tmp = get_alpha_surf()
        pygame.draw.rect(tmp, (*col, int(pulse*65)),
                         (n["x"]-4, n["y"]-4, n["w"]+8, n["h"]+8), 3, border_radius=12)
        surf.blit(tmp, (0,0))
    r(surf, (238,244,255), n["x"], n["y"], n["w"], n["h"], border=col, bw=1)
    lbl = f"Agent  →  {vs.last_order}u" if vs.last_order > 0 else "Agent  —  hold"
    t(surf, lbl, cfg.FONT_MD, col, n["cx"], n["cy"]-7, align="center")

def draw_supplier_node(surf, vs):
    n  = cfg.NODES["sup"]
    dis_on = vs.dis_days > 0 and vs.dis_type in (1,2)
    acc = cfg.C["red"] if dis_on else cfg.C["border"]
    r(surf, cfg.C["card"], n["x"], n["y"], n["w"], n["h"], border=acc, bw=2 if dis_on else 1)
    t(surf, "Supplier", cfg.FONT_MD, cfg.C["t1"], n["cx"], n["cy"]-2, align="center")
    if dis_on:
        t(surf, f"cap {vs.dis_cap_ratio*100:.0f}%  +{vs.dis_lead_delta:.0f}d",
          cfg.FONT_MONO_SM, cfg.C["red"], n["cx"], n["cy"]+20, align="center")

def draw_warehouse_node(surf, vs):
    n   = cfg.NODES["dc"]
    acc = cfg.C["blue"] if vs.last_order > 0 else cfg.C["border"]
    r(surf, cfg.C["card"], n["x"], n["y"], n["w"], n["h"], border=acc, bw=2 if vs.last_order else 1)
    t(surf, "WAREHOUSE",     cfg.FONT_MONO_SM, cfg.C["t2"], n["cx"], n["y"]+10, align="center")
    t(surf, str(vs.inv),     cfg.FONT_MONO_LG, _ic(vs.inv), n["cx"], n["cy"]-20, align="center")
    t(surf, "units on hand", cfg.FONT_SM,      cfg.C["t2"], n["cx"], n["cy"]+6,  align="center")
    if vs.backlog > 0:
        t(surf, f"backlog {vs.backlog}u", cfg.FONT_MONO_SM, cfg.C["red"],
          n["cx"], n["cy"]+24, align="center")
    bx2, by2 = n["x"]+12, n["by"]-12
    bw2, bh2 = n["w"]-24, 5
    pct = min(1.0, vs.inv / max(vs.demand_mean*30, 60))
    r(surf, cfg.C["border"], bx2, by2, bw2, bh2, rd=3)
    if pct > 0:
        r(surf, _ic(vs.inv), bx2, by2, max(4, int(bw2*pct)), bh2, rd=3)

def draw_store_node(surf, vs):
    n = cfg.NODES["store"]
    r(surf, cfg.C["card"], n["x"], n["y"], n["w"], n["h"], border=cfg.C["border"])
    t(surf, vs.store_id, cfg.FONT_MD,      cfg.C["t1"], n["cx"], n["cy"]-14, align="center") # Tách text xa ra
    t(surf, vs.item_id,  cfg.FONT_MONO_SM, cfg.C["t2"], n["cx"], n["cy"]+6,  align="center")
    if vs.dis_days > 0 and vs.dis_type == 3:
        t(surf, f"surge x {vs.dis_dem_mult:.1f}", cfg.FONT_MONO_SM,
          cfg.C["red"], n["cx"], n["cy"]+24, align="center")
    elif vs.last_demand > 0:
        t(surf, f"demand {vs.last_demand}u", cfg.FONT_MONO_SM,
          cfg.C["amber"], n["cx"], n["cy"]+24, align="center")

def draw_pipeline_dots(surf, vs):
    n  = cfg.NODES
    x1, y1 = n["sup"]["rx"], n["sup"]["cy"]
    x2, y2 = n["dc"]["lx"],  n["dc"]["cy"]
    for order in vs.pipeline:
        dl   = order.eta - vs.day
        lead = max(vs.last_lead_time, 1)
        prog = min(0.92, max(0.08, 1 - dl/lead))
        px   = int(x1+(x2-x1)*prog)
        py   = int(y1+(y2-y1)*prog)
        pygame.draw.circle(surf, cfg.C["violet"], (px, py), 9)
        t(surf, str(order.qty), cfg.FONT_MONO_SM, (255,255,255), px, py-7, align="center")

def draw_agent_line(surf, tick):
    n  = cfg.NODES
    x1, y1 = n["agent"]["cx"], n["agent"]["by"]
    x2, y2 = n["dc"]["cx"],    n["dc"]["ty"]
    col    = (*cfg.C["blue"], 50)
    tmp    = get_alpha_surf()
    dx, dy = x2-x1, y2-y1
    ln     = max(1, math.hypot(dx,dy))
    offset = -(tick*0.3) % 9
    drawn  = 0
    while drawn < ln:
        t0 = max(0.0, min(1.0, (drawn+offset)/ln))
        t1 = max(0.0, min(1.0, (drawn+4+offset)/ln))
        pygame.draw.line(tmp, col,
                         (int(x1+dx*t0), int(y1+dy*t0)),
                         (int(x1+dx*t1), int(y1+dy*t1)), 1)
        drawn += 9
    surf.blit(tmp, (0,0))

_PH = 110   
_P1 = (10,   _PAN_Y, 330, _PH)
_P2 = (348,  _PAN_Y, 390, _PH)
_P3 = (746,  _PAN_Y, 344, _PH)
_P4 = (10,   _SVC_Y, 1080, 85)

def _panel(surf, bx, by, bw, bh, title):
    r(surf, cfg.C["card"], bx, by, bw, bh, border=cfg.C["border"])
    t(surf, title, cfg.FONT_SM, cfg.C["t2"], bx+10, by+8)

def draw_inventory_chart(surf, vs):
    bx, by, bw, bh = _P1
    _panel(surf, bx, by, bw, bh, "Inventory (60 days)")
    safety = vs.demand_mean * vs.last_lead_time
    _sparkline(surf, bx, by+18, bw, bh-22,
               data=vs.inv_hist, color=cfg.C["teal"],
               ref=safety, ref_col=cfg.C["amber"],
               data2=vs.fct_hist, col2=cfg.C["blue"], ymin=0)
    t(surf, "── inv",          cfg.FONT_MONO_SM, cfg.C["teal"],  bx+10,  by+bh-15)
    t(surf, "-- forecast",     cfg.FONT_MONO_SM, cfg.C["blue"],  bx+58,  by+bh-15)
    t(surf, f"safety {safety:.0f}u", cfg.FONT_MONO_SM, cfg.C["amber"], bx+140, by+bh-15)

def draw_cost_panel(surf, vs):
    bx, by, bw, bh = _P2
    _panel(surf, bx, by, bw, bh, "Cumulative Cost Breakdown")
    total = max(vs.cum_total_cost, 1.0)
    rows  = [
        ("Holding",  vs.cum_hold_cost, cfg.C["teal"]),
        ("Stockout", vs.cum_stk_cost,  cfg.C["red"]),
        ("Ordering", vs.cum_ord_cost + vs.cum_bc_cost, cfg.C["blue"]),
    ]
    bx2  = bx + 72
    bw2  = bw - 155
    vx   = bx + bw - 10
    for i, (label, val, color) in enumerate(rows):
        ry = by + 28 + i*26
        t(surf, label,          cfg.FONT_SM,      cfg.C["t2"], bx+10, ry)
        t(surf, f"${val:,.0f}", cfg.FONT_MONO_SM, color,       vx,    ry, align="right")
        r(surf, cfg.C["border"], bx2, ry+12, bw2, 6, rd=3)
        fill = max(4, int(bw2 * val/total))
        r(surf, color, bx2, ry+12, fill, 6, rd=3)
    t(surf, f"Total  ${vs.cum_total_cost:,.0f}",
      cfg.FONT_MONO_SM, cfg.C["t1"], bx+10, by+bh-10)

def draw_pipeline_panel(surf, vs):
    bx, by, bw, bh = _P3
    _panel(surf, bx, by, bw, bh, "In Transit")
    t(surf, f"{vs.pipeline_qty}u total",
      cfg.FONT_MONO_SM, cfg.C["violet"], bx+bw-10, by+8, align="right")
    if not vs.pipeline:
        t(surf, "No orders in transit", cfg.FONT_MONO_SM, cfg.C["t3"], bx+10, by+44)
        return
    orders = sorted(vs.pipeline, key=lambda o: o.eta)
    for i, order in enumerate(orders[:3]):
        dl  = order.eta - vs.day
        col = cfg.C["amber"] if dl <= 2 else cfg.C["violet"]
        ry  = by + 28 + i*26
        pygame.draw.circle(surf, col, (bx+15, ry+6), 5)
        t(surf, f"{order.qty}u", cfg.FONT_MONO_SM, cfg.C["t1"], bx+26, ry)
        t(surf, f"ETA {order.eta}  ({dl}d)", cfg.FONT_MONO_SM, cfg.C["t2"], bx+bw-10, ry, align="right")
        lead = max(vs.last_lead_time, 1)
        prog = min(1.0, max(0.0, 1 - dl/lead))
        r(surf, cfg.C["border"], bx+26, ry+16, bw-76, 3, rd=2)
        r(surf, col, bx+26, ry+16, max(3, int((bw-76)*prog)), 3, rd=2)
    if len(vs.pipeline) > 3:
        t(surf, f"+{len(vs.pipeline)-3} more", cfg.FONT_SM, cfg.C["t3"], bx+10, by+bh-18)

def draw_service_chart(surf, vs):
    bx, by, bw, bh = _P4
    _panel(surf, bx, by, bw, bh, "Service Level & Demand vs Forecast  (60 days)")
    if len(vs.svc_hist) < 2: return
    half = (bw-30)//2
    _sparkline(surf, bx+8, by+18, half, bh-22,
               data=vs.svc_hist, color=cfg.C["teal"],
               ref=95.0, ref_col=cfg.C["amber"], ymin=0.0, ymax=100.0)
    t(surf, "Service level %", cfg.FONT_SM, cfg.C["t2"],   bx+10,    by+bh-15)
    t(surf, "-- 95% target",   cfg.FONT_SM, cfg.C["amber"], bx+110,  by+bh-15)

    pygame.draw.line(surf, cfg.C["border"],
                     (bx+half+15, by+10), (bx+half+15, by+bh-10), 1)

    rx = bx+half+22
    _sparkline(surf, rx, by+18, half, bh-22,
               data=vs.dem_hist, color=cfg.C["amber"],
               data2=vs.fct_hist, col2=cfg.C["blue"], ymin=0)
    t(surf, "- actual demand", cfg.FONT_SM, cfg.C["amber"], rx,     by+bh-15)
    t(surf, "-- forecast",      cfg.FONT_SM, cfg.C["blue"],  rx+120, by+bh-15)


# ── Disruption banner ─────────────────────────────────────────────
def draw_disruption_banner(surf, vs, tick):
    if vs.dis_days <= 0: return
    names = {1: "PORT CLOSURE", 2: "SUPPLIER FAILURE", 3: "DEMAND SURGE"}
    
    # Đặt hộp thông báo nằm ở sát mép trên bên phải (ngay phía trên cụm nút bấm)
    bx = cfg.W - 360
    by = 6
    bw = 350
    bh = 24
    
    # Vẽ một hộp nền hồng nhạt sang trọng kèm viền đỏ mảnh mang tính cảnh báo cao
    r(surf, (254, 242, 242), bx, by, bw, bh, rd=5, border=(239, 68, 68), bw=1)
    
    dis_name = names.get(vs.dis_type, "DISRUPTION")
    msg = f"{dis_name} ({vs.dis_days}d left)"
    
    # Sử dụng FONT_MD giúp tiêu đề biến cố hiển thị TO và RÕ RÀNG hơn hẳn bản cũ
    t(surf, msg, cfg.FONT_MD, (192, 43, 43), bx + 8, by + 1)
    
    # Các chỉ số phụ được gom gọn gàng, tinh tế ở góc phải hộp thông báo
    info = f"+{vs.dis_lead_delta:.0f}d | cap {vs.dis_cap_ratio*100:.0f}%"
    t(surf, info, cfg.FONT_MONO_SM, (120, 113, 108), bx + bw - 8, by + 5, align="right")


_LCOL = {"ok": (13,144,104), "warn": (180,83,9), "bad": (192,43,43), "": (120,113,108)}

def draw_log(surf, logs):
    bx, by, bw, bh = 10, _LOG_Y, cfg.W-20, 80
    r(surf, cfg.C["card"], bx, by, bw, bh, border=cfg.C["border"])
    y = by+8
    for entry in logs[:4]:
        col = _LCOL.get(entry.get("level",""), cfg.C["t2"])
        t(surf, entry["text"], cfg.FONT_MONO_SM, col, bx+10, y)
        y += 18


def draw_legend(surf):
    items = [
        (cfg.C["blue"],   "Order"),
        (cfg.C["teal"],   "Receive"),
        (cfg.C["amber"],  "Demand"),
        (cfg.C["violet"], "In transit"),
        (cfg.C["red"],    "Stockout"),
    ]
    x = 10
    y = _LEG_Y
    for color, label in items:
        pygame.draw.circle(surf, color, (x+5, y+5), 4)
        t(surf, label, cfg.FONT_SM, cfg.C["t3"], x+14, y-10)
        x += len(label)*6 + 28
    hint = "Space=Play/Pause   R=Reset   D=Disruption   ←→=Item   ↑↓=Speed"
    t(surf, hint, cfg.FONT_SM, cfg.C["t3"], cfg.W-10, _LEG_Y-1, align="right")