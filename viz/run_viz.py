import sys
import math
import pygame
from viz import config as cfg
from viz.sim import get_item_keys, make_state, sim_step, trigger_disruption
from viz import draw


# ── Controls layout ───────────────────────────────────────────────

_CTRL_Y  = 34
_CTRL_H  = 28
_CTRL_Y2 = 64

def _btn(surf, label, x, y, w=72, h=_CTRL_H,
         bg=None, fg=None, border=None):
    bg     = bg     or cfg.C["card"]
    fg     = fg     or cfg.C["t1"]
    border = border or cfg.C["border"]
    pygame.draw.rect(surf, bg,     (x, y, w, h), border_radius=7)
    pygame.draw.rect(surf, border, (x, y, w, h), 1, border_radius=7)
    img = cfg.FONT_SM.render(label, True, fg)
    surf.blit(img, (x + (w - img.get_width())//2, y + (h - img.get_height())//2))
    return pygame.Rect(x, y, w, h)


def draw_controls(surf, item_keys, item_idx, playing, speed):
    btns = {}

    # Item selector — left
    sel_w = 260
    sel_x = 12
    pygame.draw.rect(surf, cfg.C["card"], (sel_x, _CTRL_Y, sel_w, _CTRL_H), border_radius=7)
    pygame.draw.rect(surf, cfg.C["border"], (sel_x, _CTRL_Y, sel_w, _CTRL_H), 1, border_radius=7)

    btns["PREV"] = _btn(surf, "‹", sel_x+3, _CTRL_Y+3, w=22, h=22, bg=cfg.C["bg"])
    btns["NEXT"] = _btn(surf, "›", sel_x+sel_w-25, _CTRL_Y+3, w=22, h=22, bg=cfg.C["bg"])

    lbl = item_keys[item_idx] if item_keys else "—"
    if len(lbl) > 28: lbl = "…" + lbl[-26:]
    img = cfg.FONT_MONO_SM.render(lbl, True, cfg.C["t1"])
    surf.blit(img, (sel_x + sel_w//2 - img.get_width()//2, _CTRL_Y + 7))

    # Play / Pause
    x0 = cfg.W - 420
    play_lbl = "PAUSE" if playing else "PLAY"
    play_bg  = cfg.C["t1"] if not playing else (230,230,230)
    play_fg  = (255,255,255) if not playing else cfg.C["t1"]
    btns[play_lbl] = _btn(surf, play_lbl, x0, _CTRL_Y, w=76,
                          bg=play_bg, fg=play_fg, border=cfg.C["t1"])

    # Reset
    btns["RESET"] = _btn(surf, "RESET", x0+82, _CTRL_Y, w=72)

    # Disruption
    btns["DISRUPTION"] = _btn(surf, "DISRUPTION", x0+160, _CTRL_Y, w=100,
                               fg=cfg.C["red"], border=(210,170,170))

    # Speed row - Move to Top Bar
    draw.t(surf, f"Speed {speed}×", cfg.FONT_SM, cfg.C["t2"], x0+270, _CTRL_Y+6)
    btns["SPEED-"] = _btn(surf, "−", x0+335, _CTRL_Y, w=26, h=_CTRL_H, bg=cfg.C["bg"])
    btns["SPEED+"] = _btn(surf, "+", x0+365, _CTRL_Y, w=26, h=_CTRL_H, bg=cfg.C["bg"])

    return btns


# ── Main ─────────────────────────────────────────────────────────

def main():
    pygame.init()
    cfg.load_fonts()
    cfg.init_nodes()

    screen = pygame.display.set_mode((cfg.W, cfg.H))
    pygame.display.set_caption("Supply Chain RL — Live Inference")
    clock  = pygame.time.Clock()

    item_keys = get_item_keys()
    if not item_keys:
        print("[run_viz] No items found — check config or data path.")
        pygame.quit(); sys.exit(1)

    item_idx = 0
    vs       = make_state(item_keys[item_idx])
    logs     = [{"text": f"Loaded {vs.item_key} — inv {vs.inv}", "level": "ok"}]

    playing    = False
    speed      = 2
    tick       = 0
    step_timer = 0
    dot_phase  = 0.0
    ag_pulse   = 0.0

    DELAYS = {1:1400, 2:900, 3:550, 4:300, 5:120}

    def load_item(idx):
        nonlocal vs, logs, playing
        vs   = make_state(item_keys[idx])
        logs = [{"text": f"Loaded {vs.item_key} — inv {vs.inv}", "level": "ok"}]
        playing = False

    while True:
        dt = clock.tick(60)
        tick      += 1
        dot_phase  = (dot_phase + 0.45) % 60
        ag_pulse   = max(0.0, ag_pulse - 0.03)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if event.type == pygame.KEYDOWN:
                if   event.key == pygame.K_SPACE: playing = not playing
                elif event.key == pygame.K_r:     load_item(item_idx)
                elif event.key == pygame.K_d:
                    trigger_disruption(vs)
                    logs.insert(0, {"text": f"Day {vs.day} — DISRUPTION triggered!", "level": "bad"})
                elif event.key == pygame.K_UP:    speed = min(5, speed+1)
                elif event.key == pygame.K_DOWN:  speed = max(1, speed-1)
                elif event.key == pygame.K_RIGHT:
                    item_idx = (item_idx+1) % len(item_keys)
                    load_item(item_idx)
                elif event.key == pygame.K_LEFT:
                    item_idx = (item_idx-1) % len(item_keys)
                    load_item(item_idx)

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                btns = draw_controls(screen, item_keys, item_idx, playing, speed)

                play_key = "PAUSE" if playing else "PLAY"
                if   play_key in btns and btns[play_key].collidepoint(mx, my): playing = not playing
                elif btns.get("RESET",pygame.Rect(0,0,0,0)).collidepoint(mx,my):     load_item(item_idx)
                elif btns.get("DISRUPTION",pygame.Rect(0,0,0,0)).collidepoint(mx,my):
                    trigger_disruption(vs)
                    logs.insert(0, {"text": f"Day {vs.day} — DISRUPTION triggered!", "level": "bad"})
                elif btns.get("SPEED+",pygame.Rect(0,0,0,0)).collidepoint(mx,my): speed = min(5,speed+1)
                elif btns.get("SPEED-",pygame.Rect(0,0,0,0)).collidepoint(mx,my): speed = max(1,speed-1)
                elif btns.get("PREV",  pygame.Rect(0,0,0,0)).collidepoint(mx,my):
                    item_idx = (item_idx-1) % len(item_keys); load_item(item_idx)
                elif btns.get("NEXT",  pygame.Rect(0,0,0,0)).collidepoint(mx,my):
                    item_idx = (item_idx+1) % len(item_keys); load_item(item_idx)

        if playing and not vs.done:
            step_timer += dt
            if step_timer >= DELAYS[speed]:
                step_timer = 0
                entry = sim_step(vs)
                logs.insert(0, entry)
                if len(logs) > 40: logs.pop()
                ag_pulse = 1.0

        if vs.done:
            playing = False

        screen.fill(cfg.C["bg"])
        n  = cfg.NODES
        ra = 200 if vs.last_received > 0 else 55
        oa = 180 if vs.last_order    > 0 else 46
        da = 180 if vs.last_demand   > 0 else 55

        draw.arrow(screen, cfg.C["teal"],
                   n["sup"]["rx"], n["sup"]["cy"], n["dc"]["lx"],  n["dc"]["cy"], ra)
        draw.arrow(screen, cfg.C["blue"],
                   n["dc"]["cx"],  n["dc"]["ty"], n["sup"]["cx"],  n["sup"]["ty"]+10, oa, dashed=True)
        draw.arrow(screen, cfg.C["amber"],
                   n["dc"]["rx"],  n["dc"]["cy"], n["store"]["lx"], n["store"]["cy"], da)

        draw.flow_dots(screen, cfg.C["teal"],
                       n["sup"]["rx"], n["sup"]["cy"], n["dc"]["lx"], n["dc"]["cy"],
                       dot_phase, bool(vs.pipeline))
        draw.flow_dots(screen, cfg.C["amber"],
                       n["dc"]["rx"], n["dc"]["cy"], n["store"]["lx"], n["store"]["cy"],
                       dot_phase, vs.last_demand > 0)

        draw.draw_agent_line(screen, tick)

        draw.draw_agent_node(screen, vs, ag_pulse)
        draw.draw_supplier_node(screen, vs)
        draw.draw_warehouse_node(screen, vs)
        draw.draw_store_node(screen, vs)
        draw.draw_pipeline_dots(screen, vs)

        draw.draw_header(screen, vs)
        draw.draw_kpis(screen, vs)

        draw.draw_inventory_chart(screen, vs)
        draw.draw_cost_panel(screen, vs)
        draw.draw_pipeline_panel(screen, vs)
        draw.draw_service_chart(screen, vs)

        draw.draw_log(screen, logs)
        draw.draw_legend(screen)
        draw.draw_disruption_banner(screen, vs, tick)

        draw_controls(screen, item_keys, item_idx, playing, speed)

        pygame.display.flip()

if __name__ == "__main__":
    main()