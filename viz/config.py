import pygame

C = {
    "bg":     (252, 250, 247),
    "card":   (255, 255, 255),
    "border": (220, 215, 208),
    "t1":     ( 28,  25,  23),
    "t2":     (120, 113, 108),
    "t3":     (190, 183, 175),
    "blue":   ( 37,  99, 235),
    "teal":   ( 13, 144, 104),
    "amber":  (180,  83,   9),
    "red":    (192,  43,  43),
    "violet": (109,  57, 192),
}

W, H = 1100, 720

FONT_SM      = None
FONT_MD      = None
FONT_LG      = None
FONT_MONO_SM = None
FONT_MONO_MD = None
FONT_MONO_LG = None

NODES = {
    "agent": {"x": 450, "y": 140, "w": 220, "h": 44},
    "sup":   {"x":  110, "y": 270, "w": 165, "h": 72},
    "dc":    {"x": 450, "y": 240, "w": 220, "h": 110},
    "store": {"x": 800, "y": 268, "w": 165, "h": 72},
}

current_item_key = "—"
current_store_id = "—"
current_item_id  = "—"
model_path       = "models/best_model"


def init_nodes():
    for n in NODES.values():
        n["cx"] = n["x"] + n["w"] // 2
        n["cy"] = n["y"] + n["h"] // 2
        n["lx"] = n["x"]
        n["rx"] = n["x"] + n["w"]
        n["ty"] = n["y"]
        n["by"] = n["y"] + n["h"]


def load_fonts():
    def _f(name, size, bold=False):
        f = pygame.font.SysFont(name, size, bold=bold)
        if f is None:
            f = pygame.font.SysFont(None, size, bold=bold)
        return f

    global FONT_SM, FONT_MD, FONT_LG
    global FONT_MONO_SM, FONT_MONO_MD, FONT_MONO_LG

    FONT_SM      = _f("segoeui",  11)
    FONT_MD      = _f("segoeui",  13)
    FONT_LG      = _f("segoeui",  16, bold=True)
    FONT_MONO_SM = _f("consolas", 10)
    FONT_MONO_MD = _f("consolas", 12)
    FONT_MONO_LG = _f("consolas", 19, bold=True)