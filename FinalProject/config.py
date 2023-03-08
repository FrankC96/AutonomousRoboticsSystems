FPS = 30

ENV_CONFIG = {
    "size": (1280, 720),
    "border": (80, 100, 1120, 600),
    "obstacles": [
        (300, 300, 100, 100),
        (400, 400, 100, 100),
        (800, 500, 100, 100),
        (900, 200, 200, 300),
    ],
}

ROBOT_CONFIG = {"pos": (200, 200), "radius": 40}

NET_LAYERS = [12, 4, 2]
NETS_CONFIG = {"num": 3, "layers": NET_LAYERS}
