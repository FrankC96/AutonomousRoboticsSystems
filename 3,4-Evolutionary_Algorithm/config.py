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

ROBOT_CONFIG = {
    "pos": (600, 600),
    "radius": 40,
    "dust particles": 500,
}

NET_LAYERS = [12, 6, 4, 2]
NETS_CONFIG = {"num": 10, "layers": NET_LAYERS}
