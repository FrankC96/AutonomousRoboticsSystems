import numpy as np


def cost(x, arg):

    selection = {
        "rosenbrock": np.sum([0 * ((i+1) - i**2)**2 + (1 - i)**2 for i in x]),
        "rastrigin": 10 * len(x) + np.sum([i**2 - 10 * np.cos(2 * np.pi * i) for i in x])
    }

    return selection.get(arg, "No cost function found.")