import numpy as np

from network import *
from simulation import *


class Evolution:
    def __init__(self, fps, nets_config, env_config, robot_config):
        self.nets_num = nets_config["num"]
        self.nets = [Network(nets_config["layers"]) for _ in range(self.nets_num)]

        self.fps = fps
        self.env_config = env_config
        self.robot_config = robot_config

    # ==================== Evaluation ====================

    def evaluate_single(self, net):
        sim = Simulation(self.fps, net, self.env_config, self.robot_config)
        eval = sim.run()

        return eval

    def evaluate_all(self):
        evals = []
        for net in self.nets:
            evals.append(self.evaluate_single(net))

        return evals

    # TODO: selection, crossover & mutation, reproduction
