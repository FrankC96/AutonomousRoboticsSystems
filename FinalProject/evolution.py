import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from network import *
from simulation import *


OUTPUT_PATH = "./plots"


class Evolution:
    """
    A class to perform the evolutionary algorithm and produce plots with the results.
    """

    def __init__(self, fps, nets_config, env_config, robot_config):
        self.nets_num = nets_config["num"]
        self.net_layers = nets_config["layers"]
        self.nets = [Network(self.net_layers) for _ in range(self.nets_num)]

        self.fps = fps
        self.env_config = env_config
        self.robot_config = robot_config

        # Histories for plots
        self.make_output_dir(OUTPUT_PATH)
        self.history_best = []
        self.history_average = []
        self.history_worst = []
        self.history_diversity = []

    # ==================== Evaluation ====================

    def evaluate_all(self, draw=False):
        """
        Evaluate all networks.
        """
        sim = Simulation(self.fps, self.nets, self.env_config, self.robot_config)
        evals = sim.run(draw)

        return np.array(evals)

    def sort_by_evaluation(self, draw=False):
        """
        Sort the networks in descending order (i.e. best to worst) by their evaluation.
        """
        nets_sorted, values_sorted = zip(
            *sorted(
                zip(self.nets, self.evaluate_all(draw)),
                key=lambda nv: nv[1],
                reverse=True,
            )
        )
        return nets_sorted, np.array(values_sorted)

    def diversity(self):
        """
        Calculate the diversity of the network population by summing the
        euclidean differences of their weights and biases.
        """
        diversity = 0
        for i, net_0 in enumerate(self.nets):
            for net_1 in self.nets[i + 1 :]:
                diversity += np.sum(
                    np.linalg.norm(x - y) for x, y in zip(net_0.weights, net_1.weights)
                )
                diversity += np.sum(
                    np.linalg.norm(x - y) for x, y in zip(net_0.biases, net_1.biases)
                )

        return diversity

    # TODO: selection, crossover & mutation, reproduction

    # ==================== Evolutionary algorithm ====================

    def log_histories(self, values_sorted):
        """
        Log the histories for plotting.
        """
        self.history_best.append(values_sorted[0])
        self.history_worst.append(values_sorted[-1])
        self.history_average.append(np.average(values_sorted))
        self.history_diversity.append(self.diversity())

    def generation(self, gen, draw=False):
        """
        Creates a new generation of networks from the
        current one.
        """
        new_generation = []
        nets_sorted, values_sorted = self.sort_by_evaluation(draw)
        print(f"Generation {gen}: {values_sorted}")
        print()

        for net in nets_sorted[:-1]:
            for i, w in enumerate(net.weights):
                w += 1 / (gen + 1) * np.sign(w) * np.random.uniform(size=w.shape)
                net.weights[i] = w

            for i, b in enumerate(net.biases):
                b += 1 / (gen + 1) * np.sign(b) * np.random.uniform(size=b.shape)
                net.biases[i] = b

            new_generation.append(net)

        new_generation.append(Network(self.net_layers))

        # Update the organisms with the new generation
        self.nets = new_generation

        # Log the histories for plotting
        self.log_histories(values_sorted)

    def evolve(self, generations, draw=False):
        """
        Implements the evolutionary algorithm.
        """
        # progress_bar = tqdm(range(generations), leave=False)

        for i in range(generations):
            self.generation(i, draw)
            # progress_bar.update()

        # progress_bar.close()

        # Create the plots
        self.plot_history_best_worst_avg()
        self.plot_history_diversity()

    # ==================== Plotting ====================

    def plot_history_best_worst_avg(self):
        """
        Produces and saves a plot of the best, worst and average values history.
        """
        plt.plot(
            list(range(1, len(self.history_best) + 1)), self.history_best, label="best"
        )
        plt.plot(
            list(range(1, len(self.history_average) + 1)),
            self.history_average,
            label="average",
        )
        plt.plot(
            list(range(1, len(self.history_worst) + 1)),
            self.history_worst,
            label="worst",
        )
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/best_worst_avg.png")
        plt.clf()

    def plot_history_diversity(self):
        """
        Produces and saves a plot of the diversity history.
        """
        plt.plot(
            list(range(1, len(self.history_diversity) + 1)), self.history_diversity
        )
        plt.xlabel("Generations")
        plt.ylabel("Diversity")
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/diversity.png")
        plt.clf()

    def make_output_dir(self, path):
        """
        Creates the output directory in which the plots will be saved.
        """
        run = 0
        for name in os.listdir(path):
            prev_run = int(name.split("_")[1])
            if prev_run > run:
                run = prev_run
        run += 1

        self.output_path = f"{path}/run_{run}"
        os.mkdir(self.output_path)
