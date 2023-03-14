import os
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

from network import *
from simulation import *


OUTPUT_PATH = "./plots"


class Evolution:
    """
    A class to perform the evolutionary algorithm and produce plots with the results.
    """

    def __init__(
        self, fps: int, nets_config: dict, env_config: dict, robot_config: dict
    ):
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

    # ==================== Selection ====================

    def tournament_selection(
        self,
        selected_num: int,
        tournament_size: int,
        nets_sorted: list[Network],
        values_sorted: np.ndarray,
    ):
        """
        Selects a number of networks using tournament selection with tournaments
        of a fixed size.
        """
        selected = []
        selected_values = []
        for _ in range(selected_num):
            # Randomly select individuals
            indices = random.sample(range(len(nets_sorted)), tournament_size)

            # Determine the winner and make a copy
            winner_idx = min(indices)
            selected.append(nets_sorted[winner_idx].copy())
            selected_values.append(values_sorted[winner_idx])

        return selected, np.array(selected_values)

    def rank_based_selection(
        self, selected_num: int, nets_sorted: list[Network], values_sorted: np.ndarray
    ):
        """
        Selects a number of networks using rank-based selection.
        """
        selected = []
        selected_values = []
        n = len(nets_sorted)
        denom = n * (n + 1) / 2

        # Rank based selection
        # p(i) ~ 1 - r(i) / Sum r(i)
        indices = np.random.choice(
            n,
            size=(selected_num,),
            replace=False,
            p=[(1 - (i + 1) / denom) / (n - 1) for i in range(n)],
        ).astype("int")

        # Make a copy of the selected individuals
        for i in indices:
            selected.append(nets_sorted[i].copy())
            selected_values.append(values_sorted[i])

        return selected, np.array(selected_values)

    # ==================== Crossover ====================

    def crossover_average(self, selected: list[Network]):
        """
        Creates a list of networks by taking the average
        of the weights and biases of the parents.
        """
        crossover_networks = []
        weight1 = 0.5
        if len(selected) % 2 == 1:
            selected.pop()

        for i in range(0, len(selected), 2):  # Take each pair of the selected networks
            parent1 = selected[i]
            parent2 = selected[i + 1]
            crossover_weights = []
            crossover_biases = []

            # Crossover weights
            for w1, w2 in zip(parent1.weights, parent2.weights):
                w = weight1 * w1 + (1 - weight1) * w2
                crossover_weights.append(w)

            # Crossover biases
            for b1, b2 in zip(parent1.biases, parent2.biases):
                b = weight1 * b1 + (1 - weight1) * b2
                crossover_biases.append(b)

            crossover_network = Network(
                self.net_layers, crossover_weights, crossover_biases
            )
            crossover_networks.append(crossover_network)

        return crossover_networks

    def crossover_replace(self, selected: list[Network]):
        """
        Creates a list of networks by combining the weights
        and biases of the parents.
        """
        crossover_networks = []
        window = 0.6
        if len(selected) % 2 == 1:
            selected.pop()

        for i in range(0, len(selected), 2):  # Take each pair of the selected networks
            parent1 = selected[i]
            parent2 = selected[i + 1]
            crossover_weights = []
            crossover_biases = []

            # Crossover weights
            for w1, w2 in zip(parent1.weights, parent2.weights):
                w = np.zeros(w1.shape)
                w[: int(w1.shape[0] * window)] = w1[: int(w1.shape[0] * window)]
                w[int(w2.shape[0] * window) :] = w2[int(w2.shape[0] * window) :]
                crossover_weights.append(w)

            # Crossover biases
            for b1, b2 in zip(parent1.biases, parent2.biases):
                b = np.zeros(b1.shape)
                b[: int(b1.shape[0] * window)] = b1[: int(b1.shape[0] * window)]
                b[int(b2.shape[0] * window) :] = b2[int(b2.shape[0] * window) :]
                crossover_biases.append(b)
            crossover_network = Network(
                self.net_layers, crossover_weights, crossover_biases
            )
            crossover_networks.append(crossover_network)

        return crossover_networks

    # ==================== Mutation ====================

    def mutate(self, gen: int, selected: list[Network]):
        mutated_networks = []

        for net in selected:
            for i, w in enumerate(net.weights):
                w += 1 / (gen + 1) * np.sign(w) * np.random.uniform(size=w.shape)
                net.weights[i] = w

            for i, b in enumerate(net.biases):
                b += 1 / (gen + 1) * np.sign(b) * np.random.uniform(size=b.shape)
                net.biases[i] = b

            mutated_networks.append(net)

        return mutated_networks

    # ==================== Evolutionary algorithm ====================

    def log_histories(self, values_sorted: np.ndarray):
        """
        Log the histories for plotting.
        """
        self.history_best.append(values_sorted[0])
        self.history_worst.append(values_sorted[-1])
        self.history_average.append(np.average(values_sorted))
        self.history_diversity.append(self.diversity())

    def generation(self, gen: int, draw=False):
        """
        Creates a new generation of networks from the
        current one.
        """
        new_generation = []
        nets_sorted, values_sorted = self.sort_by_evaluation(draw)
        print(f"Generation {gen}: {values_sorted}")
        print()

        # Selection
        tournament_selected, _ = self.tournament_selection(
            4, 5, nets_sorted, values_sorted
        )
        rank_selected, _ = self.rank_based_selection(6, nets_sorted, values_sorted)

        # Crossover & mutation
        best = nets_sorted[0]
        best_child = self.crossover_replace(nets_sorted[:2])
        tournament_children = self.crossover_replace(tournament_selected)
        rank_children = self.crossover_average(rank_selected)

        best_child_mutated = self.mutate(gen, best_child)[0]
        mutated_rest = self.mutate(gen, nets_sorted[:-2])

        # Reproduction

        # Keep best unchanged
        new_generation.append(best)
        new_generation.append(best_child_mutated)
        # Add the children
        new_generation.extend(tournament_children)
        new_generation.extend(rank_children)
        # Fill in the rest with mutated parents
        while mutated_rest and len(new_generation) < self.nets_num - 1:
            i = random.randint(0, len(mutated_rest) - 1)
            new_generation.append(mutated_rest[i])
            mutated_rest.pop(i)

        new_generation.append(Network(self.net_layers))

        # Update the organisms with the new generation
        self.nets = new_generation

        # Log the histories for plotting
        self.log_histories(values_sorted)

    def evolve(self, generations: int, draw=False):
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

    def make_output_dir(self, path: str):
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
