import numpy as np
from collections.abc import Callable


def random_weights(layers: list[int]):
    """
    Generates random weights sampled from the normal distribution.
    """
    return [np.random.randn(l1, l0) for l1, l0 in zip(layers[1:], layers[:-1])]


def random_biases(layers: list[int]):
    """
    Generates random biases sampled from the normal distribution.
    """
    return [np.random.randn(l, 1) for l in layers[1:]]


class Network:
    """
    Defines the neural network architecture.
    """

    def __init__(
        self,
        layers: list[int],
        weights: list[np.ndarray] = None,
        biases: list[np.ndarray] = None,
        activations: list[Callable] = None,
    ):
        self.layers = layers

        # Set weights
        if weights is None:
            self.weights = random_weights(self.layers)
        else:
            self.weights = weights

        # Set biases
        if biases is None:
            self.biases = random_biases(self.layers)
        else:
            self.biases = biases

        # Set activations
        if activations is None:
            self.activations = [lambda x: np.maximum(x, 0)] * (len(layers) - 2) + [
                lambda x: np.tanh(x)
            ]
        else:
            self.activations = activations

    def __call__(self, x: np.ndarray):
        a = x.reshape(self.layers[0], 1)

        # Feedforward model
        for l in range(len(self.layers) - 1):
            z = self.weights[l] @ a + self.biases[l]
            a = self.activations[l](z)

        return a.reshape(
            len(a),
        )

    def copy(self):
        """
        Create a copy of the current network.
        """
        layers = self.layers.copy()
        weights = [w.copy() for w in self.weights]
        biases = [b.copy() for b in self.biases]
        activations = [a for a in self.activations]
        return Network(layers, weights, biases, activations)
