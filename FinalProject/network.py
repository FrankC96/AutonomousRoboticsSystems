import numpy as np


def random_weights(layers):
    """
    Generates random weights sampled from the normal distribution.
    """
    return [np.random.randn(l1, l0) for l1, l0 in zip(layers[1:], layers[:-1])]


def random_biases(layers):
    """
    Generates random biases sampled from the normal distribution.
    """
    return [np.random.randn(l, 1) for l in layers[1:]]


class Network:
    """
    Defines the neural network architecture.
    """

    def __init__(self, layers, weights=None, biases=None, activations=None):
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
            self.activations = [lambda x: x] * (len(layers) - 2) + [
                lambda x: np.tanh(x)
            ]
        else:
            self.activations = activations

    def __call__(self, x: np.ndarray):
        a = x.reshape(self.layers[0], 1)

        for l in range(len(self.layers) - 1):
            z = self.weights[l] @ a + self.biases[l]
            a = self.activations[l](z)

        return a.reshape(
            len(a),
        )
