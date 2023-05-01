import unittest
import numpy as np
from functools import partial

from pso import PSO
from cost_function import cost


class TestPso(unittest.TestCase):

    cost = partial(cost, arg="rosenbrock")
    pso = PSO(cost, max_iter=100)

    pso.init_population()

    # update for 1 particle
    pso.update_pos(1)
    pso.update_best(1)

    def test_type(self):
        """
        Assert all particle related arrays are indeed numpy arrays.
        """
        self.assertIs(type(self.pso.particle_position), np.ndarray)
        self.assertIs(type(self.pso.particle_velocity), np.ndarray)
        self.assertIs(type(self.pso.temp_position), np.ndarray)
        self.assertIs(type(self.pso.best_position), np.ndarray)

    def test_dims(self):
        """
        Assert all particle related arrays have the correct dimensions.
        """
        self.assertEqual(self.pso.particle_position.shape, (self.pso.dimensions, self.pso.n_particles))
        self.assertEqual(self.pso.particle_velocity.shape, (self.pso.dimensions, self.pso.n_particles))
        self.assertEqual(self.pso.temp_position.shape, (self.pso.dimensions, self.pso.n_particles))
        self.assertEqual(self.pso.best_position.shape, (self.pso.dimensions, ))

    def test_sol_bounds(self):
        self.assertLessEqual(self.pso.particle_position.all(), self.pso.stable_thrs)
