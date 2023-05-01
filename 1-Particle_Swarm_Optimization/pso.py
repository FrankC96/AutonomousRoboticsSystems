import numpy as np


class PSO:
    def __init__(self,
                 cost,
                 max_iter: int = 100,
                 n_particles: int = 20,
                 dimensions: int = 2,
                 b_low: float = -10.0,
                 b_up: float = 10.0,
                 w: float = 0.01,
                 phi_p: float = 2,
                 phi_g: float = 2,
                 ):
        """
        cost: expects a function object without calling it.
        max_iter: expects an integer for how many epochs the algorithm will run.
        n_particles: expects an integer for how many particles we want to deploy on the search space.
        dimensions: expects an integer for how many dimensions we want the cost function to have.
        b_low: expects a float defining the lowest boundary of the search space.
        b_up: expects a float defining the upper boundary of the search space.
        w: expects a float defining the confidence for every step we take in the search space.
        phi_p: expects a float defining the cognitive coefficient.
        phi_g: expects a float defining the social coefficient.
        """
        self.max_iter = max_iter
        self.n_particles = n_particles
        self.dimensions = dimensions

        self.b_low = b_low
        self.b_up = b_up
        self.w = w
        self.phi_p = phi_p
        self.phi_g = phi_g

        self.cost = cost

        self.particle_position = np.empty([self.dimensions, self.n_particles])
        self.particle_velocity = np.empty([self.dimensions, self.n_particles])
        self.temp_position = np.empty([self.dimensions, 1])
        self.best_position = np.empty([self.dimensions, 1])

    def init_population(self):
        """
        Initialize the initial population position and velocity by the rules found at :
        https://en.wikipedia.org/wiki/Particle_swarm_optimization
        Also initialize the initial arrays for temporary and best solutions required by the algorithm from the
        aforementioned article in wikipedia.
        """
        self.particle_position = -np.random.uniform(self.b_low, self.b_up, [self.dimensions, self.n_particles])
        self.temp_position = self.particle_position
        self.best_position = self.b_up * np.ones([self.dimensions, 1])

        for k in range(self.particle_position.shape[1]):
            if self.cost(self.temp_position[:, k]) < self.cost(self.best_position):
                self.best_position = self.temp_position[:, k]

        self.particle_velocity = np.random.uniform(-np.abs(self.b_up - self.b_low), np.abs(self.b_up - self.b_low),
                                              [self.dimensions, self.n_particles])

    def update_pos(self, i):
        """
        A function for updating the particles positions and velocities vectorized, calculations are performed
        per particle instead of iterating for each particle and each specific dimension.
        """
        rho_p = np.random.random_sample(1)
        rho_g = np.random.random_sample(1)

        self.particle_velocity[:, i] = self.w * self.particle_velocity[:, i] + self.phi_p * rho_p * (self.temp_position[:, i] - self.particle_position[:, i]) + self.phi_g * rho_g * (self.best_position - self.particle_position[:, i])
        self.particle_position[:, i] = self.particle_position[:, i] + self.particle_velocity[:, i]

    def update_best(self, i):
        """
        A function searching to find a better particle that performs better from the previous epoch.
        """
        if self.cost(self.particle_position[:, i]) < self.cost(self.temp_position[:, i]):
            self.temp_position[:, i] = self.particle_position[:, i]

        if self.cost(self.particle_position[:, i]) < self.cost(self.best_position):
            self.best_position = self.particle_position[:, i]

    def minimize(self):
        """
        The main minimization function where also the particles velocities are being limited to the search space per
        epoch.
        """
        self.init_population()

        for e in range(self.max_iter):
            print(e)
            for p in range(self.n_particles):

                self.particle_position[self.particle_position > self.b_up] = self.b_up
                self.particle_position[self.particle_position < self.b_low] = self.b_low

                self.update_pos(p)
                self.update_best(p)

        return self.best_position
