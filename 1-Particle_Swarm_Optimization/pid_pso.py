import numpy as np
import matplotlib.pyplot as plt

from cost_function import pid_eval as cost

# Visualizing the loss landscape, 2D case
Cost = 'pid'

bounds = 5
if Cost == 'rosenbrock':
    bounds = 3
elif Cost == 'rastrigin':
    bounds = 10
elif Cost == 'pid':
    bounds = 1000


n_particles = 5
dimensions = 3

w = -0.02
w_damp = 0.002
w_min = -1
w_max = -0.1

iter_max = 1

b_low = -bounds
b_upper = bounds

phi_p = 0.002 # Cognitive coeff
phi_g = 0.008  # Social coeff

# Initialize particles position
particle_position = np.random.uniform(b_low, b_upper, [dimensions, n_particles])

# Initialize iteration's best positions0
temp_position = particle_position

# Initialize global best position
best_position = np.array([bounds, bounds, bounds])

for k in range(particle_position.shape[1]):
    if cost(temp_position[:, k]) < cost(best_position):
        best_position = temp_position[:, k]

# Initialize particles velocity
particle_velocity = np.random.uniform(-np.abs(b_upper - b_low), np.abs(b_upper - b_low), [dimensions, n_particles])

# Do iter_max iter
for e in range(iter_max):
    for i in range(n_particles):

        rho_p = np.random.random_sample(1)
        rho_g = np.random.random_sample(1)

        particle_velocity[:, i] = w * particle_velocity[:, i] + phi_p * rho_p * (temp_position[:, i] - particle_position[:, i]) + phi_g * rho_g * (best_position - particle_position[:, i])
        particle_position[:, i] = particle_position[:, i] + particle_velocity[:, i]

        print(particle_position[:, i])
        if cost(particle_position[:, i]) < cost(temp_position[:, i]):
            temp_position[:, i] = particle_position[:, i]

        if cost(particle_position[:, i]) < cost(best_position):
            best_position = particle_position[:, i]

    w = w_min + (w_max - w_min) * (iter_max - e) / iter_max
    # w += w * w_damp

print("PSO: ", cost(best_position), best_position)