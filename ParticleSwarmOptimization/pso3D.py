import scipy
import numpy as np
import matplotlib.pyplot as plt

from cost_function import cost


def rosenbrock_der(x, y):
    g = np.array([x*(2 - 400*(y - x*2)), 200*(y-x**2)])
    return g


def rastrigin_der(x, y):
    g = np.array([2*x + 20*np.pi*np.sin(2*np.pi*x), 2*y + 20*np.pi*np.sin(2*np.pi*y)])
    return g

# Visualizing the loss landscape, 3D case
Cost = 'rastrigin'

bounds = 5

if Cost == 'rosenbrock':
    bounds = 3
elif Cost == 'rastrigin':
    bounds = 10

# Plotting preparation
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

x = np.linspace(-bounds, bounds, 800)
y = np.linspace(-bounds, bounds, 800)
X, Y = np.meshgrid(x, y)
Z = cost([X, Y])


n_particles = 20
dimensions = 2

w = -0.3
w_damp = 0.002
w_min = -2
w_max = -0.1

iter_max = 20

b_low = -bounds
b_upper = bounds

phi_p = 0.2  # Cognitive coeff
phi_g = 0.8  # Social coeff

# Initialize particles positions.
particle_position = -np.random.uniform(b_low, b_upper, [dimensions, n_particles])

# Initialize iteration's best know positions.
temp_position = particle_position

# Initialize global best position.
best_position = np.array([np.inf, np.inf])

for k in range(particle_position.shape[1]):
    if cost(temp_position[:, k]) < cost(best_position):
        best_position = temp_position[:, k]

# Initialize particles velocity.
particle_velocity = np.random.uniform(-np.abs(b_upper - b_low), np.abs(b_upper - b_low), [dimensions, n_particles])

alpha = 0.001
current_point = [2, 3]

path = []
for _ in range(1000):
    new_point = current_point - alpha * rosenbrock_der(current_point[0], current_point[1])
    current_point = new_point
    path.append(current_point)

# Do 20 iterations max.
for e in range(iter_max-1):
    # For each particle.
    for i in range(n_particles):

        rho_p = np.random.random_sample(1)  # Stochastic parameter no.1
        rho_g = np.random.random_sample(1)  # Stochastic parameter no.2

        # Update particles position and velocity.
        particle_velocity[:, i] = w * particle_velocity[:, i] + phi_p * rho_p * (temp_position[:, i] - particle_position[:, i]) + phi_g * rho_g * (best_position - particle_position[:, i])
        particle_position[:, i] = particle_position[:, i] + particle_velocity[:, i]

        if cost(particle_position[:, i]) < cost(temp_position[:, i]):
            temp_position[:, i] = particle_position[:, i]

        if cost(particle_position[:, i]) < cost(best_position):
            best_position = particle_position[:, i]

    # Plot results.
    ax.cla()
    ax.plot_wireframe(X, Y, Z, color = 'k', linewidth=0.2)
    image = ax.scatter3D([
        particle_position[0][n] for n in range(n_particles)],
        [particle_position[1][n] for n in range(n_particles)],
        [cost([particle_position[0][n], particle_position[1][n]]) for n in range(n_particles)], c='r', linewidths=2, marker="^")
    ax.set(xlim=(-10, 10), ylim=(-10, 10))
    ax.set_title(["Iteration: ", e, "Minima found at: ", cost([best_position[0], best_position[1]]), " at position: ", best_position])
    plt.pause(0.05)

    # Inertia update
    w = w
    # w = w_min + (w_max - w_min) * (iter_max - k) / iter_
print("Gradient Descent: ", cost(path[-1]), path[-1])
print("PSO: ", cost(best_position), best_position)
