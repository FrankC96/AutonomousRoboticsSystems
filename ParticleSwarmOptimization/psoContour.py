import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import numpy as np
import random
import matplotlib

# Benchmark functions
def rosenbrock(x, y):
    r = (-x)**2 + 100*(y - (x)**2)**2
    
    return r

def rastrigin(x, y):
    r = 10*2 + x**2 - 10*np.cos(2*np.pi*x) + y**2 - 10*np.cos(2*np.pi*y)

    return r

# Gradients of benchmark functions
def rosenbrock_der(x, y):
    g = np.array([x*(2 - 400*(y - x**2)), 200*(y - x**2)])

    return g

def rastrigin_der(x, y):
    g = np.array([2*x + 20*np.pi*np.sin(2*np.pi*x),
                  2*y + 20*np.pi*np.sin(2*np.pi*y)])

    return g

class GradientDescent():
    def __init__(self, func, grad, domain, alpha=0.0001):
        self.alpha = alpha
        self.func = func
        self.grad = grad
        self.domain = domain

    def initialize(self, point):
        start = point

        return start

    def minimize(self, iters, point):
        current_point = self.initialize(point)
        path = []
        for i in range(iters):
            new_point = current_point - self.alpha*self.grad(current_point[0], current_point[1])
            current_point = new_point
            path.append(current_point)

        return path

class PSO():
    def __init__(self, particles, w, c1, c2):
        self.particles = particles
        self.w = w # inertia parameter
        self.c1 = c1
        self.c2 = c2
        self.r1 = round(random.uniform(0, 1), 2)
        self.r2 = round(random.uniform(0, 1), 2)
        self.gbest = None
        self.positions = [] 
        self.velocities = []
        self.pbest = []
        self.f_values = []
        self.history = []
        
    def initialize_particles(self, domain, func):
        for particle in range(self.particles):
            x, y = random.uniform(domain[0], domain[1]), random.uniform(domain[0], domain[1])
            self.positions.append(np.array([x, y]))
            self.velocities.append(np.array([0, 0]))
            self.pbest.append(np.array([x, y]))
            self.f_values.append(func(x, y))
        self.gbest = self.pbest[np.argmax(self.f_values)]
        self.history.append(self.positions.copy())
            
    def minimize(self, iters, func, domain):
        self.initialize_particles(domain, func)
        path = []
        for i in range(iters):
            for particle in range(self.particles):
                new_v = (self.w*self.velocities[particle] + self.c1*self.r1*(self.pbest[particle] - self.positions[particle])
                         + self.c2*self.r2*(self.gbest - self.positions[particle]))
                self.velocities[particle] = new_v
                self.positions[particle] = self.positions[particle] + new_v
                x, y = self.positions[particle][0], self.positions[particle][1]
                new_f = func(x, y)
                if new_f < self.f_values[particle]:
                    self.pbest[particle] = self.positions[particle]
                self.f_values[particle] = new_f
                f_best = func(self.gbest[0], self.gbest[1])
                if new_f < f_best:
                    self.gbest = self.positions[particle]
            #print ('Value: ', func(self.gbest[0], self.gbest[1]))
            path.append((self.gbest[0], self.gbest[1]))
            self.history.append(self.positions.copy())

            
        return path

# Run the optimizers
iters = 100
domain = (-5, 5)

pso = PSO(20, 0.5, 0.5, 0.5) # particles, w, c1, c2
pso_path = pso.minimize(iters, rosenbrock, domain)

gd = GradientDescent(rosenbrock, rosenbrock_der, domain)
gd_path = gd.minimize(iters, np.array([-1.6, 2]))


# Make some data points first to plot the contour later
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

# Plot the contour and make the animation
fig, axs = plt.subplots(2, 2, figsize=(10, 5))
fig.suptitle('Pso vs Gradient Descent')
xdata2, ydata2 = [], []
xdata3, ydata3 = [], []
xdata4, ydata4 = [], []
axs = axs.flatten()
ln2 = axs[1].plot([], [])[0]
ln4 = axs[3].plot([], [])[0]
axs[0].contourf(X, Y, Z, 50, cmap='viridis')
axs[0].contour(X, Y, Z, colors='black', linewidths=0.4)
scat = axs[0].scatter([x[0] for x in pso.history[0]], [y[1] for y in pso.history[0]], marker='x', color='red')
axs[2].contourf(X, Y, Z, 50, cmap='viridis')
axs[2].contour(X, Y, Z, colors='black', linewidths=0.4)
ln3 = axs[2].plot([], [], ls='--', color='red', linewidth=1.5)[0] # For the animation
axs[0].set_xlim(-5, 5)
axs[0].set_ylim(-5, 5)
axs[1].set_xlim(0, iters)
axs[1].set_ylim(0, 10)
axs[3].set_xlim(0, iters)
axs[3].set_ylim(0, 40)

# Give titles to the plots
axs[0].set_title('Pso visualization')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[1].set_xlabel('Iterations')
axs[1].set_ylabel('Loss')
axs[2].set_title('Gradient Descent visualization')
axs[2].set_xlabel('x')
axs[2].set_ylabel('y')
axs[3].set_xlabel('Iterations')
axs[3].set_ylabel('Loss')


def update(frame):
    data = np.array([[x_[0] for x_ in pso.history[frame]], [y_[1] for y_ in pso.history[frame]]])
    scat.set_offsets(data.T)
    xdata2.append(frame)
    ydata2.append(rosenbrock(pso_path[frame][0], pso_path[frame][1]))
    ln2.set_data(xdata2, ydata2)

    xdata3.append(gd_path[frame][0])
    ydata3.append(gd_path[frame][1])
    ln3.set_data(xdata3, ydata3)

    xdata4.append(frame)
    ydata4.append(rosenbrock(gd_path[frame][0], gd_path[frame][1]))
    ln4.set_data(xdata4, ydata4)
    #print (rosenbrock(gd_path[frame][0], gd_path[frame][1]))
    return scat, ln2, ln3, ln4

ani = FuncAnimation(fig, update, frames=np.linspace(0, iters-1, iters).astype('int32'),
                    blit=True, repeat=False)


##ani.save('Rosen_GD.gif', writer=writergif)
plt.show()
