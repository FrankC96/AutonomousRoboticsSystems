from pso import PSO
from cost_function import cost
from functools import partial

"""
Sample code, demonstrating the Particle Swarm Optimization algorithm minimizing the 
rastrigin benchmark function in 10 dimensions while keeping a constant inertial weight.
A next step optimizing this code could be moving all the appropriate arrays calculations
to the GPU by using famous packages like Pytorch/Tensorflow. Also adaptive laws for
the inertia weight can be implemented in order to reduce no. of epochs and no. of
particles needed. 
"""
cost = partial(cost, arg="rastrigin")

pso = PSO(cost,
          dimensions=10,
          w=0.01,
          n_particles=200,
          max_iter=1000)

res = pso.minimize()
print(f"Solution found at {res}.")

