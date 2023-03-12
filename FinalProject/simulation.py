import numpy as np
import pygame as pg
import time

from colors import *
from environment import *
from robot import *
from network import *


SIMULATION_SECS = 10
MAX_MOTOR_SPEED = 10

FITNESS_WEIGHTS = (2, 10, 1)


class Simulation:
    """
    A class to perform a simulation for a robot controlled by a neural network
    in an environment and calculates the fitness value.
    """

    def __init__(self, fps, network: Network, env_config, robot_config):
        self.fps = fps
        self.env_config = env_config
        self.robot_config = robot_config
        self.network = network

    def run_init(self):
        env = make_env(self.env_config)
        robot = make_robot(self.robot_config, env)

        return env, robot

    def run(self):
        """
        Performs a simulation with the robot. Returns the fitness value.
        """
        # TODO: run and export simulation without
        # pygame opening a window

        env, robot = self.run_init()
        clock = pg.time.Clock()

        start_time = time.time()

        # Run loop
        running = True
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False

            # Movement
            sensors = robot.sensors_out / MAX_SENSOR_DISTANCE
            motors = self.network(sensors) * MAX_MOTOR_SPEED

            robot.set_motors(motors, self.fps)
            robot.move(self.fps)

            # Draw environment and robot
            env.draw()
            robot.draw()

            pg.display.update()
            clock.tick(self.fps)

            if time.time() - start_time > SIMULATION_SECS:
                running = False

        pg.quit()

        fitness_value = self.fitness(
            robot.dist_travelled, robot.collected_dust, robot.collisions_num
        )

        print("Total distance travelled:", robot.dist_travelled)
        print("Total number of dust particles collected:", robot.collected_dust)
        print("Total number of collisions:", robot.collisions_num)
        print("Fitness value:", fitness_value)
        print()

        return fitness_value

    def fitness(self, dist_travelled, collected_dust, collisions_num):
        """
        Fitness function.
        """
        return (
            FITNESS_WEIGHTS[0] * dist_travelled
            + FITNESS_WEIGHTS[1] * collected_dust
            - FITNESS_WEIGHTS[2] * collisions_num
        )
