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

    def __init__(self, fps, networks: list[Network], env_config, robot_config):
        self.fps = fps
        self.env_config = env_config
        self.robot_config = robot_config
        self.networks = networks

    def run_init(self, draw=False):
        """
        Initialise the environment and the robot for the simulation.
        """
        env = make_env(self.env_config, draw)
        robots = [make_robot(self.robot_config, env) for _ in self.networks]

        return env, robots

    def run(self, draw=False):
        """
        Performs a simulation with the robot. Returns the fitness value.
        """
        # TODO: run and export simulation without
        # pygame opening a window

        env, robots = self.run_init(draw)
        clock = pg.time.Clock()

        start_time = time.time()

        # Run loop
        running = True
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False

            if draw:
                # Draw environment
                env.draw()

                # Draw dust
                for robot in robots:
                    for x, y in robot.dust:
                        pg.draw.circle(env.surface, LIGHTER_GRAY, (x, y), 1)

            # Movement
            for robot, network in zip(robots, self.networks):
                sensors = robot.sensors_out / MAX_SENSOR_DISTANCE
                motors = network(sensors) * MAX_MOTOR_SPEED

                robot.set_motors(motors, self.fps)
                robot.move(self.fps)

                if draw:
                    # Draw robot
                    robot.draw()

            if draw:
                pg.display.update()

            clock.tick(self.fps)

            if time.time() - start_time > SIMULATION_SECS:
                running = False

        pg.quit()

        fitness_values = []
        for robot in robots:
            fitness_values.append(
                self.fitness(
                    robot.dist_travelled, robot.collected_dust, robot.collisions_num
                )
            )

        return fitness_values

    def fitness(self, dist_travelled, collected_dust, collisions_num):
        """
        Fitness function.
        """
        return (
            FITNESS_WEIGHTS[0] * dist_travelled
            + FITNESS_WEIGHTS[1] * collected_dust
            - FITNESS_WEIGHTS[2] * collisions_num
        )
