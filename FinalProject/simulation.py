import numpy as np
import pygame as pg
import time

from colors import *
from environment import *
from robot import *
from network import *


SIMULATION_SECS = 10


class Simulation:
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
        # TODO: run and export simulation without
        # pygame opening a window

        env, robot = self.run_init()
        clock = pg.time.Clock()

        start_time = time.time()
        start_pos = robot.pos.copy()

        # Run loop
        running = True
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False

            # Movement
            # TODO: move robot based on sensor input
            sensors = np.random.uniform(-2, 2, (12,))
            motors = self.network(sensors)
            motors = (robot.motors + motors) / 2

            robot.set_motors(motors, self.fps)
            robot.move(self.fps)

            # Draw environment and robot
            env.draw()
            robot.draw()

            pg.display.update()
            clock.tick(self.fps)

            # TODO: terminate simulation when
            if time.time() - start_time > SIMULATION_SECS:
                running = False

        end_pos = robot.pos.copy()

        pg.quit()

        return self.fitness(start_pos, end_pos)

    def fitness(self, start, end):
        # TODO: fitness function
        return np.linalg.norm(end - start)
