import pygame as pg

from colors import *
from environment import *
from robot import *
from network import *
from simulation import *
from evolution import *
from config import *


def game_loop():
    clock = pg.time.Clock()
    env = make_env(ENV_CONFIG)
    robot = make_robot(ROBOT_CONFIG, env)

    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

            if event.type == pg.KEYDOWN:
                robot.keyboard_input(FPS, event)

        robot.move(FPS)

        env.draw()
        robot.draw()

        pg.display.update()
        clock.tick(FPS)

    pg.quit()


def test_simulation():
    net = Network(NET_LAYERS)
    sim = Simulation(FPS, net, ENV_CONFIG, ROBOT_CONFIG)
    evaluation = sim.run()
    print(evaluation)


def test_evolution():
    evolution = Evolution(FPS, NETS_CONFIG, ENV_CONFIG, ROBOT_CONFIG)
    evolution.evolve(3)


if __name__ == "__main__":
    test_evolution()
