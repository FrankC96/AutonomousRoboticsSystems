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
    env = make_env(ENV_CONFIG, True)
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
    print()
    net = Network(NET_LAYERS)
    sim = Simulation(FPS, net, ENV_CONFIG, ROBOT_CONFIG)
    evaluation = sim.run()
    print("Fitness:", evaluation)


def test_evolution():
    print()
    evolution = Evolution(FPS, NETS_CONFIG, ENV_CONFIG, ROBOT_CONFIG)
    evolution.evolve(3, draw=True)


if __name__ == "__main__":
    # game_loop()
    # test_simulation()
    test_evolution()
