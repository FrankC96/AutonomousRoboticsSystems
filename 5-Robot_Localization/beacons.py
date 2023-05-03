import pygame as pg
import numpy as np


class Beacon:
    def __init__(self, screen, pos: tuple):
        self.screen = screen
        self.x = pos[0]
        self.y = pos[1]

        self.lm = None
        self.color = (0, 0, 0)

    def check_inside(self, r_pos):
        if 200 > np.sqrt((self.x - r_pos[0])**2 + (self.y - r_pos[1])**2):
            self.color = (255, 0, 0)
            self.lm = (self.x, self.y)
        else:
            self.color = (0 , 0 , 0)
            self.lm = None

    def draw(self):
        pg.draw.circle(self.screen, (255, 0, 0), (self.x, self.y), 5)
        pg.draw.circle(self.screen, self.color, (self.x, self.y), 200, 5)