import pygame as pg
import numpy as np
from colors import *
from robot import *

pg.init()

fps = 60
clock = pg.time.Clock()

def make_surface(size, pos): # size=(w, h), pos=(x, y)
    surf = pg.Surface(size)
    rect = surf.get_rect()
    rect.topleft = pos
    return surf, rect
    
# Screen
WIDTH = 1280
HEIGHT = 720
size = (WIDTH, HEIGHT)
screen = pg.display.set_mode(size)
# Split the screen into two surfaces
surf1_height = 150
surf2_height = HEIGHT - surf1_height
surf1, rect1 = make_surface((WIDTH, surf1_height), (0, 0))
surf2, rect2 = make_surface((WIDTH, surf2_height), (0, surf1_height))

# Initialize objects
robot = Robot((200, 200), 40)

def GAME_LOOP(running):
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_w:
                    robot.update_motor((robot.accel,0))
                if event.key == pg.K_s:
                    robot.update_motor((-robot.accel,0))
                if event.key == pg.K_o:
                    robot.update_motor((0, robot.accel))
                if event.key == pg.K_l:
                    robot.update_motor((0, -robot.accel))
                elif event.key == pg.K_x:
                    robot.motors = np.array([0.0, 0.0])
                elif event.key == pg.K_t:
                    robot.update_motor((robot.accel, robot.accel))
                elif event.key == pg.K_g:
                    robot.update_motor((-robot.accel, -robot.accel))

        # Movement
        if robot.motors[0] == robot.motors[1]:
            robot.move()
                    
        # Draw section
        surf1.fill(DARK_GRAY)
        surf2.fill(LIGHT_GRAY)
        robot.draw(surf2)
        screen.blit(surf1, rect1.topleft)
        screen.blit(surf2, rect2.topleft)
        pg.draw.rect(screen, WHITE, rect1, 1) # Outline surface 1
        pg.draw.rect(screen, WHITE, rect2, 1) # Outline surface 2

        pg.display.update()
        clock.tick(fps)

    pg.quit()
        








