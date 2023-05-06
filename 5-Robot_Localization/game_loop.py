import pygame as pg
import numpy as np

from colors import *
from environment import *
from robot import *
from beacons import *

from kalman_filter import KalmanFilter

pg.font.init()


FPS = 30
clock = pg.time.Clock()

WIDTH = 1280
HEIGHT = 720
size = (WIDTH, HEIGHT)
screen = pg.display.set_mode(size)


def check_collisions(clipped_line):

    dist = []
    start = []
    if clipped_line:
        start, end = clipped_line
        x1, y1 = start
        dist.append(
            np.sqrt((robot.pos[0] - x1) ** 2 + (robot.pos[1] - y1) ** 2) - robot.radius
        )
    else:
        dist.append(200)

    return [dist, [start]]


def draw_Radar(
    radar_center, angle_dev, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11
):
    angles = np.arange(0, 360, 30)
    radar_len = 200

    for line_index, k in enumerate(angles):
        r_x = radar_center[0] + np.cos(np.radians(k) - angle_dev) * radar_len
        r_y = radar_center[1] + np.sin(np.radians(k) - angle_dev) * radar_len

        color = (0, 0, 0)
        if k == 0:
            color = (255, 0, 0)

        d = []
        for idx, o in enumerate(obs):
            clipped_line = o.clipline(((radar_center), (r_x, r_y)))
            d.append(check_collisions(clipped_line))

            if d[0][1][0]:
                pg.draw.line(
                    screen, color, radar_center, (d[0][1][0][0], d[0][1][0][1]), 1
                )
            else:
                pg.draw.line(screen, color, radar_center, (r_x, r_y), 1)

            D = min(d)
            if line_index == 0:
                d0 = D[0]
            if line_index == 1:
                d1 = D[0]
            if line_index == 2:
                d2 = D[0]
            if line_index == 3:
                d3 = D[0]
            if line_index == 4:
                d4 = D[0]
            if line_index == 5:
                d5 = D[0]
            if line_index == 6:
                d6 = D[0]
            if line_index == 7:
                d7 = D[0]
            if line_index == 8:
                d8 = D[0]
            if line_index == 9:
                d9 = D[0]
            if line_index == 10:
                d10 = D[0]
            if line_index == 11:
                d11 = D[0]

    print_sensors(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11)


def print_sensors(
    d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11
):  # a list with distances for each of the 12 lines
    font = pg.font.Font(None, 25)

    screen.blit(
        font.render(f"Sensor 0 {np.round(d0, 2)}", True, (255, 255, 255)), (0, 20)
    )
    screen.blit(
        font.render(f"Sensor 1 {np.round(d1, 2)}", True, (255, 255, 255)), (200, 20)
    )
    screen.blit(
        font.render(f"Sensor 2 {np.round(d2, 2)}", True, (255, 255, 255)), (400, 20)
    )
    screen.blit(
        font.render(f"Sensor 3 {np.round(d3, 2)}", True, (255, 255, 255)), (600, 20)
    )
    screen.blit(
        font.render(f"Sensor 4 {np.round(d4, 2)}", True, (255, 255, 255)), (800, 20)
    )
    screen.blit(
        font.render(f"Sensor 5 {np.round(d5, 2)}", True, (255, 255, 255)), (1000, 20)
    )
    screen.blit(
        font.render(f"Sensor 6 {np.round(d6, 2)}", True, (255, 255, 255)), (0, 50)
    )
    screen.blit(
        font.render(f"Sensor 7 {np.round(d7, 2)}", True, (255, 255, 255)), (200, 50)
    )
    screen.blit(
        font.render(f"Sensor 8 {np.round(d8, 2)}", True, (255, 255, 255)), (400, 50)
    )
    screen.blit(
        font.render(f"Sensor 9 {np.round(d9, 2)}", True, (255, 255, 255)), (600, 50)
    )
    screen.blit(
        font.render(f"Sensor 10 {np.round(d10, 2)}", True, (255, 255, 255)), (800, 50)
    )
    screen.blit(
        font.render(f"Sensor 11 {np.round(d11, 2)}", True, (255, 255, 255)), (1000, 50)
    )


# Initialize objects
BORDER_LTWH = (80, 100, 1120, 600)
env = Environment(screen, *BORDER_LTWH)
# Obstacles
OBSTACLE_1_LTWH = (350, 50, 50, 500)
OBSTACLE_2_LTWH = (600, 250, 50, 500)

# Border obstacles
bord1 = (80, 100, 1120, 1)
bord2 = (80, 100, 1, env.border.bottom + 100)
bord3 = (80, env.border.bottom, env.border.width, 1)
bord4 = (WIDTH - 80, 0, 1, env.border.bottom)

# Initialize beacons
b1 = Beacon(screen, (80, 450))
b2 = Beacon(screen, (350, 500))
b3 = Beacon(screen, (80, 600))
# b4 = Beacon(screen, (80, 400))

env.add_obstacle(*OBSTACLE_1_LTWH)
env.add_obstacle(*OBSTACLE_2_LTWH)

env.add_obstacle(*bord1)
env.add_obstacle(*bord2)
env.add_obstacle(*bord3)
env.add_obstacle(*bord4)

obs = env.return_obs()
bords = env.return_bords()

robot = Robot(pos=(500, 500), radius=40, env=env)

d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11 = (
    (),
    (),
    (),
    (),
    (),
    (),
    (),
    (),
    (),
    (),
    (),
    (),
)
font = pg.font.Font(None, 36)

P = 5 * np.eye(3)
Q = 100 * np.eye(2)
R = 0 * np.eye(3)

kf = KalmanFilter(x=[*robot.pos, robot.theta], u=[0, 0], meas=0, P=P, Q=Q, R=R, dt=0.33, lm=[])
# f = KalmanFilter(dim_x=3, dim_z=2)

def GAME_LOOP(running):
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

            # Controls
            if event.type == pg.KEYDOWN:
                # Positive increment left wheel
                if event.key == pg.K_w:
                    robot.update_motors((1, 0), FPS)

                # Negative increment left wheel
                if event.key == pg.K_s:
                    robot.update_motors((-1, 0), FPS)

                # Positive increment right wheel
                if event.key == pg.K_o:
                    robot.update_motors((0, 1), FPS)

                # Negative increment right wheel
                if event.key == pg.K_l:
                    robot.update_motors((0, -1), FPS)

                # Both motor speeds zero
                elif event.key == pg.K_x:
                    robot.reset_motors()

                # Positive increment both wheels
                elif event.key == pg.K_t:
                    robot.update_motors((1, 1), FPS)

                # Negative increment both wheels
                elif event.key == pg.K_g:
                    robot.update_motors((-1, -1), FPS)


        # Movement
        robot.move(FPS)

        # Draw environment and robot
        env.draw()
        robot.draw()

        b1.draw()
        b1.check_inside(robot.pos)

        b2.draw()
        b2.check_inside(robot.pos)

        b3.draw()
        b3.check_inside(robot.pos)

        # b4.draw()
        # b4.check_inside(robot.pos)

        lm = (b1.lm, b2.lm, b3.lm, b3.lm)
        draw_Radar(
            robot.pos, robot.theta, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11
        )

        kf.u[0] = 2 / (robot.VL + robot.VR)
        kf.u[1] = (robot.VR - robot.VL) / (robot.radius * 2)
        kf.predict()
        if lm[0] and lm[1] and lm[2]:

            kf.lm = (lm[0], lm[1], lm[2])
            kf.compute_loc()
            kf.correct()

        print(f"filter {kf.x} | true {(robot.pos, robot.theta)}.")
        pg.draw.circle(screen, (255, 0, 0), (kf.x[0], kf.x[1]), 20)
        pg.display.flip()

        clock.tick(FPS)
        # print (clock.get_fps())

    pg.quit()


if __name__ == "__main__":
    GAME_LOOP(True)
