import pygame as pg
import numpy as np

from colors import *
from environment import *


SENSOR_DISTANCE = 200


class Robot:
    """
    A class to handle the robot's movements and appearance
    within an environment.
    """

    def __init__(self, pos, radius, env: Environment):
        # Initialise attributes of the robot
        self.pos = np.array(pos).astype("float64")  # (x, y)
        self.radius = radius
        self.motors = np.array([0.0, 0.0])  # Left wheel, Right wheel
        self.VL = 0
        self.VR = 0
        self.accel = 1  # How fast to increment the speed of each wheel
        self.theta = np.pi / 2  # Angle of the robot with the x axis in rads
        self.velocity = np.array([0.0, 0.0])
        # Initialize the environment in which the robot moves
        self.env = env
        self.update_sensors()

    def set_motors(self, motors, fps):
        """
        Sets the motor speeds.
        """
        self.motors = np.array(motors)
        self.VL = fps * self.motors[0]
        self.VR = fps * self.motors[1]

    def reset_motors(self):
        """
        Sets the motor speeds to 0.
        """
        self.motors = np.array([0.0, 0.0])
        self.VL = 0
        self.VR = 0

    def update_motors(self, update, fps):
        """
        Updates the speeds of each motor according to user input.
        """
        self.motors += self.accel * np.array(update)
        self.VL = fps * self.motors[0]
        self.VR = fps * self.motors[1]
        # print('Motors: ', self.motors)

    def keyboard_input(self, fps, event):
        # Positive increment left wheel
        if event.key == pg.K_w:
            self.update_motors((1, 0), fps)

        # Negative increment left wheel
        if event.key == pg.K_s:
            self.update_motors((-1, 0), fps)

        # Positive increment right wheel
        if event.key == pg.K_o:
            self.update_motors((0, 1), fps)

        # Negative increment right wheel
        if event.key == pg.K_l:
            self.update_motors((0, -1), fps)

        # Both motor speeds zero
        elif event.key == pg.K_x:
            self.reset_motors()

        # Positive increment both wheels
        elif event.key == pg.K_t:
            self.update_motors((1, 1), fps)

        # Negative increment both wheels
        elif event.key == pg.K_g:
            self.update_motors((-1, -1), fps)

    def calculate_dpos_new_theta(self, fps):
        """
        Calculates the displacement and the new angle of the robot assuming
        there are no borders or obstacles to stop its movement.
        """
        # Forward movement
        if self.motors[0] == self.motors[1]:
            x_component = np.cos(self.theta)  # x component of direction vector
            y_component = np.sin(-self.theta)  # y component of direction vector
            new_theta = self.theta

            return self.motors * np.array([x_component, y_component]), new_theta

        # Angular movement
        else:
            R = self.radius * ((self.VL + self.VR) / (self.VR - self.VL))
            omega = (self.VR - self.VL) / (self.radius * 2)
            ICC = np.array(
                [
                    self.pos[0] - R * np.sin(-self.theta),
                    self.pos[1] + R * np.cos(self.theta),
                ]
            )
            delta_t = 1 / fps
            A = np.array(
                [
                    [np.cos(omega * delta_t), -np.sin(omega * delta_t), 0],
                    [np.sin(omega * delta_t), np.cos(omega * delta_t), 0],
                    [0, 0, 1],
                ]
            )
            b = np.array([self.pos[0] - ICC[0], self.pos[1] - ICC[1], self.theta])
            c = np.array([ICC[0], ICC[1], omega * delta_t])
            new_x, new_y, new_theta = np.matmul(A, b) + c

            return np.array((new_x, new_y)) - self.pos, new_theta

    def collision_handler(self, dpos):
        """
        Updates the displacement dpos on the robot according to any potential collisions
        that might affect its movement. This is done by calculating the normal vector n
        of each possible collision and subtracting from dpos the projection of dpos on n
        (see also the "collision_normals_and_current_distances" method from the Border class
        and the "collision_normal_and_current_distance" method from the Obstacle class).
        To make animation smoother, dpos is also updated such that the distance of the robot
        from the collision wall (real or imaginary, in case when collision happens at a corner
        of an obstacle) is exactly 0 (which might not be the case if the collision has been detected
        from a further distance, especially at high speeds).
        """
        dpos_og = dpos

        # Collisions with the border
        normals_and_distances = self.env.border.collision_normals_and_current_distances(
            self.pos, dpos, self.radius
        )
        for collision_normal, current_dist in normals_and_distances:
            dpos += collision_normal * (
                (current_dist - self.radius) - np.dot(dpos_og, collision_normal)
            )

        # Collisions with any obstacles
        for obstacle in self.env.obstacles:
            (
                collision_normal,
                current_dist,
            ) = obstacle.collision_normal_and_current_distance(
                self.pos, dpos, self.radius
            )
            if collision_normal is not None:
                dpos += collision_normal * (
                    (current_dist - self.radius) - np.dot(dpos_og, collision_normal)
                )

        return dpos

    def move(self, fps):
        """
        Updates the position and the angle of the robot.
        """
        dpos, new_theta = self.calculate_dpos_new_theta(fps)

        # Update theta
        self.theta = new_theta

        # Check for collisions and calculate the positional displacement accordingly
        dpos = self.collision_handler(dpos)

        # Update position vector
        self.pos += dpos

        self.update_sensors()

    def draw(self):
        """
        Draws the robot inside the environment.
        """
        # Inner circles of the robot just for fun!
        inn_circles = 4
        pg.draw.circle(self.env.surface, BLACK, self.pos, self.radius)
        for cir in range(inn_circles):
            pg.draw.circle(self.env.surface, WHITE, self.pos, self.radius - cir * 10, 1)

        # Direction line
        x_component = np.cos(self.theta) * self.radius
        y_component = np.sin(-self.theta) * self.radius
        start_pos = self.pos  # Start position of line
        end_pos = start_pos + np.array(
            [x_component, y_component]
        )  # End position of line
        pg.draw.line(self.env.surface, WHITE, start_pos, end_pos)
        self.draw_sensors()

    def draw_sensors(self):
        for sensor in self.sensors:
            pg.draw.line(
                self.env.surface,
                LIGHT_GRAY,
                self.pos + self.radius * sensor,
                self.pos + (self.radius + SENSOR_DISTANCE) * sensor,
            )

    def update_sensors(self):
        self.sensors = [
            np.array(
                (np.cos(k * np.pi / 6 - self.theta), np.sin(k * np.pi / 6 - self.theta))
            )
            for k in range(12)
        ]

    def sensor_output(self, sensor):
        intersections = []
        for obstacle in self.env.obstacles:
            intersection = obstacle.intersection_with_segment(
                self.pos + self.radius * sensor,
                self.pos + (self.radius + SENSOR_DISTANCE) * sensor,
            )
            intersections.append(intersection)

        return intersections


def make_robot(robot_config, env):
    robot = Robot(pos=robot_config["pos"], radius=robot_config["radius"], env=env)
    return robot
