import pygame as pg
import numpy as np

from colors import *


class Wall:
    """
    A wall is defined as a line segment with a start and an end.
    This class takes care of calculations such as finding the normal
    vector to a wall and the distance from a point on the plane to a wall.
    """

    def __init__(self, start, end, normal_dir):
        self.start = np.array(start)
        self.end = np.array(end)
        self.vector = self.end - self.start
        normal_sign = 1 if normal_dir == "in" else -1
        self.normal = (
            normal_sign
            * np.array((-self.vector[1], self.vector[0]))
            / np.linalg.norm(self.vector)
        )

    def closest_point_and_distance_from(self, origin):
        """
        This function returns the closest point from a
        wall to another point on the plane (origin), as well
        as the shortest distance from origin to the wall.
        """

        # Let O, S, E be the origin, start of the wall and
        # end of the wall respectively. If the angle OSE is
        # larger than pi/2, then the closest point of the wall
        # to O is S, e.g.:
        #
        #     O
        #      \
        #       \
        #        \
        #         S ------------------- E

        cos_angle_vector_start = (
            np.dot(origin - self.start, self.vector)
            / np.linalg.norm(origin - self.start)
            / np.linalg.norm(self.vector)
        )

        if cos_angle_vector_start < 0:
            closest_point = self.start
            dist = np.linalg.norm(origin - self.start)
            return closest_point, dist

        # Otherwise, if the angle OES is larger than pi/2,
        # then the closest point of the wall to O is E, e.g.:
        #
        #                                    O
        #                                   /
        #                                  /
        #                                 /
        #           S ------------------ E

        cos_angle_vector_end = (
            np.dot(origin - self.end, -self.vector)
            / np.linalg.norm(origin - self.end)
            / np.linalg.norm(self.vector)
        )

        if cos_angle_vector_end < 0:
            closest_point = self.end
            dist = np.linalg.norm(origin - self.end)
            return closest_point, dist

        # Otherwise, calculate the distance of O
        # from the line formed by S,E.
        dist = np.abs(
            np.cross(self.vector, origin - self.start)
            / np.linalg.norm(self.end - self.start)
        )

        # For our purposes the direction of the normal
        # is always such that the closest point to O
        # can be found as follows:
        closest_point = origin + dist * self.normal

        return closest_point, dist


def get_walls(left, top, width, height, normal_dir):
    """
    A function that generates the walls associated to a
    rectangle defined by the corner (left, top) and of
    size (width, height).
    """
    return [
        Wall((left, top), (left + width, top), normal_dir),
        Wall((left + width, top), (left + width, top + height), normal_dir),
        Wall((left + width, top + height), (left, top + height), normal_dir),
        Wall((left, top + height), (left, top), normal_dir),
    ]


class Border(pg.Rect):
    """
    A child of the pygame Rect class. It defines the outside border
    of the environment along with its walls.
    """

    def __init__(self, surface: pg.Surface, left, top, width, height):
        pg.Rect.__init__(self, left, top, width, height)
        self.surface = surface
        self.walls = get_walls(left, top, width, height, "out")

    def draw(self):
        """
        Draws the border.
        """
        pg.draw.rect(self.surface, WHITE, self)

    def outside(self, point):
        """
        Checks if a point is outside the border.
        """
        return (
            point[0] <= self.left
            or self.right <= point[0]
            or point[1] <= self.top
            or self.bottom <= point[1]
        )

    def collision_normals_and_current_distances(self, pos, dpos, radius):
        """
        Given a circle centered at pos and a vector dpos, this function
        calculates which walls (if any) will collide with the circle if
        it moves at pos' = pos + dpos. It then returns the normals of
        said walls as well as the distances from pos to each wall.
        """
        normals_and_distances = []
        for wall in self.walls:
            # A collision with a wall W with normal n occurs if
            # pos + dpos + r*n is outside the border
            if self.outside(pos + dpos + radius * wall.normal):
                _, current_dist = wall.closest_point_and_distance_from(pos)
                normals_and_distances.append((wall.normal, current_dist))

        return normals_and_distances


class Obstacle(pg.Rect):
    """
    A child of the pygame Rect class. It defines an obstacle
    of the environment along with its walls.
    """

    def __init__(self, surface: pg.Surface, left, top, width, height):
        pg.Rect.__init__(self, left, top, width, height)
        self.surface = surface
        self.walls = get_walls(left, top, width, height, "in")
        self.corners = [wall.start for wall in self.walls]

    def draw(self):
        """
        Draws the obstacle.
        """
        pg.draw.rect(self.surface, BLACK, self)

    def inside(self, point):
        """
        Checks if a point is inside the obstacle.
        """
        return (
            self.left <= point[0] <= self.right and self.top <= point[1] <= self.bottom
        )

    def collision_normal_and_current_distance(self, pos, dpos, radius):
        """
        Given a circle centered at pos and a vector dpos, this function
        calculates which point of the obstacle (if any) will collide with the circle
        if it moves at pos' = pos + dpos. Note that since obstacles are assumed to be
        rectangular, there can be at most one point of collision (either on the inside of
        a wall or at one of the corners).
            - In case when collision is happening at a point on the inside of a wall, the function
            returns the normal of said wall as well as the distance from pos to the wall.
            - In case when collision is happening at a corner, the function calculates the normal
            of an "imaginary wall" which is tangent to the circle at the point of collision,
            and returns said normal as well as the distance from pos to the "imaginary wall".
        """
        # Collision on the inside of a wall
        for wall in self.walls:
            if self.inside(pos + dpos + radius * wall.normal):
                _, current_dist = wall.closest_point_and_distance_from(pos)
                return wall.normal, current_dist

        # Collision at a corner of the obstacle
        for corner in self.corners:
            if np.linalg.norm(corner - (pos + dpos)) <= radius:
                collision_normal = (corner - (pos + dpos)) / np.linalg.norm(
                    corner - (pos + dpos)
                )
                current_dist = np.dot(corner - pos, collision_normal)
                return collision_normal, current_dist

        return None, None


class Environment:
    """
    An environment is defined by its border and any potential obstacles
    within it. This class is used for easy creation of different environments
    for the robot to move in. The border and the obstacles of the
    environment are assumed to be rectangular.
    """

    def __init__(self, surface: pg.Surface, left, top, width, height):
        self.surface = surface
        self.border = Border(surface, left, top, width, height)
        self.obstacles: list[Obstacle] = []

    def add_obstacle(self, left, top, width, height):
        """
        Adds an obstacle to the environment.
        """
        self.obstacles.append(Obstacle(self.surface, left, top, width, height))

    def draw(self):
        """
        Draws the environment.
        """
        self.surface.fill(BLACK)
        self.border.draw()
        for obstacle in self.obstacles:
            obstacle.draw()


def make_env(env_config, draw=False):
    """
    Creates an environment from a configuration dictionary.
    """
    if draw:
        screen = pg.display.set_mode(env_config["size"])
    else:
        pg.init()
        screen = pg.Surface(env_config["size"])

    env = Environment(screen, *env_config["border"])

    for obstacle_ltwh in env_config["obstacles"]:
        env.add_obstacle(*obstacle_ltwh)

    return env
