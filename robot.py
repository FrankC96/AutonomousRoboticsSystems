import pygame as pg
import numpy as np
from colors import *

# Write better the movement :)
class Robot():
   def __init__(self, pos, radius):
      # Initialise attributes of the robot
      self.pos = pos # (x, y)
      self.radius = radius
      self.motors = np.array([0.0, 0.0]) # Left wheel, Right wheel
      self.accel = 0.05 # How fast to increment the speed of each wheel
      self.line_direction = np.pi # Direction of the robot in pi

   def update_motor(self, direction):
      self.motors += np.array(direction)
      #print(self.motors)

   def move(self):
      pos = np.array(self.pos).astype('float64')
      tx = round(np.cos(self.line_direction)*self.radius, 2) # Trigonometry x
      ty = round(np.sin(self.line_direction)*self.radius, 2) # Trigonometry y
      pos += self.motors*np.array([tx, ty])
      self.pos = tuple(pos)
      print (self.pos)


   def draw(self, surf):
    inn_circles = 4 # Inner circles of the robot just for fun!

    pg.draw.circle(surf, BLACK, self.pos, self.radius)
    for cir in range(inn_circles):
       pg.draw.circle(surf, WHITE, self.pos, self.radius-cir*10, 1)
    tx = round(np.cos(self.line_direction)*self.radius, 2) # Trigonometry x
    ty = round(np.sin(self.line_direction)*self.radius, 2) # Trigonometry y
    start_pos = np.array(self.pos) # Start position of line
    end_pos = start_pos + np.array([tx, ty]) # End position of line
    pg.draw.line(surf, WHITE, tuple(start_pos), tuple(end_pos))      

