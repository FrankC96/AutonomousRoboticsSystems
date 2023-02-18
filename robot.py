import pygame as pg
import numpy as np
from colors import *

class Robot():
   def __init__(self, pos, radius):
      # Initialise attributes of the robot
      self.pos = pos # (x, y)
      self.radius = radius
      self.motors = np.array([0.0, 0.0]) # Left wheel, Right wheel
      # The speed of motors is px/s
      self.VL = 0
      self.VR = 0
      self.accel = 1 # How fast to increment the speed of each wheel
      self.theta = np.pi/2 # Angle of the robot with the x axis in rads
      
   def update_motor(self, direction, fps):
      self.motors += np.array(direction)
      self.VL = fps*self.motors[0]
      self.VR = fps*self.motors[1]
      # print('Motors: ', self.motors)

   def move(self, fps):
      if self.motors[0] == self.motors[1]: # Forward movement
         x_component = np.cos(self.theta) # x component of direction vector
         y_component = np.sin(-self.theta) # y component of direction vector
         pos = np.array(self.pos).astype('float64')
         pos += self.motors*np.array([x_component, y_component])
         self.pos = tuple(pos)
      else: # Angular movement
         R = self.radius*((self.VL + self.VR) / (self.VR - self.VL))
         omega = (self.VR - self.VL) / (self.radius*2)
         ICC = np.array([self.pos[0] - R*np.sin(-self.theta), self.pos[1] + R*np.cos(self.theta)])
         delta_t = 1 / fps
         A = np.array([[np.cos(omega*delta_t), -np.sin(omega*delta_t), 0],
                       [np.sin(omega*delta_t), np.cos(omega*delta_t), 0],
                      [0, 0, 1]])
         b = np.array([self.pos[0] - ICC[0], self.pos[1] - ICC[1], self.theta])
         c = np.array([ICC[0], ICC[1], omega*delta_t])
         new_x, new_y, new_theta = np.matmul(A, b) + c
         self.pos = (new_x, new_y)
         self.theta = new_theta

         
   def draw(self, surf):
    inn_circles = 4 # Inner circles of the robot just for fun!

    pg.draw.circle(surf, BLACK, self.pos, self.radius)
    for cir in range(inn_circles):
       pg.draw.circle(surf, WHITE, self.pos, self.radius-cir*10, 1)
    x_component = np.cos(self.theta)*self.radius
    y_component = np.sin(-self.theta)*self.radius
    start_pos = np.array(self.pos) # Start position of line
    end_pos = start_pos + np.array([x_component, y_component]) # End position of line
    pg.draw.line(surf, WHITE, tuple(start_pos), tuple(end_pos))      

