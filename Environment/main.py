import pygame
pygame.init()

from Ball_position import circle

screen = pygame.display.set_mode([600, 600])
pygame.display.set_caption('Stupid thing hopefully colliding in walls')

red =   (255, 0  ,   0)
green = (0  , 255,   0)
blue =  (0  , 0  , 255)
white = (255, 255, 255)

background = blue
framerate = 60.0
timer = pygame.time.Clock()

running = True
while running:
    timer.tick(60)
    circle = circle()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((background))
    pygame.draw.circle(screen, white, (circle.circle_x), 30, 5)

    pygame.display.flip()

pygame.quit()