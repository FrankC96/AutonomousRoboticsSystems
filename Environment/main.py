import pygame
from sprites.robot import Robot
from sprites.heading import R_heading
pygame.init()


bounds = 600
screen = pygame.display.set_mode([bounds, bounds])
pygame.display.set_caption('Stupid thing hopefully colliding in walls')

# Colors
red =   (255, 0  ,   0)
green = (0  , 255,   0)
blue =  (122  , 146  , 226)
white = (255, 255, 255)
black = (0  ,   0,   0)

# Sprite attributes
block_width = 30
r1_size = 100
background = blue
framerate = 60

timer = pygame.time.Clock()

group = pygame.sprite.Group()
r1 = Robot(black, r1_size, r1_size)
r1_heading = R_heading(black, r1_size, r1_size, block_width)

r1.rect.x = 300 - r1_size/2
r1.rect.y = 300 - r1_size/2

r1_heading.rect.x = r1.rect.x
r1_heading.rect.y = r1.rect.y

group.add(r1, r1_heading)

running = True
while running:
    timer.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((background))
    # pygame.draw.circle(screen, (255, 255, 255), (300, 300), 25, 5)
    for i in range(30):  # 600 px divided by 20 blocks
        pygame.draw.rect(screen, black, (0 + i*20, 0, block_width, block_width))
        pygame.draw.rect(screen, black, (0, 0 + i*20, block_width, block_width))

        pygame.draw.rect(screen, black, (bounds-block_width, 0 + i * 20, block_width, block_width))
        pygame.draw.rect(screen, black, (0 + i * 20, bounds-block_width, block_width, block_width))

        # pygame.draw.line(screen, black, (60, 80), (130, 100))
    # pygame.draw.line(screen, black, (300, 300), (300, 300-block_width), 5)
    group.update()
    group.draw(screen)


    pygame.display.flip()

pygame.quit()