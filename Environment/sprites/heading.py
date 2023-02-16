import pygame


class R_heading(pygame.sprite.Sprite):
    def __init__(self, color, width, height, block_width):
        # Call the parent class (Sprite) constructor
        super().__init__()

        # Pass in the color of the player, and its x and y position, width and height.
        # Set the background color and set it to be transparent
        self.image = pygame.Surface([width, height], pygame.SRCALPHA)
        # self.image.fill(color)

        # Initialise attributes of the car.
        self.width = width
        self.height = height
        self.color = color
        self.rect = self.image.get_rect()

        # Draw the player
        pygame.draw.line(self.image, self.color, (width/2 - self.rect.x, height/2 - self.rect.y), (width/2 - self.rect.x, self.rect.y + 25), 5)





