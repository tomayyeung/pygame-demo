import pygame, sys

WIDTH = 1000
HEIGHT = 800
FPS = 60
BG_COLOR = (255, 128, 128)
WHITE = (255, 255, 255)
CIRCLE_SIZE = 50

def main():
    pygame.init()
    DISPLAYSURF = pygame.display.set_mode([WIDTH, HEIGHT])
    clock = pygame.time.Clock()

    center = [300, 300]

    loop = True
    while loop:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                loop = False
            
    
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            center[0] -= 5
        if keys[pygame.K_RIGHT]:
            center[0] += 5        
        if keys[pygame.K_UP]:
            center[1] -= 5
        if keys[pygame.K_DOWN]:
            center[1] += 5
        
        DISPLAYSURF.fill(BG_COLOR)

        pygame.draw.circle(DISPLAYSURF, WHITE, center, CIRCLE_SIZE)

        pygame.display.update()
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()