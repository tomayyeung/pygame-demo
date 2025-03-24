import pygame, sys

WIDTH = 1000
HEIGHT = 800

def main():
    pygame.init()
    DISPLAYSURF = pygame.display.set_mode([WIDTH, HEIGHT])

    while 1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                return

if __name__ == "__main__":
    main()