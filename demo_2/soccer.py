import pygame, sys
import pygame.locals
import numpy as np

# ---------------------- define constants
# pygame
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# text
DISPLAY_SIZE = 80
INFO_SIZE = 35
SCORE_SIZE = 30
TITLE_SIZE = 160

# colors
GREEN = (0, 170, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
GOLD = (190, 190, 0)
BLACK = (0, 0, 0)
YELLOW = (210, 210, 0)
TRANSPARENT_BLACK = (0, 0, 0, 185)
TRANSPARENT_YELLOW = (210, 210, 0, 185)

# game constants
# field
FIELD_WIDTH = 600
FIELD_HEIGHT = 400
X_GAP = (SCREEN_WIDTH-FIELD_WIDTH)/2
Y_GAP = (SCREEN_HEIGHT-FIELD_HEIGHT)/2
GOAL_DEPTH = 45
GOAL_HEIGHT = 100
GOAL_TOP = Y_GAP+FIELD_HEIGHT/2-GOAL_HEIGHT/2
GOAL_BOTTOM = GOAL_TOP+GOAL_HEIGHT
LEFT_GOAL_BACK = X_GAP-GOAL_DEPTH
RIGHT_GOAL_BACK = SCREEN_WIDTH-X_GAP+GOAL_DEPTH
# ball
BALL_MASS = 8
BALL_SIZE = 10
# player
PLAYER_MASS = 30
PLAYER_SIZE = 20
# grenade powerup
GRENADE_SIZE = 15 # during powerup select
FRAG_COUNT = 16
FRAG_MASS = 40
FRAG_SIZE = 3
FRAG_VEL = FPS * 0.25
FRAG_LIFETIME = 120 # milliseconds
# glue powerup
GLUE_SIZE = 50
GLUE_FRICTION = 0.9
GLUE_LIFE = 3 # rounds - is removed the moment the 3rd round begins (player moves)
# physics
MAX_VEL = FPS*0.2
AIM_TWEAK = 10 # smaller number = less difference bt big aim and small aim
FRICTION = 0.97
FRICTION_COEFFICIENT = 0.15 # for collisions
RESTITUTION = 0.8 # bounciness
# buttons
NUM_BUTTONS = 2
ICON_SIZE = 64
BUTTON_GAP = Y_GAP/2
BUTTON_Y = SCREEN_HEIGHT-Y_GAP/2-ICON_SIZE/2
BUTTON_RECT = pygame.Rect(0, 0, ICON_SIZE, ICON_SIZE) # for powerups, drawn on a different surface
# misc
SELECTED_THICKNESS = 5
SPAWNS = ((FIELD_WIDTH/5, FIELD_HEIGHT/3), (FIELD_WIDTH/5,FIELD_HEIGHT*2/3), (FIELD_WIDTH/3, FIELD_HEIGHT/2))
WIN_SCORE = 3

# strings
GRENADE = "Grenade"
GLUE = "Glue"


# ---------------------- define classes
class PhysicalObject:
    def __init__(self, x: int, y: int, mass: int, size: int, color: tuple, type=""):
        self.x = x
        self.y = y
        self.mass = mass
        self.size = size # radius of circle
        self.color = color
        self.type = type

        self.v = np.zeros(2, dtype=np.float64)
        self.moving = np.linalg.norm(self.v) > 0.001

    def draw(self, surf: pygame.Surface):
        pygame.draw.circle(surf, self.color, (self.x, self.y), self.size)

    def updatePos(self, glues=[]):
        self.x += self.v[0]
        self.y += self.v[1]

        friction = FRICTION
        for glue in glues:
            if distance(glue.x, glue.y, self.x,self.y) <= GLUE_SIZE+PLAYER_SIZE: # player is in glue
                friction = GLUE_FRICTION # so they are frictioned more
                break
        self.v *= friction

        self.moving = np.linalg.norm(self.v) > 0.001

    # does NOT check for collision, only handles it
    def handleCollision(self, other): # THANKS ALEX
        # Calculate the vector between the objects
        delta = np.array([self.x - other.x, self.y - other.y])
        distance = np.linalg.norm(delta)
        
        # Calculate overlap
        overlap = self.size + other.size - distance
        
        if overlap > 0:
            # Normalize the delta vector
            normal = delta / distance
            
            # Separate the objects
            separation = overlap * normal * 0.5
            self.x += separation[0]
            self.y += separation[1]
            other.x -= separation[0]
            other.y -= separation[1]
            
            # Calculate relative velocity
            relative_velocity = self.v - other.v
            
            # Calculate velocity along the normal
            velocity_along_normal = np.dot(relative_velocity, normal)
            
            # Do not resolve if velocities are separating
            if velocity_along_normal > 0:
                return
            
            # Calculate impulse scalar
            impulse_scalar = -(1 + RESTITUTION) * velocity_along_normal
            impulse_scalar /= 1/self.mass + 1/other.mass
            
            # Apply impulse
            impulse = impulse_scalar * normal
            self.v += impulse / self.mass
            other.v -= impulse / other.mass
            
            # Apply friction
            tangent = np.array([-normal[1], normal[0]])
            friction_impulse_scalar = np.dot(relative_velocity, tangent) * FRICTION_COEFFICIENT
            friction_impulse_scalar /= 1/self.mass + 1/other.mass
            
            # Ensure friction doesn't reverse velocity
            if friction_impulse_scalar < 0:
                friction_impulse = friction_impulse_scalar * tangent
            else:
                friction_impulse = -friction_impulse_scalar * tangent
            
            self.v += friction_impulse / self.mass
            other.v -= friction_impulse / other.mass
            
            # Limit velocities to MAX_VEL
            self.v = np.clip(self.v, -MAX_VEL, MAX_VEL)
            other.v = np.clip(other.v, -MAX_VEL, MAX_VEL)
    
    # detects and handles collision with the wall
    def handleWallCollision(self):
        # field walls
        if (self.y-self.size < Y_GAP): # top
            self.v[1] = -self.v[1]
            self.y = Y_GAP + self.size
        if (self.y+self.size > Y_GAP+FIELD_HEIGHT): # bottom
            self.v[1] = -self.v[1]
            self.y = Y_GAP + FIELD_HEIGHT - self.size
        # take into account the goal for left/right
        if (self.y-self.size < GOAL_TOP or self.y+self.size > GOAL_BOTTOM):
            if (self.x-self.size < X_GAP): # left
                self.v[0] = -self.v[0] # reflect velocity horizontally
                self.x = X_GAP + self.size # move it out of the wall
            if (self.x+self.size > X_GAP+FIELD_WIDTH): # right
                self.v[0] = -self.v[0]
                self.x = X_GAP + FIELD_WIDTH - self.size
        
        # goal walls
        if (self.x-self.size < LEFT_GOAL_BACK): # left goal back
            self.v[0] = -self.v[0]
            self.x = LEFT_GOAL_BACK + self.size
        if (self.x+self.size > RIGHT_GOAL_BACK): # right goal back
            self.v[0] = -self.v[0]
            self.x = RIGHT_GOAL_BACK - self.size
        # take into account object has to be inside goal
        if (self.x < X_GAP or self.x > X_GAP + FIELD_WIDTH):
            if (self.y-self.size < GOAL_TOP): # goal top
                self.v[1] = -self.v[1]
                self.y = GOAL_TOP + self.size
            if (self.y+self.size > GOAL_BOTTOM): # goal bottom
                self.v[1] = -self.v[1]
                self.y = GOAL_BOTTOM - self.size

class Player(PhysicalObject):
    hovered = False
    def __init__(self, x, y, color):
        super().__init__(x, y, PLAYER_MASS, PLAYER_SIZE, color)

    def draw(self, surf):
        super().draw(surf)
        if self.hovered:
            pygame.draw.circle(surf, WHITE, (self.x, self.y), self.size, width=SELECTED_THICKNESS)

class Fragment(PhysicalObject):
    def __init__(self, x: int, y: int, mass: int, size: int, color: tuple):
        super().__init__(x, y, mass, size, color, "frag")
        self.spawnTime = pygame.time.get_ticks()
        
class FieldObject:
    def __init__ (self, x: int, y: int, size: int, color: tuple, lifetime=-1):
        self.x = x
        self.y = y
        self.size = size
        self.color = color
        self.lifetime = lifetime
    
    def draw(self, surf: pygame.Surface):
        pygame.draw.circle(surf, self.color, (self.x, self.y), self.size)

class MenuButton:
    def __init__(self, color: tuple, center: tuple, img: str):
        self.color = color
        self.center = center
        self.img = pygame.image.load(img)
        self.rect = pygame.Rect(self.center[0]-ICON_SIZE/2, self.center[1]-ICON_SIZE/2, ICON_SIZE, ICON_SIZE)

        self.hovered = False
    
    def draw(self, surf: pygame.Surface):
        pygame.draw.circle(surf, self.color, self.center, ICON_SIZE/2)
        surf.blit(self.img, self.rect)

        if self.hovered:
            pygame.draw.circle(surf, WHITE, self.center, ICON_SIZE/2, width = SELECTED_THICKNESS)

class Button:
    def __init__(self, color: tuple, rect: pygame.Rect):
        self.color = color
        self.rect = rect

        self.hovered = False
        self.selected = False

    def draw(self, surf: pygame.Surface, drawRect=None):
        if drawRect is None: # special rect needed for powerup uttons so as to have transparent surface
            drawRect = self.rect
        pygame.draw.rect(surf, self.color, drawRect, border_radius=8)
        # if self.img:
        #     surf.blit(self.img, drawRect)

        if self.hovered:
            pygame.draw.rect(surf, WHITE, drawRect, width=SELECTED_THICKNESS, border_radius=8)
        if self.selected:
            pygame.draw.rect(surf, GOLD, drawRect, width=SELECTED_THICKNESS, border_radius=8)

class PowerupButton(Button):
    def __init__(self, rect: pygame.Rect, name: str, img: str):
        self.name = name
        self.img = pygame.image.load(img)
        assert self.img.get_width() == ICON_SIZE and self.img.get_height() == ICON_SIZE, "Image is wrong size for powerup"

        super().__init__(BLUE, rect)
   
    def draw(self, surf: pygame.Surface, powerupAvailable=True):
        buttonSurf = pygame.Surface((ICON_SIZE, ICON_SIZE), pygame.SRCALPHA)
        buttonSurf.fill((0,0,0,0))
        if not powerupAvailable: # if powerup isn't available, "grey out" the icon
            buttonSurf.set_alpha(170)

        super().draw(buttonSurf, drawRect=BUTTON_RECT) # draw button rectangle
        buttonSurf.blit(self.img, BUTTON_RECT)
        surf.blit(buttonSurf, (self.rect.left, self.rect.top))

class TextButton(Button):
    def __init__(self, bgColor: tuple, text: str, font: pygame.font, textColor: tuple):
        self.text = font.render(text, True, textColor)
        self.rect = self.text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))

        super().__init__(bgColor, self.rect)
    
    def draw(self, surf: pygame.Surface):
        super().draw(surf, self.rect)
        surf.blit(self.text, self.rect)

#  ---------------------- define functions
def distance(x1: int, y1: int, x2: int, y2: int):
    return np.sqrt( (x1-x2)**2 + (y1-y2)**2 )

def angle(x1: int, y1: int, x2: int, y2: int):
    return np.atan2( (y2-y1), (x2-x1) )

def vectorToXY(magnitude: int, direction: float):
    x = np.cos(direction)*magnitude
    y = np.sin(direction)*magnitude
    return x, y

def inField(x: int, y: int):
    if (x > X_GAP and x < X_GAP+FIELD_WIDTH) and (y > Y_GAP and y < Y_GAP+FIELD_HEIGHT):
        return True # main part of field
    if (x > LEFT_GOAL_BACK and x < RIGHT_GOAL_BACK) and (y > GOAL_TOP and y < GOAL_BOTTOM):
        return True # goals
    return False

def spawnGrenade(objects: list, x: int, y: int):
    for i in range(0, FRAG_COUNT):
        frag = Fragment(x, y, FRAG_MASS, FRAG_SIZE, BLACK)
        vel = np.array(vectorToXY(FRAG_VEL, np.pi*i/(FRAG_COUNT/2)))
        frag.v = vel
        
        objects.append(frag)

def infoDisplay(DISPLAYSURF: pygame.Surface, font: pygame.font, clock: pygame.time.Clock, info: Button):
    # display info until user exits back to menu
    while 1:
        mouseX, mouseY = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.locals.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.locals.MOUSEBUTTONUP:
                if info.hovered: # info button is back button
                    return
        info.hovered = distance(info.center[0], info.center[1], mouseX, mouseY) <= ICON_SIZE/2        
        
        infoText = '''
        Welcome to Soccer!\nDrag and release pieces to launch them.\nOne powerup per turn.\n
        Grenade sets off an explosive,\nlaunching nearby objects\nGlue makes an area sticky for a round.\n
        Try to get the ball in your opponent's goal.\nFirst to 3 goals wins.\nGood luck!
        '''
        infoLines = infoText.split("\n")

        DISPLAYSURF.fill(GREEN)
        for i in range(len(infoLines)): # display each line of text in its own line
            text = font.render(infoLines[i], True, WHITE)
            textRect = text.get_rect(midtop = (SCREEN_WIDTH/2, INFO_SIZE/2+INFO_SIZE*i))
            DISPLAYSURF.blit(text, textRect)
        info.draw(DISPLAYSURF)

        pygame.display.update()
        clock.tick(FPS)

def gameLoop(DISPLAYSURF: pygame.Surface):
    pass

def main():
    # initialize pygame
    pygame.init()
    DISPLAYSURF = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    DISPLAYSURF.fill(GREEN)
    pygame.display.set_caption("Soccer")
    clock = pygame.time.Clock()
    pygame.font.init()
    displayFont = pygame.font.SysFont("krungthep", DISPLAY_SIZE)
    infoFont = pygame.font.SysFont("kefa", INFO_SIZE)
    scoreFont = pygame.font.SysFont("menlo", SCORE_SIZE)
    titleFont = pygame.font.SysFont("bradleyhand", TITLE_SIZE)

    titleText = titleFont.render("Soccer", True, WHITE)
    titleRect = titleText.get_rect(midtop=(SCREEN_WIDTH/2,0))
    playButton = TextButton(GOLD, "Play", displayFont, WHITE)

    infoButton = MenuButton(GOLD, (SCREEN_WIDTH-ICON_SIZE, SCREEN_HEIGHT-ICON_SIZE), "buttons/info.png")

    # initialize game
    objects = []

    selected = None
    startingX, startingY = 0,0

    turn = BLUE
    nothingMoving = True

    buttonsX = []
    for i in range(NUM_BUTTONS):
        buttonsX.append(BUTTON_GAP * (1+i) + ICON_SIZE*i)
    grenadeButton = PowerupButton(pygame.Rect(buttonsX[0], BUTTON_Y, ICON_SIZE, ICON_SIZE), GRENADE, "buttons/grenade.png")
    glueButton = PowerupButton(pygame.Rect(buttonsX[1], BUTTON_Y, ICON_SIZE, ICON_SIZE), GLUE, "buttons/glue.png")
    buttons = [grenadeButton, glueButton]
    selectedButton, selectedButtonObj = None, None # first is for game loop, second is to set the button instance variable's selected = False once powerup is used
    powerup = True
    while 1:
        gameLoop = False

        mouseX, mouseY = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.locals.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.locals.MOUSEBUTTONUP:
                if playButton.hovered:
                    gameLoop = True
                if infoButton.hovered:
                    infoDisplay(DISPLAYSURF, infoFont, clock, infoButton)
        
        playButton.hovered = playButton.rect.collidepoint(mouseX, mouseY)
        infoButton.hovered = distance(infoButton.center[0], infoButton.center[1], mouseX, mouseY) <= ICON_SIZE/2

        DISPLAYSURF.fill(GREEN)
        DISPLAYSURF.blit(titleText, titleRect)
        playButton.draw(DISPLAYSURF)
        infoButton.draw(DISPLAYSURF)
        pygame.display.update()
        clock.tick(FPS)

        # reset game variables -----------
        glues = []

        scored = True # start at True to set inital object positions   
        blueScore = 0
        redScore = 0
        win = False
        while gameLoop:
            # handle scored -----------------------
            # let it run until everything stops moving, then reset
            if scored:
                stoppedMoving = True
                for obj in objects:
                    if obj.moving:
                        stoppedMoving = False
                        break
                
                if stoppedMoving and not win:
                    ball = PhysicalObject(SCREEN_WIDTH/2, SCREEN_HEIGHT/2, BALL_MASS, BALL_SIZE, WHITE)
                    objects = [ball]
                    for spawn in SPAWNS:
                        objects.append(Player(X_GAP + spawn[0], Y_GAP + spawn[1], BLUE))
                        objects.append(Player(SCREEN_WIDTH - X_GAP - spawn[0], Y_GAP + spawn[1], RED))

                    players = objects[1:]
                    
                    scored = False
                    continue

            # handle input -----------------------
            # hold click & drag to aim
            mouseX, mouseY = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.locals.QUIT:
                    pygame.quit()
                    sys.exit()
                            
                if event.type == pygame.locals.MOUSEBUTTONDOWN and nothingMoving:
                    # on click, check if anything's selected
                    # if not, check the cursor is on any player to mark it as selected
                    if selected is None:
                        for player in players:
                            if player.color == turn and distance(mouseX, mouseY, player.x, player.y) <= PLAYER_SIZE:
                                # distance from any player is within the player size = mouse is on the circle
                                startingX, startingY = mouseX, mouseY
                                player.hovered = True # mark to draw the circle around it
                                selected = player
                                break
                    
                    # check for button click
                    if powerup:
                        for button in buttons:
                            if button.rect.collidepoint(mouseX, mouseY):
                                if button.hovered: # click on hovered button = select/unselect the button
                                    if button.selected:
                                        button.selected = False
                                        selectedButton, selectedButtonObj = None, None
                                    elif selectedButton is None:
                                        button.selected = True
                                        selectedButton = button.name
                                        selectedButtonObj = button
                
                if event.type == pygame.locals.MOUSEBUTTONUP:
                    # on unclick, check if anything's selected
                    # if so, check if the cursor's outside the player
                    if powerup: # unnecessary (but just an extra check), as button cannot be clicked if powerup is False
                        if selectedButton == GRENADE:
                            if inField(mouseX, mouseY):
                                spawnGrenade(objects, mouseX, mouseY)
                                powerup = False
                                selectedButtonObj.selected = False
                                selectedButton, selectedButtonObj = None, None
                        if selectedButton == GLUE:
                            if inField(mouseX, mouseY):
                                glues.append(FieldObject(mouseX, mouseY, GLUE_SIZE, YELLOW, lifetime = GLUE_LIFE))
                                powerup = False
                                selectedButtonObj.selected = False
                                selectedButton, selectedButtonObj = None, None
                    
                    # only handle player stuff if a powerup isn't selected
                    if (selectedButton is None) and (selected) and (distance(mouseX, mouseY, selected.x, selected.y) > PLAYER_SIZE):
                        # calculate drag distance, applying a slight tweak
                        vel = pygame.math.Vector2(-(mouseX-startingX)/AIM_TWEAK, -(mouseY-startingY)/AIM_TWEAK)
                        
                        # limit velocity to MAX_VEL
                        if vel.magnitude_squared() > MAX_VEL**2: # can use vel.clamp_magnitude_ip(), but is experimental
                            vel.scale_to_length(MAX_VEL)
                        
                        selected.v = vel

                        player.hovered = False
                        selected = None

                        # swap turns, new round
                        if turn == RED:
                            turn = BLUE
                        else:
                            turn = RED
                        powerup = True                    
                        for glue in glues:
                            glue.lifetime -=1
                            if glue.lifetime == 0:
                                glues.remove(glue)

                    # if not, unselect the thing
                    else:
                        player.hovered = False
                        selected = None
            
            # update game -----------------------
            fragsToRemove = []
            nothingMoving = True
            # check for wall collision
            for obj in objects:
                if obj.moving:
                    nothingMoving = False
                obj.handleWallCollision()
                obj.updatePos(glues)

                if (obj.type == "frag"):
                    #if (not obj.moving):
                    if (pygame.time.get_ticks()-obj.spawnTime > FRAG_LIFETIME):
                        fragsToRemove.append(obj)
            
            for frag in fragsToRemove:
                objects.remove(frag)
            
            pairs = [(a, b) for i, a in enumerate(objects) for b in objects[i+1:]]
            for i in range(2):
                for obj1, obj2 in pairs:
                    if distance(obj1.x, obj1.y, obj2.x, obj2.y) <= obj1.size+obj2.size:
                        obj1.handleCollision(obj2)
            
            # display ----------------------------
            DISPLAYSURF.fill(GREEN)
            pygame.draw.rect(DISPLAYSURF, WHITE, pygame.Rect(X_GAP, Y_GAP, FIELD_WIDTH, FIELD_HEIGHT), 1) # field lines
            pygame.draw.rect(DISPLAYSURF, BLUE, pygame.Rect(LEFT_GOAL_BACK, GOAL_TOP, GOAL_DEPTH, GOAL_HEIGHT), 1) # left goal
            pygame.draw.rect(DISPLAYSURF, RED, pygame.Rect(X_GAP+FIELD_WIDTH, GOAL_TOP, GOAL_DEPTH, GOAL_HEIGHT), 1) # right goal
            pygame.draw.line(DISPLAYSURF, GOLD, (X_GAP, GOAL_TOP), (X_GAP, GOAL_BOTTOM), 4) # left goal line
            pygame.draw.line(DISPLAYSURF, GOLD, (X_GAP+FIELD_WIDTH, GOAL_TOP), (X_GAP+FIELD_WIDTH, GOAL_BOTTOM), 4) # right goal line
            
            # draw objects ----------------
            for glue in glues:
                glue.draw(DISPLAYSURF)
            for obj in objects:
                obj.draw(DISPLAYSURF)
            if selected:
                pygame.draw.line(DISPLAYSURF, WHITE, (selected.x, selected.y), (mouseX, mouseY), SELECTED_THICKNESS)
            
            # draw buttons ----------------
            for button in buttons:
                button.color = turn
                button.hovered = button.rect.collidepoint(mouseX, mouseY)
                button.draw(DISPLAYSURF, powerupAvailable=powerup)
                # DISPLAYSURF.blit(buttonSurf, (button.rect.left, button.rect.top))
            
            if selectedButton is not None:
                buttonText = scoreFont.render(selectedButton, True, turn)
                buttonTextRect = buttonText.get_rect(midbottom=(SCREEN_WIDTH/2, Y_GAP))
                DISPLAYSURF.blit(buttonText, buttonTextRect)
            if selectedButton == GRENADE:
                alphaSurf = pygame.Surface((GRENADE_SIZE*2, GRENADE_SIZE*2), pygame.SRCALPHA) # new surface to draw the transparency
                pygame.draw.circle(alphaSurf, TRANSPARENT_BLACK, (GRENADE_SIZE, GRENADE_SIZE), GRENADE_SIZE)
                DISPLAYSURF.blit(alphaSurf, (mouseX-GRENADE_SIZE, mouseY-GRENADE_SIZE))
            if selectedButton == GLUE:
                alphaSurf = pygame.Surface((GLUE_SIZE*2, GLUE_SIZE*2), pygame.SRCALPHA)
                pygame.draw.circle(alphaSurf, TRANSPARENT_YELLOW, (GLUE_SIZE, GLUE_SIZE), GLUE_SIZE)
                DISPLAYSURF.blit(alphaSurf, (mouseX-GLUE_SIZE, mouseY-GLUE_SIZE))
            
            # show turn ----------------
            # blinks for a moment, fix:
            # nothingMoving is true the frame when a player moves, even as the turn switches, so the new turn flashes before nothing shows bc there's movement
            if nothingMoving and not scored:
                if turn == BLUE:
                    turnText = scoreFont.render("Blue Turn", True, BLUE)
                if turn == RED:
                    turnText = scoreFont.render("Red Turn", True, RED)
                turnTextRect = turnText.get_rect(midtop=(SCREEN_WIDTH/2, 0))
                DISPLAYSURF.blit(turnText, turnTextRect)
                
            # detect for scoring ----------------
            if (ball.x < X_GAP and not scored):
                redScore += 1
                scored = True
                if (redScore >= WIN_SCORE):
                    displayText = displayFont.render("RED WINS", True, RED)
                    win = True
                else:
                    displayText = displayFont.render("RED SCORE", True, RED)
                displayTextRect = displayText.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
                turn = BLUE
                powerup = False
                scoreTime = pygame.time.get_ticks()
            if (ball.x > X_GAP+FIELD_WIDTH and not scored):
                blueScore += 1
                scored = True
                if (blueScore >= WIN_SCORE):
                    displayText = displayFont.render("BLUE WINS", True, BLUE)
                    win = True
                else:
                    displayText = displayFont.render("BLUE SCORE", True, BLUE)
                displayTextRect = displayText.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
                turn = RED
                powerup = False
                scoreTime = pygame.time.get_ticks()
            
            if scored and pygame.time.get_ticks() - scoreTime < 5000: # keep SCORED text on 5 seconds after score
                DISPLAYSURF.blit(displayText, displayTextRect)
            if win and pygame.time.get_ticks() - scoreTime > 5000:
                gameLoop = False # after win, leave after 5 seconds

            # show score for blue & red ----------------
            DISPLAYSURF.blit(scoreFont.render("Blue score: " + str(blueScore), True, BLUE), (0,0))
            redScoreText = scoreFont.render("Red score: " + str(redScore), True, RED)
            redScoreRect = redScoreText.get_rect(topright = (SCREEN_WIDTH, 0))
            DISPLAYSURF.blit(redScoreText, redScoreRect)

            # update window
            pygame.display.update()
            clock.tick(FPS)

if __name__ == "__main__":
    main()