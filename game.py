import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import sys
sys.setrecursionlimit(10000)

pygame.init()

class Direction(Enum):
    # Crate ENUM for Directions
    RIGHT   = 1
    LEFT    = 2
    UP      = 3
    DOWN    = 4
    
Position = namedtuple('Position', 'x, y')

TILE_PIXELS = 20
SNAKE_SPEED = 100

class Game:
    # Create initial variables
    x = 0
    y = 0
    i = 0
    
    # Create display and it's size
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((self.width, self.height))
        self.timer = pygame.time.Clock()
        self.reset()
        
        self.direction = Direction.RIGHT
        # Set snake's position
        self.front = Position(40,40)
        self.body = [self.front, 
                      Position(self.front.x-TILE_PIXELS, self.front.y),
                      Position(self.front.x-(2*TILE_PIXELS), self.front.y)]
        
        self.currentScore = 0
        self.food = None
        self.placeNewFood()
        self.i = 0

    # Function used to restart game after each iteraction
    def reset(self):

        self.direction = Direction.RIGHT
        
        self.front = Position(40,40)
        self.body = [self.front, 
                      Position(self.front.x-TILE_PIXELS, self.front.y),
                      Position(self.front.x-(2*TILE_PIXELS), self.front.y)]
        
        self.currentScore = 0
        self.food = None
        self.placeNewFood()
        self.iterationFrame = 0
        self.i = 0
        
    # Function used to create new food on map
    def placeNewFood(self):

        # Create initial food positions (x)
        self.food_x = [20, 60,  20, 80, 100, 120, 200, 260, 300, 320, 80, 240, 380]
        self.x = self.food_x[self.i]

        self.y =self.food_x[self.i]

        self.food = Position(self.x, self.y)
        if self.food in self.body:
            if (self.i == 12):
                self.i = 0
            else:
                self.i += 1
            self.placeNewFood()
        
    def play(self, action):
        # Get signal, if event is quit, quit the game.
        self.iterationFrame += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            
        
        # Move, update display
        self._move(action)
        self.body.insert(0, self.front)
        
        # Check if snake has hit the wall, if it hits the wall, add score and move food to different position, also increase it's size. 
        reward = 0
        gameOver = False
        # Additional check to prevent infinite loop, checks if snake's moves is bigger than his length * 100, then resets game
        # gradually time to find food is getting bigger as the snake is getting bigger
        if self.isColliding() or self.iterationFrame > 100*len(self.body): 
            gameOver = True
            reward = -10
            return reward, gameOver, self.currentScore
            
        if self.front == self.food:
            if (self.i == 12):
                self.i = 0
            else:
                self.i += 1
            self.currentScore += 1
            reward = 10
            self.placeNewFood()
        else:
            self.body.pop()
        
        self.updateUi()
        self.timer.tick(SNAKE_SPEED)
        return reward, gameOver, self.currentScore
    
    def isColliding(self, pt=None):
        if pt is None:
            pt = self.front
        # Check if snake is out of map
        if pt.x > self.width - TILE_PIXELS or pt.x < 0 or pt.y > self.height - TILE_PIXELS or pt.y < 0:
            return True
        # Check if snake has hit itself
        if pt in self.body[1:]:
            return True
        
        return False
        
    # Function used to draw map, food and snake
    def updateUi(self):
        self.display.fill((66, 77, 89))
        
        # draw snake
        for pt in self.body:
            pygame.draw.rect(self.display, (230, 55, 72), pygame.Rect(pt.x, pt.y, TILE_PIXELS, TILE_PIXELS))
            pygame.draw.rect(self.display, (230, 55 , 72), pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        # draw food
        pygame.draw.rect(self.display, (12,200,31), pygame.Rect(self.food.x, self.food.y, TILE_PIXELS, TILE_PIXELS))
        
        pygame.display.flip()
        
    def _move(self, action):
        # [1, 0, 0] - stay in current position
        # [0, 1, 0] - move right
        # [0, 0, 1] - move left

        # This is to block moving opposite direction due to game's rules, also to narrow prediction to 3 positions

        timerWise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = timerWise.index(self.direction)

        if np.array_equal(action, [1,0,0]):
            newDir = timerWise[idx] # stay in current position
        elif np.array_equal(action, [0,1,0]):
            nextIndex = (idx + 1) % 4
            newDir = timerWise[nextIndex] # move right, down, left and up
        else: # [0,0,1]
            nextIndex = (idx - 1) % 4
            newDir = timerWise[nextIndex] # ruch odwrotny do wskazówek zegara

        self.direction = newDir

        x = self.front.x
        y = self.front.y
        if self.direction == Direction.RIGHT:
            x += TILE_PIXELS
        elif  self.direction == Direction.LEFT:
            x -= TILE_PIXELS
        elif  self.direction == Direction.DOWN:
            y += TILE_PIXELS
        elif  self.direction == Direction.UP:
            y -= TILE_PIXELS
            
        self.front = Position(x, y)