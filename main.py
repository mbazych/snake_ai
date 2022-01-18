import torch
import random
import numpy as np
from collections import deque
from game import Game, Direction, Position
from model import Linear_QNet, QTrainer
from plotter import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
# learning tempo
LR = 0.01

class IntelligentGameAgent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3) # 11 neuroons on input, 256 in hidden layer and 3 in output [1, 0, 0]]
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def getState(self, game):
        front = game.body[0]
        # create 4 points around snake's head in order to check where is obstacle
        positionLeft = Position(front.x - 20, front.y)
        positionRight = Position(front.x + 20, front.y)
        positionUp = Position(front.x, front.y - 20)
        positionDown = Position(front.x, front.y + 20)
        
        directionLeft = game.direction == Direction.LEFT
        directionRight = game.direction == Direction.RIGHT
        directionUp = game.direction == Direction.UP
        directionDown = game.direction == Direction.DOWN

        # Discover treat in those points
        state = [            
            # Threat in current direction
            (directionRight and game.isColliding(positionRight)) or 
            (directionLeft and game.isColliding(positionLeft)) or 
            (directionUp and game.isColliding(positionUp)) or 
            (directionDown and game.isColliding(positionDown)),

            # After turning right
            (directionUp and game.isColliding(positionRight)) or 
            (directionDown and game.isColliding(positionLeft)) or 
            (directionLeft and game.isColliding(positionUp)) or 
            (directionRight and game.isColliding(positionDown)),

            # After turning left
            (directionDown and game.isColliding(positionRight)) or 
            (directionUp and game.isColliding(positionLeft)) or 
            (directionRight and game.isColliding(positionUp)) or 
            (directionLeft and game.isColliding(positionDown)),
            
            # Direction which snake moves
            directionLeft,
            directionRight,
            directionUp,
            directionDown,
            
            # Localization of food
            game.food.x < game.front.x,  # left
            game.food.x > game.front.x,  # right
            game.food.y < game.front.y,  # up
            game.food.y > game.front.y  # down
            ]

        return np.array(state, dtype=int) # convert to int

    # function used to store tuples, if excveeds maximum memory throw from left to right
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # learn based on all movements from previous iteration and choose ones that won the best prize
    def trainLongMemory(self):

        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # tuples list with size of memory
        else:
            mini_sample = self.memory # sample equal to memory size if its lower than 1000

        states, actions, rewards, next_states, dones = zip(*mini_sample) # function used to iterate over all variables from try and assign them to tables
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    # function used to learn single movements
    def trainShortMemory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    # a function that initially performs random movements and as the model learns the board and where to eat, start using your knowledge
    def getAction(self, state):
        self.epsilon = 80 - self.n_games
        finalMove = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            finalMove[move] = 1
        else: # after completing the learning stage, around 80 iterations, the model begins to use predictions
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item() # zwracane zostaje największe przewidywanie np [8, 2, 3] co oznacza ruch w obecnym kierunku bez zmian
            finalMove[move] = 1

        return finalMove


def train():
    plotScores = []
    plotMeanScores = []
    totalScore = 0
    record = 0
    intelligentGameAgent = IntelligentGameAgent()
    game = Game()
    while True:
        # 
        stateOld = intelligentGameAgent.getState(game)
        finalMove = intelligentGameAgent.getAction(stateOld)

        # Based on the move made, a reward, game end status, and current points are assigned.
        reward, done, score = game.play(finalMove)

        # Assign a new game state
        newState = intelligentGameAgent.getState(game)

        # train short memory
        intelligentGameAgent.trainShortMemory(stateOld, finalMove, reward, newState, done)

        # remember
        intelligentGameAgent.remember(stateOld, finalMove, reward, newState, done)

        if done:
            # train long memory, plot result
            game.reset()
            intelligentGameAgent.n_games += 1
            intelligentGameAgent.trainLongMemory()

            if score > record:
                record = score
                intelligentGameAgent.model.save() # calling the function that records the results
            print(game.x, game.y)
            print('Game', intelligentGameAgent.n_games, 'Score', score, 'Record:', record)

            plotScores.append(score)
            totalScore += score
            # average of the points obtained
            mean_score = totalScore / intelligentGameAgent.n_games
            plotMeanScores.append(mean_score)
            plot(plotScores, plotMeanScores)


if __name__ == '__main__':
    train()