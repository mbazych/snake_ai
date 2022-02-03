import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


# Q Learning algorithm focuses on next action's value (chooses action, does it, gets prize and updates value, which is responsible for future moves.)

class Linear_QNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.linear1 = nn.Linear(inputSize, hiddenSize) # first layer, transmit scores to second hidden layer
        self.linear2 = nn.Linear(hiddenSize, outputSize) # second hidden layer, transmit scores to third, result layer as prediction

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    
    # function used to save highest scores to file
    def save(self, fileName='model.pth'):
        modelFolderPath = './model'
        if not os.path.exists(modelFolderPath):
            os.makedirs(modelFolderPath)

        fileName = os.path.join(modelFolderPath, fileName)
        torch.save(self.state_dict(), fileName) # save scores dict to file


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()


    # function to predict next move
    def train_step(self, state, action, reward, nextState, done):
        state = torch.tensor(state, dtype=torch.float)
        nextState = torch.tensor(nextState, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # function to check if state is multi dimensionsal or single (teaches for one movement and many moevements in learning whole iteration)
        if len(state.shape) == 1:
            # if single dimensional, return single dimensional values
            state = torch.unsqueeze(state, 0)
            nextState = torch.unsqueeze(nextState, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, ) #single value tuple

        # 1. Predicted Q value for current value and observing current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            # choose action and do it
            Q_new = reward[idx]
            if not done[idx]:
                # based on onbservation select new Q (sum of prizes and learning coefficient and the product of the maximum value of the next prediction)
                Q_new = reward[idx] + self.gamma * torch.max(self.model(nextState[idx]))

            # set current maximum value of action as predicted Q
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # otpimize using Adam, delete unused predictions
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred) # (Q_new, Q)
        loss.backward()

        self.optimizer.step()



