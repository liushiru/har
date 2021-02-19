from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(561, 561)
        self.fc2 = nn.Linear(561, 512)
        # self.dropout = nn.Dropout(p=0.2)
        self.out_layer = nn.Linear(512, 12)
        self.out_act = nn.Softmax()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # x = self.dropout(x)
        x = self.out_layer(x)
        # x = torch.softmax(x, dim=1)
        # output = self.out_act(x)

        return x


    def predict(self, x):
        x = self.forward(x)
        prediction = np.argmax(x.numpy())

        return prediction

class CNN2d(nn.Module):

    def __init__(self):
        super(CNN2d, self).__init__()
        self.conv1 = nn.Conv2d(1, 9, kernel_size=(2, 5))
        self.conv2 = nn.Conv2d(9, 27, kernel_size=(2, 5))
        # self.conv3 = nn.Conv2d(6, 9, kernel_size=(1, 5))
        self.pool1 = nn.MaxPool2d(3, 3)
        # self.fc1 = nn.Linear(1000, 120)
        self.out_layer = nn.Linear(1080, 12)

    def forward(self, x):

        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        x = self.pool1(torch.relu(x))
        x = x.reshape(x.size()[0], -1)
        # x = self.fc1(x)
        # x = torch.relu(x)
        x = self.out_layer(x)
        return x

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(6, 12, kernel_size=5)
        self.conv2 = nn.Conv1d(12, 18, kernel_size=5)
        self.conv3 = nn.Conv1d(18, 24, kernel_size=5)
        self.pool2 = nn.MaxPool1d(3, stride=1)
        self.out_layer = nn.Linear(1200, 12)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        # x = self.pool1(torch.relu(x))
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = x.reshape(x.size()[0], -1)
        x = self.out_layer(x)
        return x
