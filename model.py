from __future__ import print_function, division

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(561, 561)
        self.fc2 = nn.Linear(561, 512)
        # self.dropout = nn.Dropout(p=0.2)
        self.out_layer = nn.Linear(512, 12)
        self.out_act = nn.Softmax()

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        # x = self.dropout(x)
        x = self.out_layer(x)
        # x = torch.softmax(x, dim=1)
        # output = self.out_act(x)

        return x
