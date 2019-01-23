import hlt
from hlt import constants
from hlt.positionals import Direction
from hlt.positionals import Position
import random
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
import numpy as np
from itertools import count

model_path = 'model/state'

# Model
class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(27, 81)
        self.fc2 = nn.Linear(81, 81)
        self.fc3 = nn.Linear(81, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

policy_net = PolicyNet()
policy_net.load_state_dict(torch.load(model_path))

batch_size = 5
learning_rate = 0.01
gamma = 0.99
optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=learning_rate)

# Load data

f = open("data/batch_data", "r")
for line in f.readlines():

    print(line)

    input()
    
    data = line.split("|")
    print(data)
    print(len(data))
    turn, reward, move, state = data

