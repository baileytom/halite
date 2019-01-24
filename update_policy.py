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
        # The input is 1: our sight (5x5 square) 2: our halite amount
        # Possible alternate inputs: entire map, x pos, y pos, halite amount

        # Single linear transform
        self.l1 = nn.Linear(28, 81)

        # Action out
        self.action_head = nn.Linear(81, 6)

        # Value out
        self.value_head = nn.Linear(81, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values
    
policy_net = PolicyNet()

# Load state
policy_net.load_state_dict(torch.load(model_path))

# Road data
f = open("data/batch_data", "r")
for line in f.readlines():
    data = line.strip().split("|")
    t, reward, log_prob, state = data

    print(state)
    
    input()


