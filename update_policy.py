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
from collections import namedtuple
import ast

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

    def get_state_value(self, x):
        x = F.relu(self.l1(x))
        return self.value_head(x)
        
    def forward(self, x):
        x = F.relu(self.l1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1)
    
policy_net = PolicyNet()

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# Load state
policy_net.load_state_dict(torch.load(model_path))

# Read data

rewards = []
actions = []

gamma = 0.99
batch_size = 5

saved_actions = []
policy_losses = []
value_losses = []
saved_rewards = []

# Preprocessing
f = open("data/batch_data", "r")
for line in f.readlines():
    data = line.strip().split("|")
    t, reward, sample, state = data

    # Format typing
    t = int(data[0])
    reward = float(data[1])
    sample = torch.tensor(float(sample))
    state = ast.literal_eval(state)

    # Calculating
    state = torch.FloatTensor(state)
    state = Variable(state)
    probs = policy_net(state)
    m = Categorical(probs)
    log_prob = m.log_prob(sample)
    print(log_prob)
    input()
    
    # Save reward for turn t
    saved_rewards.append(reward)

    # Save action for turn t
    action = SavedAction(log_prob, policy_net.get_state_value(state))
    saved_actions.append(action)
    
    
    
    


