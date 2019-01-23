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

# Utilities
def cell_data(cell):
    return [cell.halite_amount, cell.is_occupied, cell.has_structure]

def get_vision(ship, sight_range):    
    map = game.game_map
    sight = []
    for x in range(ship.position.x-sight_range,ship.position.x+sight_range+1):
        for y in range(ship.position.y-sight_range,ship.position.y+sight_range+1):
            sight += cell_data(map[map.normalize(Position(x, y))])
    return sight


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

# Game loop
game = hlt.Game()
game.ready("MyPythonBot")

# Set up policy
policy_net = PolicyNet()

batch_size = 5
learning_rate = 0.01
gamma = 0.99
optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=learning_rate)

state_pool = []
action_pool = []
reward_pool = []
steps = 0

while True:
    game.update_frame()
    me = game.me
    game_map = game.game_map
    commands = []

    for ship in me.get_ships():
        vision = get_vision(ship, 1)

        state = torch.FloatTensor(vision)
        state = Variable(state)
        probs = policy_net(state)
        m = Categorical(probs)
        sample = int(m.sample())

        possible_actions = [
            ship.move(Direction.North),
            ship.move(Direction.South),
            ship.move(Direction.East),
            ship.move(Direction.West),
            ship.move(Direction.Still)
        ]
        action = possible_actions[sample]
        
        commands.append(action)

    if len(me.get_ships()) == 0:
        commands.append(me.shipyard.spawn())
        
    game.end_turn(commands) 
    


