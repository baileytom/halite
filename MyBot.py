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

"""

This script loads in the model and collects training data.

"""


model_path = 'model/state'

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

# Set up policy
policy_net = PolicyNet()

# Load existing / save new model
try:
    policy_net.load_state_dict(torch.load(model_path))
except:
    torch.save(policy_net.state_dict(), model_path)
    
# Set up variables
f = open("data/batch_data", "a")
last_halite = 0
last_action = None
last_state = None
t = 0

# Game loop
game = hlt.Game()
game.ready("MyPythonBot")

while True:
    game.update_frame()
    me = game.me
    game_map = game.game_map
    commands = []

    for ship in me.get_ships():
        # Record data from last turn
        last_reward = me.halite_amount - last_halite
        if last_action:
            f.write("{}|{}|{}|{}|\n".format(
                t,
                last_reward,
                last_action,
                last_state
            ))

        # Forward pass NN for this turns action
        vision = get_vision(ship, 1)
        state = torch.FloatTensor(vision)
        state = Variable(state)
        probs = policy_net(state)
        m = Categorical(probs)
        sample = int(m.sample())

        # Do action
        possible_actions = [
            ship.move(Direction.North),
            ship.move(Direction.South),
            ship.move(Direction.East),
            ship.move(Direction.West),
            ship.move(Direction.Still)
        ]
        action = possible_actions[sample]
        commands.append(action)

        last_state = state
        last_action = action
        last_halite = me.halite_amount
        t += 1
        
    if len(me.get_ships()) == 0:
        commands.append(me.shipyard.spawn())
        
    game.end_turn(commands) 
    


