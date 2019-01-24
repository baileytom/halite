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
data_path = 'data/batch_data'

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

def navigate_dir(ship, destination):
    # Naive navigate for now
    return game.game_map.naive_navigate(ship, destination)

# Set up policy

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

# Policy utilities

def select_action(state):
    state = torch.FloatTensor(state)
    state = Variable(state)
    probs, state_value = policy_net(state)
    m = Categorical(probs)
    action_sample = m.sample()
    action = [
            ship.move(Direction.North),
            ship.move(Direction.South),
            ship.move(Direction.East),
            ship.move(Direction.West),
            ship.move(Direction.Still),
            ship.move(navigate_dir(ship, shipyard))
        ][int(action_sample)]
    return action, m.log_prob(action_sample), state_value

policy_net = PolicyNet()

# Load existing / save new model
try:
    policy_net.load_state_dict(torch.load(model_path))
except:
    torch.save(policy_net.state_dict(), model_path)
    
# Set up variables
last_halite = 0
last_action_prob = None
last_state = None
t = 0

# Game loop
game = hlt.Game()
game.ready("MyPythonBot")

while True:
    game.update_frame()
    me = game.me
    game_map = game.game_map
    shipyard = me.shipyard.position
    commands = []

    for ship in me.get_ships():
        # Record data from last turn
        last_reward = me.halite_amount - last_halite
        if last_state:
            #out_state = np.array2string(np.asarray(last_state)).replace('\n', ''
            with open(data_path, "a") as f:
                # For turn t-1
                #   time, reward, log prob action, state values
                f.write("{}|{}|{}|{}\n".format(
                    t,
                    last_reward,
                    last_action_prob,
                    last_state
                ))
                logging.info("wrote")
                
        # Forward pass NN for this turns action, collect data
        vision = get_vision(ship, 1)
        state = vision + [ship.halite_amount]
        action, action_prob, state_value = select_action(state) 

        # Add action to commands
        commands.append(action)

        # Save data to store with next turn's reward diff
        last_state = state_value
        last_action_prob = action_prob
        last_halite = me.halite_amount
        t += 1
        
    if len(me.get_ships()) == 0:
        commands.append(me.shipyard.spawn())
        
    game.end_turn(commands) 
    


