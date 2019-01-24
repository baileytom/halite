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
    return [cell.halite_amount, cell.is_occupied]

def get_vision(ship, sight_range):    
    map = game.game_map
    sight = []
    for x in range(ship.position.x-sight_range,ship.position.x+sight_range+1):
        for y in range(ship.position.y-sight_range,ship.position.y+sight_range+1):
            sight += cell_data(map[map.normalize(Position(x, y))])
    return sight

def nav_dir(ship, destination):
    # Naive navigate for now
    return game.game_map.naive_navigate(ship, destination)

# Set up policy

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        # The input is 1: our sight (5x5 square) 2: our halite amount
        # Possible alternate inputs: entire map, x pos, y pos, halite amount

        # Single linear transform
        self.l1 = nn.Linear(50, 81)
        self.l2 = nn.Linear(81, 269)
        self.l3 = nn.Linear(269, 81)
        
        # Action out
        self.action_head = nn.Linear(81, 5)

        # Value out
        self.value_head = nn.Linear(81, 1)

    def get_state_value(self, x):
        x = F.relu(self.l1(x))
        return self.value_head(x)
        
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1)

# Policy utilities

def select_action(state):
    state = torch.FloatTensor(state)
    state = Variable(state)
    probs = policy_net(state)
    m = Categorical(probs)
    action_sample = m.sample()
    action = [
            ship.move(Direction.North),
            ship.move(Direction.South),
            ship.move(Direction.East),
            ship.move(Direction.West),
            ship.move(Direction.Still)
        ][int(action_sample)]

    logging.info(m.log_prob(action_sample))
    
    return action, action_sample

policy_net = PolicyNet()

# Load existing / save new model
try:
    policy_net.load_state_dict(torch.load(model_path))
except:
    torch.save(policy_net.state_dict(), model_path)
    
# Set up variables
ship_data = {}
last_halite = 0
last_action_sample = None
last_state = None
t = 0

# If ship position = shipyard && ship.halite_amount = 0
#    reward = ship.last_halite amount - ship.halite amount

# Game loop
game = hlt.Game()
game.ready("MyPythonBot")

while True:
    t += 1
    game.update_frame()
    me = game.me
    game_map = game.game_map
    shipyard = me.shipyard.position
    commands = []

    for ship in me.get_ships():
        if ship.id in ship_data:
            continue
        # last state, last action sample, last position, go home, last halite
        ship_data[ship.id] = [None, None, None, False, 0]

    for sid, values in ship_data.items():
        if not values:
            continue
        
        last_state, last_action_sample, last_position, go_home, last_halite = values
        
        # Calculate reward
        last_reward = 0
        if not me.has_ship(sid):
            last_reward = -100 # You died.
        else:
            last_reward = (me.get_ship(sid).halite_amount - last_halite)/10
            
        if not last_state or go_home:
            continue
            
        # Write to file
        with open(data_path, "a") as f:
            f.write("{}|{}|{}|{}|{}\n".format(
                t,
                sid,
                last_reward,
                last_action_sample,
                last_state
            ))

        if not me.has_ship(sid):
            ship_data[sid] = None
            
    for ship in me.get_ships():
        state = None
        action = None
        action_sample = None
        go_home = ship_data[ship.id][3]
        
        if ship.is_full:
            ship_data[ship.id][3] = True

        if ship.position == shipyard:
            ship_data[ship.id][3] = False
        
        # Check for go home flag
        if ship_data[ship.id][3]:
            action = nav_dir(ship, shipyard)
        else:
            # Forward pass NN for this turns action, collect data
            state = get_vision(ship, 2)
            action, action_sample = select_action(state) 

        # Add action to commands
        commands.append(action)

        # Collect data
        data = [state, action_sample, ship.position, go_home, ship.halite_amount]
        ship_data[ship.id] = data
        
    if len(me.get_ships()) < 1 and not game_map[me.shipyard.position].is_occupied and me.halite_amount >= 1000:
        commands.append(me.shipyard.spawn())

    game.end_turn(commands) 
    


