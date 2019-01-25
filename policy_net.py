import torch
import torch.nn as nn
import torch.nn.functional as F

# Define model
class PolicyNet(nn.Module):
    model_path = 'model/state'
    
    def __init__(self):
        super(PolicyNet, self).__init__()
        
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

    def load_state(self):
        self.load_state_dict(torch.load(PolicyNet.model_path))

    def save_state(self):
        torch.save(self.state_dict(), PolicyNet.model_path)
        
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1)


