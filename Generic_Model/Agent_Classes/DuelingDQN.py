import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DuelingDQN(nn.Module):
    """Dueling D Q Network"""
    """Description: This class contains the Dueling Deep Q Network, able to retrieve the Q-values and 
       advantages values for a given state"""
    """Input:   - Input_dims: the number of input dimensions, equal to the length of the state representation
                - Output_dims: the number of output dimensions, equal to the number of actions the agent can take
                - FCL1_layer: first fully connected layer
                - FCL2_layer: second fully connected layer 
    """

    def __init__(self, input_size, output_size, FCL1_layer, FCL2_layer):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(input_size, FCL1_layer)
        self.fc2 = nn.Linear(FCL1_layer, FCL2_layer)
        self.V = nn.Linear(FCL2_layer, 1)
        self.A = nn.Linear(FCL2_layer, output_size)
    

    def forward(self, state, req_grad = True):
        state = torch.Tensor(state)
        
        #For training, keep track of gradients, for evaluation, don't
        if req_grad: 
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))   
    
            value = self.V(x)	
            advantage = self.A(x)


            q_values = (value + (advantage - torch.mean(advantage)))
            return q_values
        
        else: 
            with torch.no_grad():
                x = F.relu(self.fc1(state))
                x = F.relu(self.fc2(x))   
    
                value = self.V(x).numpy()
                advantage = self.A(x).numpy()

                #q_values = (value + (advantage - advantage.mean(dim=1, keepdim=True)))
                q_values= (value+(advantage - np.mean(advantage)))
                return q_values
    
    def get_advantages(self, state):
        """function to retrieve the advantage values for a given state"""
        STATE = torch.Tensor(state)
        with torch.no_grad():
            x = F.relu(self.fc1(STATE))
            x = F.relu(self.fc2(x))   
            advantage = self.A(x)
            return advantage
         