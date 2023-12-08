import torch
import torch.nn as nn
import torch.nn.functional as F

import ReplayBuffer 
import DuelingDQN as DDQN



        
class Agent:
    """Description: Hold all the variables regarding the functioning of the agent """
    """Input:  - rfc: The Reward Function Code, defining which reward function to use
            - lr: The learning rate applied in training the neural network used within the agent
            - gamma: The discounting factor to 'de-value' later rewards
            - actions: Lists for every one of the three variables that can be adjusted by the agent
            - batch_size: The batch size used in training the neural network
            - input_size: The dimensions of the input to the network, i.e. length of the state representation
            - epsilon: The pobability of selecting a random action by the agent
            - eps_dec_lin: The amount for linear epsilon decay over consecutive episodes
            - eps_dec_exp: The factor of the exponential decaying epsilon over consecutive episodes
            - eps_end: The lowest value of epsilon, at which the decrementation stops
            - mem_size: The maximal size of the interactions (memory) archive
            - fname: The name of the file in which the learned weights of the neural network dictating the policy of the agent will be saved
            - FCL1_layer: The dimensions of the first fully-connected neural layer
            - FCL2_layer: The dimensions of the second fully-connected neural layer
            - replace: The amount of episodes between updates of the behavioral policy network with the weights of the value network.
    """
def __init__(self, lr, gamma, actions, batch_size, input_size, 
                 epsilon=1.0, eps_dec_lin=1e-4, eps_dec_exp=0.998, 
                 eps_end=1e-2, mem_size=100000, fname='DDQN.h5', 
                 FCL1_layer=32, FCL2_layer=64, replace=100):
        # Initiate agent characteristics
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_start = epsilon
        self.eps_dec_lin = eps_dec_lin
        self.eps_dec_exp = eps_dec_exp
        self.eps_end = eps_end
        self.fname = fname
        self.replace = replace
        self.batch_size = batch_size
        
        # Initiate counters and replay buffer
        self.learn_step_counter = 0
        self.reward_counter = 0
        self.memory = ReplayBuffer(mem_size)
        
        # Create entire action space and decide number of actions
        self.actions = self.create_action_space(as1=actions[0], as2=actions[1])
        n_actions = len(self.actions)
        self.action_space = [i for i in range(n_actions)]
        
        # Initialize Online Evaluation Network
        self.q_online_network = DuelingDQN(n_actions, input_size, FCL1_layer, FCL2_layer)

        # Initialize Target Network for action selection
        self.q_target_network = DuelingDQN(n_actions, input_size, FCL1_layer, FCL2_layer)
        
        # Set criterion and optimizer
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.q_online_network.parameters(), lr=lr, momentum=0.9)
        
        # Reward tracker for reference induced reward calculation
        self.reward_reference = []



