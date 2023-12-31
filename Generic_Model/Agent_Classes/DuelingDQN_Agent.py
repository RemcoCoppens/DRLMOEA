import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from Agent_Classes.DuelingDQN import DuelingDQN 
from Agent_Classes.ReplayBuffer import ReplayBuffer

        
class Agent:
    #COMMENT: CHANGE THIS TEXT
    """Description: Hold all the variables regarding the functioning of the agent """
    """Input: 
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
        
        # Initiate counters and memory
        self.learn_step_counter = 0
        self.reward_counter = 0
        self.memory = ReplayBuffer(mem_size, input_size)


        # Create entire action space and decide number of actions
        self.actions = self.create_action_space(as1 = actions[0], as2=actions[1], as3=actions[2])
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

    def save_model(self, fname = None):
        """Save the model weights of the target network to the given file."""
        if fname == None:
            torch.save(self.q_target_network.state_dict(), os.path.join(os.getcwd() + '/Model_Saves', self.fname))
        else:
            torch.save(self.q_target_network.state_dict(), os.path.join(os.getcwd() + '/Model_Saves', fname))

    def load_model(self, fname = None):
        """Load the model weights of the target network from the given file."""
        if fname == None:
            self.q_online_network.load_state_dict(torch.load(os.path.join(os.getcwd() + '/Model_Saves', self.fname)))
            self.q_target_network.load_state_dict(torch.load(os.path.join(os.getcwd() + '/Model_Saves', self.fname)))
        else:
            self.q_online_network.load_state_dict(torch.load(os.path.join(os.getcwd() + '/Model_Saves', fname)))    
            self.q_target_network.load_state_dict(torch.load(os.path.join(os.getcwd() + '/Model_Saves', fname)))

    def store_transition(self, state, action, reward, state_):
        """Store the interaction in the memory"""
        return self.memory.store_transition(state, action, reward, state_)

    def epsilon_decay_exponential(self, run):
        """Decay the epsilon value exponentially"""
        self.epsilon = max(selfs.eps_start * self.eps_dec_exp**(run-1), self.eps_end)

    def normalize(self, val, LB, UB, clip=True):
        """ Apply (bounded) normalization on the given value using the given bounds (LB, UB) """
        if clip:
            return min(max((val - LB)/(UB - LB), 0.0), 1.0)
        else:
            return (val - LB)/(UB - LB)
    
    def create_state_representation(self, optim, gen, hv, pareto_size):
        """Create the state features and save as a single vector"""
        """Input: 
                - optim: class object of the optimisation problem
                - gen: current generation number
                - hv: hypervolume of the current generation
                - pareto_size: number of individuals in the pareto front
                """
        log = optim.logbook[gen -1]
        avgs, mins, stds = log.select("avg", "min", "std")

        #normalise values
        norm_avgs = [self.normalize(val = avgs[obj], LB = optim.obj_bounds[obj][0], UB = optim.obj_bounds[obj][1]) for obj in range(0, optim.n_obj)]
        norm_mins = [self.normalize(val = mins[obj], LB = optim.obj_bounds[obj][0], UB = optim.obj_bounds[obj][1]) for obj in range(0, optim.n_obj)]	

        # Create state representation
        state_repre = np.array([gen/optim.NGEN,
                                min(1.0, optim.stagnation_counter/10),
                                np.mean(norm_avgs),
                                np.mean(norm_mins),
                                hv,
                                pareto_size/optim.POP_SIZE]).flatten() 
        
        return state_repre

    def create_action_space(self, as1, as2, as3):
        """Create the action space for the agent using all combinations of the
        indivual action spaces of the three variables."""
        #COMMENT: KIJKEN OF WE COMBINATIES WILLEN OF LOS VAN ELKAAR, Reijnen heeft volgens mij los
        #van elkaar maar dat weet ik niet zeker
        """Input: 
                - as1: The list of possible actions for the crossover distrubution parameter
                - as2: The list of possible actions for the mutation distrubution parameter
                - as3: The list of possible actions for the independent mutation probability
        """
        action_space = []
        for i in as1:
            for j in as2:
                for k in as3:
                    action_space.append([i,j,k])
        return action_space

    def choose_action(self, state):
        """Choose an action based on the current state"""
        """Input: 
                - state: The current state of the agent
        """
        #with probability epsilon, choose an aciton based on advantages values or choose random action
        if np.random.random() > self.epsilon:
            state = torch.tensor(state, dtype=torch.float).to(self.q_online_network.device)
            advantages = self.q_online_network.get_advantages(state)
            action = torch.argmax(advantages).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    
    def learn(self):
        """Learn from experience by a batch of interactions from the memory and replace the target network if needed"""
        #Skip learning if not enough interactions are stored in the memory
        if self.memory.mem_cntr < self.batch_size:
            return
        
        #Replace the target network with the online network if needed
        if self.learn_step_counter % self.replace == 0:
            self.q_target_network.load_state_dict(self.q_online_network.state_dict())

        #Sample a batch from replay buffer
        states, actions, rewards, states_ = self.memory.sample_buffer(self.batch_size)

        #Conduct forward pass of online network
        q_pred = self.q_online_network.forward(states, req_grad = False)
        
        #Conduct forward pass of target network
        q_next = self.q_target_network.forward(states_, req_grad = False)

        #Create target values
        q_target = np.copy(q_pred)

        #Loop through all states
        for idx in range(len(states)):
            #Update target value for the action taken
            q_target[idx, actions[idx]] = rewards[idx] + self.gamma * q_next[idx]

        #Convert to tensor
        q_target = torch.Tensor(q_target)

        #Set NN to training mode and zero gradients
        self.q_online_network.train()
        self.optimizer.zero_grad()

        # Conduct forward pass of the states through the network
        out = self.q_eval.forward(states)
        
        # Calculate the loss compared to the target values and backpropagate the loss
        loss = self.criterion(out, q_target)
        loss.backward()
        
        # Update the parameters
        self.optimizer.step()

        # Increment the learn step counter, to keep track when to update the target network
        self.learn_step_counter += 1
        



        


