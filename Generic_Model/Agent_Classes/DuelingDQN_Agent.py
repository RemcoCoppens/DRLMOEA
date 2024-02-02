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
    def __init__(self, lr, gamma, actions, batch_size, input_size, replace,
                 epsilon=1.0, eps_dec_lin=1e-4, eps_dec_exp=0.998, 
                 eps_end=1e-2, mem_size=100000, fname='DDQN_test1.h5', 
                 FCL1_layer=32, FCL2_layer=64):
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
        self.q_online_network = DuelingDQN(input_size, n_actions, FCL1_layer, FCL2_layer)

        # Initialize Target Network for action selection
        self.q_target_network = DuelingDQN(input_size, n_actions, FCL1_layer, FCL2_layer)
        
        # Set criterion and optimizer
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.q_online_network.parameters(), lr=lr, momentum=0.9)
        
        # Reward tracker for reference induced reward calculation
        self.reward_reference = []
        self.best_performance = 0

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

    def store_transition(self, state, action, state_):
        """Store the interaction in the memory"""
        return self.memory.store_transition(state, action, state_)
    
    def store_reward(self, performance, runs):
        """Store reward of the agent"""
        self.memory.reward_memory[runs] = performance
        self.reward_counter += len(runs)
        
    def epsilon_decay_exponential(self, run):
        """Decay the epsilon value exponentially"""
        self.epsilon = max(self.eps_start * self.eps_dec_exp**(run-1), self.eps_end)

    def normalize(self, val, LB, UB, clip=True):
        """ Apply (bounded) normalization on the given value using the given bounds (LB, UB) """
        if clip:
            return min(max((val - LB)/(UB - LB), 0.0), 1.0)
        else:
            return (val - LB)/(UB - LB)
        
    def normalizestd(self, std, LB, UB, clip = True):
        if clip:
            return min(max(std/(UB - LB), 0.0), 1.0)
        else:
            return std/(UB - LB)

    def calculate_spacing(self, sorted_pareto_set):
        distances = [np.linalg.norm(sorted_pareto_set[i+1] - sorted_pareto_set[i]) for i in range(len(sorted_pareto_set)-1)]
        avg_distance = np.mean(distances)
        spacing = np.sqrt(np.sum([(avg_distance- distance)**2 for distance in distances]) / (len(sorted_pareto_set) - 1))
        return spacing

    def calculate_hole_relative_size(self, sorted_pareto_set):
        distances = [np.linalg.norm(sorted_pareto_set[i+1] - sorted_pareto_set[i]) for i in range(len(sorted_pareto_set)-1)]
        max_distance = np.max(distances)
        avg_distance = np.mean(distances)
        hole_relative_size = max_distance/ avg_distance
        return hole_relative_size
        


    def create_state_representation(self, optim, gen, hv, pareto_size, pareto_front, sorted_pareto_front, norm_hv, binary_hv, firstder_hv, secondder_hv):
        """Create the state features and save as a single vector"""
        """Input: 
                - optim: class object of the optimisation problem
                - gen: current generation number
                - hv: hypervolume of the current generation
                - pareto_size: number of individuals in the pareto front
                - pareto_front: list of individuals in the pareto front
                - sorted_pareto_front: list of individuals in the pareto front sorted on the 
                """
        log = optim.logbook[gen -1]
        avgs, mins, stds = log['avg'], log['min'], log['std']

        #normalise values
        norm_avgs = [self.normalize(val = avgs[obj], LB = optim.val_bounds[obj][0], UB = optim.val_bounds[obj][1]) for obj in range(0, optim.NBOJ)]
        norm_mins = [self.normalize(val = mins[obj], LB = optim.val_bounds[obj][0], UB = optim.val_bounds[obj][1]) for obj in range(0, optim.NBOJ)]	
        norm_stds = [self.normalizestd(std = stds[obj], LB = optim.val_bounds[obj][0], UB = optim.val_bounds[obj][1]) for obj in range(0, optim.NBOJ)]	
        


        #Spacing and hole relative size
        normalized_sorted_pareto_set = np.array([tuple([self.normalize(val=obj_v[i], 
                                                 LB=optim.val_bounds[i][0], 
                                                 UB=optim.val_bounds[i][1]) for i in range(optim.NBOJ)]) for obj_v in sorted_pareto_front])        
    
        spacing = self.calculate_spacing(normalized_sorted_pareto_set)
        hole_relative_size = self.calculate_hole_relative_size(normalized_sorted_pareto_set)


        # Create state representation first version
        # state_repre = np.array([gen/optim.NGEN,
        #                         min(1.0, optim.stagnation_counter/10),
        #                         np.mean(norm_avgs),
        #                         np.mean(norm_mins),
        #                         hv,
        #                         pareto_size/optim.POP_SIZE]).flatten() 
        
        # Create state representation with all features from previous literature
        # state_repre = np.array([gen/optim.NGEN,
        #                         min(1.0, optim.stagnation_counter/10),
        #                         np.mean(norm_avgs),
        #                         np.mean(norm_mins),
        #                         hv,
        #                         pareto_size/optim.POP_SIZE, #cardinality: simple the number of points in pareto front
        #                         spacing,
        #                         hole_relative_size]).flatten() 
        
        # Create state representation with all features from previous literature INCLUDING STD
        # state_repre = np.array([gen/optim.NGEN,
        #                         min(1.0, optim.stagnation_counter/10),
        #                         np.mean(norm_avgs),
        #                         np.mean(norm_mins),
        #                         np.mean(norm_stds),
        #                         hv,
        #                         pareto_size/optim.POP_SIZE, #cardinality: simple the number of points in pareto front
        #                         spacing,
        #                         hole_relative_size]).flatten() 

        #REPLACE HV WITH NEW STATE FEATURES
        #normalised HV #9
        state_repre = np.array([gen/optim.NGEN,
                                min(1.0, optim.stagnation_counter/10),
                                np.mean(norm_avgs),
                                np.mean(norm_mins),
                                np.mean(norm_stds),
                                norm_hv,
                                pareto_size/optim.POP_SIZE, #cardinality: simple the number of points in pareto front
                                spacing,
                                hole_relative_size]).flatten() 

        #Binary HV and change in HV #11
        # state_repre = np.array([gen/optim.NGEN,
        #                         min(1.0, optim.stagnation_counter/10),
        #                         np.mean(norm_avgs),
        #                         np.mean(norm_mins),
        #                         np.mean(norm_stds),
        #                         binary_hv,
        #                         firstder_hv,
        #                         secondder_hv,
        #                         pareto_size/optim.POP_SIZE, #cardinality: simple the number of points in pareto front
        #                         spacing,
        #                         hole_relative_size]).flatten() 



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
            state = np.array([state])
            advantages = self.q_online_network.get_advantages(state)
            action = np.argmax(advantages)
        else:
            action = np.random.choice(self.action_space)
        return action

    def retrieve_operator(self, action):
        """ Return the selected action """
        return self.actions[action] 

    def learn(self):
        """Learn from experience by a batch of interactions from the memory and replace the target network if needed"""
        #Skip learning if not enough interactions are stored in the memory
        if self.reward_counter < self.batch_size:
            return
        
        #Replace the target network with the online network if needed
        if self.learn_step_counter % self.replace == 0:
            self.q_target_network.load_state_dict(self.q_online_network.state_dict())

        #Sample a batch from replay buffer
        states, actions, rewards, states_ = self.memory.sample_buffer(self.batch_size)

        #Conduct forward pass of online network
        q_pred = self.q_online_network.forward(states, req_grad = False)
 
        #Conduct forward pass of target network
        q_next = np.max(self.q_target_network.forward(states_, req_grad = False), axis = 1 )

        #Create target values
        q_target = np.copy(q_pred)
          
        for idx in range(len(states)):
            #Update target value for the action taken
            q_target[idx, actions[idx]] = rewards[idx] + self.gamma * q_next[idx]

        #Convert to tensor
        q_target = torch.Tensor(q_target)

        #Set NN to training mode and zero gradients
        self.q_online_network.train()
        self.optimizer.zero_grad()

        # Conduct forward pass of the states through the network
        out = self.q_online_network.forward(states)
        
        # Calculate the loss compared to the target values and backpropagate the loss
        loss = self.criterion(out, q_target) #LOSS(INPUT,target)
        loss.backward()
        
        # Update the parameters
        self.optimizer.step()

        # Increment the learn step counter, to keep track when to update the target network
        self.learn_step_counter += 1
        



        


