# Import libraries
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.spatial import distance

class ReplayBuffer():
    """
    Description: Enabling the agent to sample non correlated samples from the data, as 1 step TD is highly unstable.
    Input:  - max_size: The maximal size of the interaction archive
            - input_shape: The length of the state representation
    """
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        
        self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        
    def store_transition(self, state, action, state_):
        """ Store (s, a, s(t+1)) tuple in interaction archive. """
        index = self.mem_cntr % self.mem_size
        
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.new_state_memory[index] = state_
        self.mem_cntr += 1 
        
        return index

    def sample_buffer(self, batch_size, reward_cntr):
        """ Sample from the obtained interactions stored in the agent memory. """
        max_mem = min(reward_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]

        return states, actions, rewards, new_states

    
class DDQNetwork(nn.Module):
    """ 
    Description: Class containing the Dueling Deep Q Network, able to retrieve both state and advantage values.
    Input:  - n_actions: The number of variables which the agent can determine
            - input_dims: The dimensions of the input to the network, i.e. length of the state representation
            - fc1_dims: The dimensions of the first fully-connected neural layer
            - fc2_dims: The dimensions of the second fully-connected neural layer
    """
    def __init__(self, n_actions, input_dims, fc1_dims, fc2_dims):
        super(DDQNetwork, self).__init__()
        self.lin_1 = nn.Linear(input_dims, fc1_dims)
        self.lin_2 = nn.Linear(fc1_dims, fc2_dims)
        self.V = nn.Linear(fc2_dims, 1)
        self.A = nn.Linear(fc2_dims, n_actions)
    
    def call(self, state, req_grad=True):
        """ Conduct forward pass of the features through the network, return Q-value """
        state_T = torch.Tensor(state)
        
        # For training, keep track of gradients
        if req_grad:
            x = F.relu(self.lin_1(state_T))
            x = F.relu(self.lin_2(x))
            
            V = self.V(x)
            A = self.A(x)
            
            Q = (V + (A - torch.mean(A)))
            return Q
        
        # If not training does not track gradients, as only forward pass is required
        else:
            with torch.no_grad():
                x = F.relu(self.lin_1(state_T))
                x = F.relu(self.lin_2(x))

                V = self.V(x).numpy()
                A = self.A(x).numpy()

                Q = (V + (A - np.mean(A)))
                return Q
    
    def advantage(self, state):
        """ Conduct forward pass only for the advantage values """
        state_T = torch.Tensor(state)

        with torch.no_grad():
            x = F.relu(self.lin_1(state_T))
            x = F.relu(self.lin_2(x))
            
            A = self.A(x).numpy()
            return A    
    

class Agent(): 
    """
    Description: Hold all functionality regarding the functioning of the agent
    Input:  - rfc: The Reward Function Code, defining which reward function to use
            - lr: The learning rate applied in training the neural network used within the agent
            - gamma: The discounting factor to 'de-value' later rewards
            - actions: Lists for every one of the three variables that can be adjusted by the agent
            - batch_size: The batch size used in training the neural network
            - input_dims: The dimensions of the input to the network, i.e. length of the state representation
            - epsilon: The pobability of selecting a random action by the agent
            - eps_dec_lin: The amount for linear epsilon decay over consecutive episodes
            - eps_dec_exp: The factor of the exponential decaying epsilon over consecutive episodes
            - eps_end: The lowest value of epsilon, at which the decrementation stops
            - mem_size: The maximal size of the interactions (memory) archive
            - fname: The name of the file in which the learned weights of the neural network dictating the policy of the agent will be saved
            - fc1_dims: The dimensions of the first fully-connected neural layer
            - fc2_dims: The dimensions of the second fully-connected neural layer
            - replace: The amount of episodes between updates of the behavioral policy network with the weights of the value network.
    """
    def __init__(self, lr, gamma, actions, batch_size, input_dims, 
                 epsilon=1.0, eps_dec_lin=1e-4, eps_dec_exp=0.998, 
                 eps_end=1e-2, mem_size=100000, fname='DDQN.h5', 
                 fc1_dims=32, fc2_dims=64, replace=100):
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
        self.memory = ReplayBuffer(mem_size, input_dims)
        
        # Create entire action space and decide number of actions
        self.actions = self.create_action_space(as1=actions[0], as2=actions[1])
        n_actions = len(self.actions)
        self.action_space = [i for i in range(n_actions)]
        
        # Initialize Online Evaluation Network
        self.q_eval = DDQNetwork(n_actions, input_dims, fc1_dims, fc2_dims)

        # Initialize Target Network for action selection
        self.q_next = DDQNetwork(n_actions, input_dims, fc1_dims, fc2_dims)
        
        # Set criterion and optimizer
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.q_eval.parameters(), lr=lr, momentum=0.9)
        
        # Reward tracker for reference induced reward calculation
        self.reward_reference = []
    
    def create_action_space(self, as1, as2):
        """ Create the global action space using all combinations of the individual action spaces  """
        action_space = []
        for a1 in as1:
            for a2 in as2:
                action_space.append((a1, a2))
        return action_space
    
    def save_model(self, fname=None):
        """ Save the model weights to the given file name """
        if fname == None:
            torch.save(self.q_next.state_dict(), os.path.join(os.getcwd() + '/Model Saves', self.fname))
        else:
            torch.save(self.q_next.state_dict(), os.path.join(os.getcwd() + '/Model Saves', fname))
        
    def load_model(self, fname=None):
        """ Load the model weights from the given file name """
        if fname == None:
            self.q_eval.load_state_dict(torch.load(os.path.join(os.getcwd() + '/Model Saves', self.fname)))
            self.q_next.load_state_dict(torch.load(os.path.join(os.getcwd() + '/Model Saves', self.fname)))
        else:
            self.q_eval.load_state_dict(torch.load(os.path.join(os.getcwd() + '/Model Saves', fname)))
            self.q_next.load_state_dict(torch.load(os.path.join(os.getcwd() + '/Model Saves', fname)))
    
    def store_transition(self, state, action, new_state):
        """ Store memory of the agent """
        return self.memory.store_transition(state, action, new_state)
    
    def store_reward(self, performance, indeces):
        """ Store episodic reward of the agent """
        self.memory.reward_memory[indeces] = performance
        self.reward_counter += len(indeces)
        
    def choose_action(self, observation):        
        """ 
        Description: Choose action (epsilon-greedily) based on the observation of the current state of the environment.
        Input:  - observation: State representation summarizing the current state the agent is in
        """
        # With probability epsilon, choose random action
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        
        # Else choose action greedily based on advantage values from network
        else:
            state = np.array([observation])
            actions = self.q_eval.advantage(state)

            action = np.argmax(actions)

        return action  
    
    def retrieve_operator(self, action):
        """ Return the selected action """
        return self.actions[action]        
    
    def learn(self):
        """ 
        Description: Let the agent learn from experience and replace target network if threshold is met.
        """
        # Skip learning if insufficient memory is present to sample a batch of the given size
        if self.reward_counter < self.batch_size:
            return 
        
        # After k learning steps replace target network with online evaluation network
        if self.learn_step_counter % self.replace == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
        
        # Sample from memory
        states, actions, rewards, states_ = self.memory.sample_buffer(batch_size=self.batch_size, 
                                                                      reward_cntr=self.reward_counter)  
        
        # Conduct forward pass through the evaluation network for state value actions
        q_pred = self.q_eval.call(states, req_grad=False)
        
        # Conduct forward pass through the next state network to retrieve value of best next action
        q_next = np.max(self.q_next.call(states_, req_grad=False), axis=1)
        
        # Copy the predicted state value figures
        q_target = np.copy(q_pred)
        
        # Loop through all states
        for idx in range(len(states)):
            # Calculate target values for taken actions
            q_target[idx, actions[idx]] = rewards[idx] + self.gamma * q_next[idx]
        
        # Convert q_target into torch Tensor
        q_target = torch.Tensor(q_target)
        
        # Set NN to train mode and zerograd the parameters
        self.q_eval.train()
        self.optimizer.zero_grad()
        
        # Conduct forward pass of the states through the network
        out = self.q_eval.call(states)
        
        # Calculate the loss compared to the target values and backpropagate the loss
        loss = self.criterion(out, q_target)
        loss.backward()
        
        # Update the parameters
        self.optimizer.step()

        # Increment the learn step counter, to keep track when to update the target network
        self.learn_step_counter += 1
        
    def epsilon_decay_exponential(self, idx):
        """ Decay epsilon exponentially over consecutive episodes """
        self.epsilon = max(self.eps_start * (self.eps_dec_exp ** (idx-1)), self.eps_end)
    
    def epsilon_linear_decay(self):
        """ Decay epsilon linearly over consecutive episodes """
        self.epsilon = max((self.epsilon - self.eps_dec_lin), self.eps_end)
    
    def normalize(self, val, LB, UB, clip=True):
        """ Apply (bounded) normalization on the given value using the given bounds (LB, UB) """
        if clip:
            return min(max((val - LB)/(UB - LB), 0.0), 1.0)
        else:
            return (val - LB)/(UB - LB)
        
    def create_state_representation(self, optim, gen_nr, hv, pareto_size):
        """ 
        Description: Summarize and normalize current state values into a single vector
        Input:  - optim: The class object of the optimization model, containing relevant information
                - gen_nr: The current generation number
                - hv: Hypervolume indicator obtained by the current population
                - pareto_size: The number of individuals comprising the pareto front
                - population: Current population used to create offspring
        """        
        # Retrieve and normalize population figures
        log = optim.logbook[gen_nr - 1]
        avgs, opts, stds = log['avg'], log['min'], log['std']
        norm_avgs = [self.normalize(val=avgs[obj], LB=optim.val_bounds[obj][0], UB=optim.val_bounds[obj][1]) for obj in range(0, optim.NOBJ)]
        norm_opts = [self.normalize(val=opts[obj], LB=optim.val_bounds[obj][0], UB=optim.val_bounds[obj][1]) for obj in range(0, optim.NOBJ)]
        norm_stds = [self.normalize(val=stds[obj], LB=optim.std_bounds[obj][0], UB=optim.std_bounds[obj][1]) for obj in range(0, optim.NOBJ)]
        
        # Return state representation
        return np.array([gen_nr/optim.NGEN, 
                         min(1.0, optim.stagnation_counter/10),  # Stagnation > 10 not relevant how much more
                         np.mean(norm_avgs),
                         np.mean(norm_opts),
                         np.mean(norm_stds),
                         hv,
                         pareto_size/optim.POP_SIZE]).flatten()
    
    def retrieve_memory(self):
        """ Return the agent memory """
        # Retrieve Agent Memory
        state_memory = self.memory.state_memory
        new_state_memory = self.memory.new_state_memory
        action_memory = self.memory.action_memory
        reward_memory = self.memory.reward_memory
        
        return state_memory, new_state_memory, action_memory, reward_memory
