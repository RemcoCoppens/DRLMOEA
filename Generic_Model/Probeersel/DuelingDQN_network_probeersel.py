import os 
import numpy as np 
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size # Maximal size of the replay buffer
        self.mem_cntr = 0 # Counter for the replay buffer
        
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32) # *input_shape unpacks the tuple
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32) # *input_shape unpacks the tuple
        
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64) # Action is an integer
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32) # Reward is a float
        
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool) # Boolean to indicate if the state is terminal

    def store_transition(self, state, action, reward, state_, done):
        """ Store the transition in the replay buffer """
        index = self.mem_cntr % self.mem_size # Index of the memory counter
        self.state_memory[index] = state
        self.new_state_memory[index] = state_   
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

        return index
    
    def sample_buffer(self, batch_size):
        """ Sample a batch from the replay buffer """
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]	

        return states, actions, rewards, states_, terminal
    

class DuelingDQN(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, name)

        self.fcl == nn.Linear(*input_dims, 512)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr) # Adam optimizer
        self.loss = nn.MSELoss() # Mean squared error loss function
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') # Use GPU if available, else CPU
        self.to(self.device) # Send to device
    
    def forward(self, state):
        """ Forward pass of the neural network """
        x = F.relu(self.fcl(state))
        V = self.V(x)
        A = self.A(x)

        return V, A
    
    def save_checkpoint(self): 
        """ Save the model parameters """
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self): 
        """ Load the model parameters """
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, mem_size, batch_size, eps_min=0.01, eps_dec=5e-7, chkpt_dir='tmp/dqn'):
        """Description of the agent"""
        """
        INPUTS:
            Gamma: Discount factor for rewards
            Epsilon: Exploration rate (probability of taking a random action)
            LR: Learning rate of the neural network
            n_actions: Number of actions in the environment
            input_dims: Input dimensions of the neural network
            mem_size: Size of the replay buffer (memory)
            batch_size: Size of the batch to sample from the replay buffer
            eps_min: Minimum exploration rate (probability of taking a random action)
            eps_dec: Exploration rate decay (probability of taking a random action)
            chkpt_dir: Directory to save the model parameters
            """ 
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(self.n_actions)]

        self.memory = ReplayBuffer(mem_size, input_dims) # Initialize replay buffer

        self.q_eval = DuelingDQN(self.lr, self.n_actions, input_dims=self.input_dims, name='q_eval', chkpt_dir=self.chkpt_dir) # Initialize the neural network
        self.q_next = DuelingDQN(self.lr, self.n_actions, input_dims=self.input_dims, name='q_next', chkpt_dir=self.chkpt_dir)

        def choose_action(self, observation):
            """ Choose an action based on the observation """
            if np.random.random() > self.epsilon:
                state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
                _, advantage = self.q_eval.forward(state)
                action = T.argmax(advantage).item()
            else:
                action = np.random.choice(self.action_space)

            return action
        
        def store_transition(self, state, action, reward, state_, done):
            """ Store the transition in the replay buffer """
            self.memory.store_transition(state, action, reward, state_, done)

        def replace_target_network(self):
            """ Replace the target network with the evaluation network """
            if self.learn_step_counter % self.replace_target_cnt == 0:
                self.q_next.load_state_dict(self.q_eval.state_dict())
        
        def decrement_epsilon(self):
            """ Decrement the exploration rate """
            self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

        def save_model(self):
            """ Save the model parameters """
            self.q_eval.save_checkpoint()
            self.q_next.save_checkpoint()
        
        def load_model(self):
            """ Load the model parameters """
            self.q_eval.load_checkpoint()
            self.q_next.load_checkpoint()
        
        def learn(self):
            """ Learn from the experience in the replay buffer """
            # Check if there are enough samples in the replay buffer
            if self.memory.mem_cntr < self.batch_size:
                return
            
            self.q_eval.optimizer.zero_grad() # Reset the gradients of the neural network
            self.replace_target_network()

            state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size) # Sample a batch from the replay buffer

            states = T.tensor(state).to(self.q_eval.device) 
            rewards = T.tensor(reward).to(self.q_eval.device)
            dones = T.tensor(done).to(self.q_eval.device)
            actions = T.tensor(action).to(self.q_eval.device)
            states_ = T.tensor(new_state).to(self.q_eval.device)

            indices = np.arange(self.batch_size)

            
            V_s, A_s = self.q_eval.forward(states) 	
            V_s_, A_s_ = self.q_next.forward(states_) 

            
            V_s_eval, A_s_eval = self.q_eval.forward(states_) 
            V_s_next, A_s_next = self.q_next.forward(states_)

            q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim = True)))[indices, actions]

            q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim = True)))

            q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim = True)))

            max_actions = T.argmax(q_eval, dim=1)

            q_next[dones] = 0.0
            q_target = reward + self.gamma*q_next[indices, max_actions]

            loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
            loss.backward()
            self.q_eval.optimizer.step()
            self.learn_step_counter += 1

            self.decrement_epsilon()



