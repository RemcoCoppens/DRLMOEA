import numpy as np


class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size # Maximal size of the replay buffer
        self.mem_cntr = 0 # Counter for the replay buffer
        
        self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32) # *input_shape unpacks the tuple
        self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32) # *input_shape unpacks the tuple
        
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64) # Action is an integer
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32) # Reward is a float
        
        #COMMENT: WSS NIET NODIG VOOR MIJN SOORT PROBLEMEN. GEBRUIK JE ALS ER EEN TERMINAL STATE IS
        #WANT NA EEN TERMINAL STATE IS DE FUTURE REWARD ZERO EN MOET ER DUS ANDERS MEE OMGEGAAN WORDEN
        #MISSCHIEN VOOR GENERALISABILITY WEL WEER LATEN STAAN WANT HET KAN MISSCHIEN NIET KWAAD
        #EN WIE WEET HEBBEN WE WEL EEN KEER EEN TERMINAL STATE        
        
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool) # Boolean to indicate if the state is terminal

    def store_transition(self, state, action, reward, state_):
        """ Store the transition in the replay buffer """
        index = self.mem_cntr % self.mem_size # Index of the memory counter
        self.state_memory[index] = state
        self.new_state_memory[index] = state_   
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        #self.terminal_memory[index] = done
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
        #terminal = self.terminal_memory[batch]	

        return states, actions, rewards, states_ #,terminal
    
